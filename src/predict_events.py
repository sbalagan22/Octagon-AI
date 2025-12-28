import json
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

"""
v10 Prediction Model
Uses comprehensive pre-fight features:
- Offensive + Defensive metrics
- Layoff & Activity
- Cardio proxies
- Style encoding
- Reach as modifier
"""

def normalize_name(name):
    if not name:
        return ""
    return name.lower().replace("'", "").replace("-", " ").strip()

def parse_height(val):
    try:
        if pd.isnull(val): return 175.0
        s = str(val)
        if '.' in s:
            parts = s.split('.')
            feet = int(parts[0])
            inches = int(parts[1]) if len(parts) > 1 else 0
            return feet * 30.48 + inches * 2.54
        return float(val) if float(val) > 100 else 175.0
    except:
        return 175.0

def parse_fight_time(t_str, r):
    try:
        parts = str(t_str).split(':')
        sec = int(parts[0])*60 + int(parts[1])
        return (int(r)-1)*300 + sec
    except:
        return 900

def load_data():
    print("Loading historical data...")
    fights = pd.read_csv('../newdata/Fights.csv')
    events = pd.read_csv('../newdata/Events.csv')
    fighters = pd.read_csv('../newdata/Fighters.csv')
    
    fights = fights.merge(events[['Event_Id', 'Date']], on='Event_Id', how='left')
    fights['Date'] = pd.to_datetime(fights['Date'])
    fights = fights.sort_values('Date').reset_index(drop=True)
    
    return fights, fighters

def build_fighter_history(fights_df, cutoff_date=None):
    """Build comprehensive history for all fighters"""
    print("Building fighter histories...")
    history = {}
    
    seen_fights = set()
    for _, row in fights_df.iterrows():
        f1 = row['Fighter_1']
        f2 = row['Fighter_2']
        fight_date = row['Date']
        
        # Deduplicate (e.g., Gaethje vs Fiziev duplicates in data)
        # Also handle cases where fighters are swapped
        f_pair = tuple(sorted([f1, f2]))
        fight_key = f_pair + (fight_date,)
        if fight_key in seen_fights: continue
        seen_fights.add(fight_key)

        # Only include fights that happened before the cutoff (useful for predictions)
        if cutoff_date and fight_date >= cutoff_date:
            continue
            
        def ensure_history(name):
            if name not in history:
                history[name] = {
                    'wins': 0, 'losses': 0,
                    'str_landed': 0, 'str_absorbed': 0,
                    'kd': 0, 'td': 0, 'ctrl': 0, 'sub': 0,
                    'time': 0, 'fights': 0,
                    'finishes': 0, 'late_rounds': 0,
                    'distance_pct': [], 'ground_pct': [],
                    'fight_dates': [],
                    'fight_results': [],  # Track individual W/L for recent form
                    'streak': 0
                }
        
        ensure_history(f1)
        ensure_history(f2)
        
        # Parse stats
        try:
            str1 = float(row.get('STR_1', 0) or 0)
            str2 = float(row.get('STR_2', 0) or 0)
            kd1 = float(row.get('KD_1', 0) or 0)
            td1 = float(row.get('TD_1', 0) or 0)
            ctrl1 = float(row.get('Ctrl_1', 0) or 0)
            sub1 = float(row.get('Sub. Att_1', 0) or 0)
            kd2 = float(row.get('KD_2', 0) or 0)
            td2 = float(row.get('TD_2', 0) or 0)
            ctrl2 = float(row.get('Ctrl_2', 0) or 0)
            sub2 = float(row.get('Sub. Att_2', 0) or 0)
            dist1 = float(row.get('Distance_%_1', 0.33) or 0.33)
            ground1 = float(row.get('Ground_%_1', 0.33) or 0.33)
            dist2 = float(row.get('Distance_%_2', 0.33) or 0.33)
            ground2 = float(row.get('Ground_%_2', 0.33) or 0.33)
            fight_sec = parse_fight_time(row.get('Fight_Time', '5:00'), row.get('Round', 3))
            fight_round = int(row.get('Round', 3))
            method = str(row.get('Method', ''))
        except:
            str1 = str2 = kd1 = kd2 = td1 = td2 = ctrl1 = ctrl2 = sub1 = sub2 = 0
            dist1 = dist2 = ground1 = ground2 = 0.33
            fight_sec = 900
            fight_round = 3
            method = ''
        
        # Update F1
        h1 = history[f1]
        h1['str_landed'] += str1
        h1['str_absorbed'] += str2
        h1['kd'] += kd1
        h1['td'] += td1
        h1['ctrl'] += ctrl1
        h1['sub'] += sub1
        h1['time'] += fight_sec
        h1['fights'] += 1
        h1['distance_pct'].append(dist1)
        h1['ground_pct'].append(ground1)
        h1['fight_dates'].append(fight_date)
        if fight_round >= 3: h1['late_rounds'] += 1
        
        if row['Result_1'] == 'W':
            h1['wins'] += 1
            h1['streak'] = max(1, h1['streak'] + 1) if h1['streak'] >= 0 else 1
            h1['fight_results'].append('W')
            if 'KO' in method or 'TKO' in method or 'SUB' in method:
                h1['finishes'] += 1
        else:
            h1['losses'] += 1
            h1['streak'] = min(-1, h1['streak'] - 1) if h1['streak'] <= 0 else -1
            h1['fight_results'].append('L')
        
        # Update F2
        h2 = history[f2]
        h2['str_landed'] += str2
        h2['str_absorbed'] += str1
        h2['kd'] += kd2
        h2['td'] += td2
        h2['ctrl'] += ctrl2
        h2['sub'] += sub2
        h2['time'] += fight_sec
        h2['fights'] += 1
        h2['distance_pct'].append(dist2)
        h2['ground_pct'].append(ground2)
        h2['fight_dates'].append(fight_date)
        if fight_round >= 3: h2['late_rounds'] += 1
        
        if row['Result_2'] == 'W':
            h2['wins'] += 1
            h2['streak'] = max(1, h2['streak'] + 1) if h2['streak'] >= 0 else 1
            h2['fight_results'].append('W')
            if 'KO' in method or 'TKO' in method or 'SUB' in method:
                h2['finishes'] += 1
        else:
            h2['losses'] += 1
            h2['streak'] = min(-1, h2['streak'] - 1) if h2['streak'] <= 0 else -1
            h2['fight_results'].append('L')
    
    return history

def get_fighter_stats(name, history, fighters_df, current_date=None):
    """Get comprehensive pre-fight stats"""
    # Fuzzy match
    h = history.get(name)
    if h is None:
        normalized = normalize_name(name)
        for key in history.keys():
            if normalize_name(key) == normalized:
                h = history[key]
                break
    
    if h is None:
        h = {
            'wins': 0, 'losses': 0, 'str_landed': 0, 'str_absorbed': 0,
            'kd': 0, 'td': 0, 'ctrl': 0, 'sub': 0, 'time': 0, 'fights': 0,
            'finishes': 0, 'late_rounds': 0, 'distance_pct': [], 'ground_pct': [],
            'fight_dates': [], 'streak': 0
        }
    
    # Physical stats from Fighters.csv (but don't override calculated record!)
    match = fighters_df[fighters_df['Full Name'] == name]
    if match.empty:
        match = fighters_df[fighters_df['Full Name'].apply(normalize_name) == normalize_name(name)]
    
    if not match.empty:
        # Use first match that has reach data (more likely to be correct fighter)
        for _, m in match.iterrows():
            reach_val = pd.to_numeric(m.get('Reach'), errors='coerce')
            if not pd.isnull(reach_val):
                height = parse_height(m.get('Ht.'))
                reach = reach_val
                # Prefer record from Fighters.csv if it's more complete (greater experience)
                f_wins = pd.to_numeric(m.get('W'), errors='coerce')
                f_losses = pd.to_numeric(m.get('L'), errors='coerce')
                if not pd.isnull(f_wins) and not pd.isnull(f_losses):
                    # For established fighters, Fighters.csv has their full career record
                    # We use whichever is higher to be safe
                    h['wins'] = max(int(h['wins']), int(f_wins))
                    h['losses'] = max(int(h['losses']), int(f_losses))
                break
        else:
            height = parse_height(match.iloc[0].get('Ht.'))
            reach = height * 1.025
    else:
        height = 175.0
        reach = 180.0
    
    # Calculate rates
    t = max(h['time'], 60) / 60.0
    
    # Layoff
    if current_date and h['fight_dates']:
        last_fight = max(h['fight_dates'])
        layoff = (current_date - last_fight).days
    else:
        layoff = 365
    
    # Activity
    if current_date and h['fight_dates']:
        cutoff_12m = current_date - pd.Timedelta(days=365)
        fights_12m = sum(1 for d in h['fight_dates'] if d > cutoff_12m)
    else:
        fights_12m = 0
    
    # Recent form (last 5 fights, newest first)
    fight_results = h.get('fight_results', [])
    # Reverse to newest first
    recent_form = '-'.join(list(reversed(fight_results))[:5]) if fight_results else 'N/A'
    
    return {
        'wins': h['wins'],
        'losses': h['losses'],
        'winrate': h['wins'] / max(h['wins'] + h['losses'], 1),
        'experience': h['wins'] + h['losses'],
        'streak': h.get('streak', 0),
        'height': height,
        'reach': reach,
        'slpm': h['str_landed'] / t if t > 0 else 0,
        'sapm': h['str_absorbed'] / t if t > 0 else 0,
        'str_diff': (h['str_landed'] - h['str_absorbed']) / t if t > 0 else 0,
        'kd_rate': h['kd'] / (t/15) if t > 0 else 0,
        'td_rate': h['td'] / (t/15) if t > 0 else 0,
        'ctrl_rate': h['ctrl'] / (t/15) if t > 0 else 0,
        'sub_rate': h['sub'] / (t/15) if t > 0 else 0,
        'layoff': layoff,
        'fights_12m': fights_12m,
        'finish_rate': h['finishes'] / max(h['wins'], 1) if h['wins'] > 0 else 0.5,
        'late_round_pct': h['late_rounds'] / max(h['fights'], 1),
        'distance_pct': np.mean(h['distance_pct']) if h['distance_pct'] else 0.33,
        'ground_pct': np.mean(h['ground_pct']) if h['ground_pct'] else 0.33,
        'recent_form': recent_form
    }

def predict():
    print("Loading v10 Global Model...")
    try:
        model = joblib.load('../models/ufc_v10_global.pkl')
    except Exception as e:
        print(f"Model load failed: {e}")
        return
    
    print("Loading upcoming events...")
    try:
        with open('../upcoming_events.json', 'r') as f:
            events = json.load(f)
    except:
        print("../upcoming_events.json not found.")
        return
    
    fights_df, fighters_df = load_data()
    # Today's date for layoff/activity
    today = pd.Timestamp.now()
    
    # Build history up to today
    fighter_history = build_fighter_history(fights_df, cutoff_date=today)
    print(f"  Built history for {len(fighter_history)} fighters")
    
    current_date = pd.Timestamp.now()
    
    # Process events
    for event in events:
        print(f"Predicting for Event: {event['event_name']}")
        for fight in event['fights']:
            f1_name = fight['fighter_1']
            f2_name = fight['fighter_2']
            
            s1 = get_fighter_stats(f1_name, fighter_history, fighters_df, current_date)
            s2 = get_fighter_stats(f2_name, fighter_history, fighters_df, current_date)
            
            # Build v10 features (must match training order)
            reach_diff = s1['reach'] - s2['reach']
            
            row = [
                # Career
                s1['winrate'] - s2['winrate'],
                s1['experience'] - s2['experience'],
                s1['streak'] - s2['streak'],
                # Offensive
                s1['slpm'] - s2['slpm'],
                s1['kd_rate'] - s2['kd_rate'],
                s1['td_rate'] - s2['td_rate'],
                s1['ctrl_rate'] - s2['ctrl_rate'],
                s1['sub_rate'] - s2['sub_rate'],
                # Defensive
                s2['sapm'] - s1['sapm'],  # Lower is better
                s1['str_diff'] - s2['str_diff'],
                # Activity
                s2['layoff'] - s1['layoff'],  # Lower is better
                s1['fights_12m'] - s2['fights_12m'],
                # Cardio
                s1['finish_rate'] - s2['finish_rate'],
                s1['late_round_pct'] - s2['late_round_pct'],
                # Style
                s1['distance_pct'] - s2['distance_pct'],
                s1['ground_pct'] - s2['ground_pct'],
                # Physical (reach as modifier)
                s1['height'] - s2['height'],
                reach_diff * s1['distance_pct'],  # reach_x_distance
                reach_diff * s1['slpm'] / 10 if s1['slpm'] > 0 else 0  # reach_x_volume
            ]
            
            # Predict
            prob = model.predict_proba([row])[0]
            prob_f1 = prob[1] if len(prob) > 1 else prob[0]
            prob_f2 = 1 - prob_f1
            
            winner = f1_name if prob_f1 > prob_f2 else f2_name
            confidence = max(prob_f1, prob_f2)
            
            fight['prediction'] = {
                'winner': winner,
                'confidence': f"{confidence*100:.1f}%",
                'odds': {
                    f1_name: f"{prob_f1*100:.1f}%",
                    f2_name: f"{prob_f2*100:.1f}%"
                },
                'factors': {
                    f1_name: {
                        'slpm': round(s1['slpm'], 2),
                        'sapm': round(s1['sapm'], 2),
                        'str_diff': round(s1['str_diff'], 2),
                        'kd_rate': round(s1['kd_rate'], 2),
                        'td_rate': round(s1['td_rate'], 2),
                        'ctrl_rate': round(s1['ctrl_rate'], 2),
                        'sub_rate': round(s1['sub_rate'], 2),
                        'wins': int(s1['wins']),
                        'losses': int(s1['losses']),
                        'win_rate': round(s1['winrate']*100, 1),
                        'height_cm': round(s1['height'], 1),
                        'reach_cm': round(s1['reach'], 1),
                        'layoff_days': int(s1['layoff']),
                        'recent_form': s1['recent_form']
                    },
                    f2_name: {
                        'slpm': round(s2['slpm'], 2),
                        'sapm': round(s2['sapm'], 2),
                        'str_diff': round(s2['str_diff'], 2),
                        'kd_rate': round(s2['kd_rate'], 2),
                        'td_rate': round(s2['td_rate'], 2),
                        'ctrl_rate': round(s2['ctrl_rate'], 2),
                        'sub_rate': round(s2['sub_rate'], 2),
                        'wins': int(s2['wins']),
                        'losses': int(s2['losses']),
                        'win_rate': round(s2['winrate']*100, 1),
                        'height_cm': round(s2['height'], 1),
                        'reach_cm': round(s2['reach'], 1),
                        'layoff_days': int(s2['layoff']),
                        'recent_form': s2['recent_form']
                    }
                }
            }
    
    # Save
    output_path = '../upcoming_events_with_predictions.json'
    webapp_path = '../web-app/public/upcoming_events.json'
    
    with open(output_path, 'w') as f:
        json.dump(events, f, indent=2)
    print(f"Saved predictions to {output_path}")
    
    import shutil
    try:
        shutil.copy(output_path, webapp_path)
        print(f"Auto-copied to {webapp_path}")
    except Exception as e:
        print(f"Could not copy to web app: {e}")

if __name__ == "__main__":
    predict()
