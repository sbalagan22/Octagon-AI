import pandas as pd
import numpy as np
import joblib
import os
import unicodedata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_name(name):
    if not name:
        return ""
    # Remove accents/diacritics
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
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

def get_mov_class(row, is_flipped=False):
    """
    Classes:
    0: F1 KO/TKO, 1: F1 SUB, 2: F1 DEC
    3: F2 KO/TKO, 4: F2 SUB, 5: F2 DEC
    """
    method = str(row.get('Method', '')).upper()
    result1 = row['Result_1']
    result2 = row['Result_2']
    
    # Normalize results if flipped
    if is_flipped:
        winner = 2 if result1 == 'W' else 1
    else:
        winner = 1 if result1 == 'W' else 2
        
    category = None
    if 'KO' in method or 'TKO' in method:
        category = 0 # KO
    elif 'SUB' in method:
        category = 1 # SUB
    elif 'DEC' in method:
        category = 2 # DEC
    else:
        return None # Ignore DQ, CNC, etc.
        
    if winner == 1:
        return category
    else:
        return category + 3

def train_mov():
    print("="*70)
    print("MOV MODEL: MULTI-CLASS VICTORY PREDICTION")
    print("="*70)
    
    data_dir = os.path.join(BASE_DIR, '..', 'newdata')
    fights = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    events = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    fighters = pd.read_csv(os.path.join(data_dir, 'Fighters.csv'))
    
    fights = fights.merge(events[['Event_Id', 'Date']], on='Event_Id', how='left')
    fights['Date'] = pd.to_datetime(fights['Date'])
    fights = fights.sort_values('Date').reset_index(drop=True)
    
    fighters['Height_cm'] = fighters['Ht.'].apply(parse_height)
    fighters['Reach_cm'] = pd.to_numeric(fighters['Reach'], errors='coerce').fillna(0)
    fighters.loc[fighters['Reach_cm'] == 0, 'Reach_cm'] = fighters.loc[fighters['Reach_cm'] == 0, 'Height_cm'] * 1.025
    
    fighter_history = {}
    rows = []
    
    for idx, row in fights.iterrows():
        f1 = row['Fighter_1']
        f2 = row['Fighter_2']
        fight_date = row['Date']
        
        def get_history(name):
            if name not in fighter_history:
                fighter_history[name] = {
                    'wins': 0, 'losses': 0, 'str_landed': 0, 'str_absorbed': 0,
                    'kd': 0, 'td': 0, 'ctrl': 0, 'sub': 0, 'time': 0, 'fights': 0,
                    'finishes': 0, 'decisions': 0, 'late_rounds': 0,
                    'distance_pct': [], 'clinch_pct': [], 'ground_pct': [],
                    'fight_dates': [], 'streak': 0
                }
            return fighter_history[name]
        
        h1 = get_history(f1)
        h2 = get_history(f2)
        
        # Features (Same as v10)
        f1_match = fighters[fighters['Full Name'] == f1]
        f2_match = fighters[fighters['Full Name'] == f2]
        h1_cm = f1_match['Height_cm'].values[0] if len(f1_match) > 0 else 175.0
        r1_cm = f1_match['Reach_cm'].values[0] if len(f1_match) > 0 else 180.0
        h2_cm = f2_match['Height_cm'].values[0] if len(f2_match) > 0 else 175.0
        r2_cm = f2_match['Reach_cm'].values[0] if len(f2_match) > 0 else 180.0
        
        t1 = max(h1['time'], 60) / 60.0
        t2 = max(h2['time'], 60) / 60.0
        slpm_1 = h1['str_landed'] / t1 if t1 > 0 else 0
        slpm_2 = h2['str_landed'] / t2 if t2 > 0 else 0
        sapm_1 = h1['str_absorbed'] / t1 if t1 > 0 else 0
        sapm_2 = h2['str_absorbed'] / t2 if t2 > 0 else 0
        str_diff_1 = slpm_1 - sapm_1
        str_diff_2 = slpm_2 - sapm_2
        kd_rate_1 = h1['kd'] / (t1/15) if t1 > 0 else 0
        kd_rate_2 = h2['kd'] / (t2/15) if t2 > 0 else 0
        td_rate_1 = h1['td'] / (t1/15) if t1 > 0 else 0
        td_rate_2 = h2['td'] / (t2/15) if t2 > 0 else 0
        ctrl_rate_1 = h1['ctrl'] / (t1/15) if t1 > 0 else 0
        ctrl_rate_2 = h2['ctrl'] / (t2/15) if t2 > 0 else 0
        sub_rate_1 = h1['sub'] / (t1/15) if t1 > 0 else 0
        sub_rate_2 = h2['sub'] / (t2/15) if t2 > 0 else 0
        
        def count_recent(dates, current, m):
            cutoff = current - pd.Timedelta(days=m*30)
            return sum(1 for d in dates if d > cutoff)
        
        layoff_1 = (fight_date - max(h1['fight_dates'])).days if h1['fight_dates'] else 365
        layoff_2 = (fight_date - max(h2['fight_dates'])).days if h2['fight_dates'] else 365

        f12_1 = count_recent(h1['fight_dates'], fight_date, 12)
        f12_2 = count_recent(h2['fight_dates'], fight_date, 12)
        
        finish_1 = h1['finishes'] / max(h1['wins'], 1) if h1['wins'] > 0 else 0.5
        finish_2 = h2['finishes'] / max(h2['wins'], 1) if h2['wins'] > 0 else 0.5
        
        late_1 = h1['late_rounds'] / max(h1['fights'], 1)
        late_2 = h2['late_rounds'] / max(h2['fights'], 1)

        dist1 = np.mean(h1['distance_pct']) if h1['distance_pct'] else 0.33
        dist2 = np.mean(h2['distance_pct']) if h2['distance_pct'] else 0.33
        ground1 = np.mean(h1['ground_pct']) if h1['ground_pct'] else 0.33
        ground2 = np.mean(h2['ground_pct']) if h2['ground_pct'] else 0.33
        
        winrate_1 = h1['wins'] / max(h1['wins'] + h1['losses'], 1)
        winrate_2 = h2['wins'] / max(h2['wins'] + h2['losses'], 1)
        
        reach_diff = r1_cm - r2_cm

        # Data mapping
        mov_class = get_mov_class(row)
        if mov_class is not None:
            pre_fight = {
                'winrate_diff': winrate_1 - winrate_2,
                'experience_diff': (h1['fights']) - (h2['fights']),
                'streak_diff': h1['streak'] - h2['streak'],
                'slpm_diff': slpm_1 - slpm_2,
                'kd_rate_diff': kd_rate_1 - kd_rate_2,
                'td_rate_diff': td_rate_1 - td_rate_2,
                'ctrl_rate_diff': ctrl_rate_1 - ctrl_rate_2,
                'sub_rate_diff': sub_rate_1 - sub_rate_2,
                'sapm_diff': sapm_2 - sapm_1,
                'str_differential_diff': str_diff_1 - str_diff_2,
                'layoff_diff': layoff_2 - layoff_1,
                'activity_12m_diff': f12_1 - f12_2,
                'finish_rate_diff': finish_1 - finish_2,
                'late_round_pct_diff': late_1 - late_2,
                'distance_style_diff': dist1 - dist2,
                'ground_style_diff': ground1 - ground2,
                'height_diff': h1_cm - h2_cm,
                'reach_x_distance': reach_diff * dist1,
                'reach_x_volume': reach_diff * slpm_1 / 10 if slpm_1 > 0 else 0,
                'target': mov_class
            }
            rows.append(pre_fight)
            
            # Flipped
            reach_diff_flipped = r2_cm - r1_cm
            mov_class_flipped = get_mov_class(row, is_flipped=True)
            pre_fight_flipped = {
                'winrate_diff': winrate_2 - winrate_1,
                'experience_diff': (h2['fights']) - (h1['fights']),
                'streak_diff': h2['streak'] - h1['streak'],
                'slpm_diff': slpm_2 - slpm_1,
                'kd_rate_diff': kd_rate_2 - kd_rate_1,
                'td_rate_diff': td_rate_2 - td_rate_1,
                'ctrl_rate_diff': ctrl_rate_2 - ctrl_rate_1,
                'sub_rate_diff': sub_rate_2 - sub_rate_1,
                'sapm_diff': sapm_1 - sapm_2,
                'str_differential_diff': str_diff_2 - str_diff_1,
                'layoff_diff': layoff_1 - layoff_2,
                'activity_12m_diff': f12_2 - f12_1,
                'finish_rate_diff': finish_2 - finish_1,
                'late_round_pct_diff': late_2 - late_1,
                'distance_style_diff': dist2 - dist1,
                'ground_style_diff': ground2 - ground1,
                'height_diff': h2_cm - h1_cm,
                'reach_x_distance': reach_diff_flipped * dist2,
                'reach_x_volume': reach_diff_flipped * slpm_2 / 10 if slpm_2 > 0 else 0,
                'target': mov_class_flipped
            }
            rows.append(pre_fight_flipped)



        # Update History
        try:
            str1 = float(row.get('STR_1', 0) or 0)
            str2 = float(row.get('STR_2', 0) or 0)
            kd1 = float(row.get('KD_1', 0) or 0)
            td1 = float(row.get('TD_1', 0) or 0)
            sub1 = float(row.get('Sub. Att_1', 0) or 0)
            kd2 = float(row.get('KD_2', 0) or 0)
            td2 = float(row.get('TD_2', 0) or 0)
            sub2 = float(row.get('Sub. Att_2', 0) or 0)
            dist1 = float(row.get('Distance_%_1', 0.33) or 0.33)
            dist2 = float(row.get('Distance_%_2', 0.33) or 0.33)
            ground1 = float(row.get('Ground_%_1', 0.33) or 0.33)
            ground2 = float(row.get('Ground_%_2', 0.33) or 0.33)
            time = parse_fight_time(row.get('Fight_Time', '5:00'), row.get('Round', 3))
            method = str(row.get('Method', ''))
        except:
            str1=str2=kd1=kd2=td1=td2=sub1=sub2=0
            dist1=dist2=ground1=ground2=0.33
            time=900
            method=''

        h1['str_landed'] += str1
        h1['str_absorbed'] += str2
        h1['kd'] += kd1
        h1['td'] += td1
        h1['sub'] += sub1
        h1['time'] += time
        h1['fights'] += 1
        h1['distance_pct'].append(dist1)
        h1['ground_pct'].append(ground1)
        h1['fight_dates'].append(fight_date)
        if row['Result_1'] == 'W':
            h1['wins'] += 1
            h1['streak'] = max(1, h1['streak'] + 1) if h1['streak'] >= 0 else 1
            if 'KO' in method or 'TKO' in method or 'SUB' in method:
                h1['finishes'] += 1
        else:
            h1['losses'] += 1
            h1['streak'] = min(-1, h1['streak'] - 1) if h1['streak'] <= 0 else -1
            
        h2['str_landed'] += str2
        h2['str_absorbed'] += str1
        h2['kd'] += kd2
        h2['td'] += td2
        h2['sub'] += sub2
        h2['time'] += time
        h2['fights'] += 1
        h2['distance_pct'].append(dist2)
        h2['ground_pct'].append(ground2)
        h2['fight_dates'].append(fight_date)
        if row['Result_2'] == 'W':
            h2['wins'] += 1
            h2['streak'] = max(1, h2['streak'] + 1) if h2['streak'] >= 0 else 1
            if 'KO' in method or 'TKO' in method or 'SUB' in method:
                h2['finishes'] += 1
        else:
            h2['losses'] += 1
            h2['streak'] = min(-1, h2['streak'] - 1) if h2['streak'] <= 0 else -1

    df = pd.DataFrame(rows)
    features = [
        'winrate_diff', 'experience_diff', 'streak_diff', 'slpm_diff', 'kd_rate_diff',
        'td_rate_diff', 'ctrl_rate_diff', 'sub_rate_diff', 'sapm_diff', 'str_differential_diff',
        'layoff_diff', 'activity_12m_diff', 'finish_rate_diff', 'late_round_pct_diff',
        'distance_style_diff', 'ground_style_diff', 'height_diff', 'reach_x_distance', 'reach_x_volume'
    ]
    
    X = df[features].fillna(0)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"MOV Model Accuracy: {acc:.1%}")
    
    model_path = os.path.join(BASE_DIR, '..', 'models', 'ufc_v10_mov.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Saved to {model_path}")

if __name__ == "__main__":
    train_mov()
