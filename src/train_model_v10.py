import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from datetime import datetime

"""
v10 Model: COMPREHENSIVE PRE-FIGHT ANALYSIS

Improvements over v9:
1. DEFENSIVE METRICS: SApM, Strike Differential
2. LAYOFF & ACTIVITY: Days since last fight, fights in 12/24 months
3. CARDIO PROXIES: Avg fight duration, finish rate, late round %
4. STYLE ENCODING: Distance/Clinch/Ground fighter classification
5. REACH NORMALIZATION: Reach as modifier, not raw feature
6. WEIGHT CLASS SPECIFIC: Train separate models per division
"""

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
    """Return total fight time in seconds"""
    try:
        parts = str(t_str).split(':')
        sec = int(parts[0])*60 + int(parts[1])
        return (int(r)-1)*300 + sec
    except:
        return 900

def train_v10():
    print("="*70)
    print("v10 MODEL: COMPREHENSIVE PRE-FIGHT ANALYSIS")
    print("="*70)
    
    print("\nLoading data...")
    fights = pd.read_csv('../newdata/Fights.csv')
    events = pd.read_csv('../newdata/Events.csv')
    fighters = pd.read_csv('../newdata/Fighters.csv')
    
    # Merge dates
    fights = fights.merge(events[['Event_Id', 'Date']], on='Event_Id', how='left')
    fights['Date'] = pd.to_datetime(fights['Date'])
    fights = fights.sort_values('Date').reset_index(drop=True)
    
    print(f"Total fights: {len(fights)}")
    
    # Prepare fighters data
    fighters['Height_cm'] = fighters['Ht.'].apply(parse_height)
    fighters['Reach_cm'] = pd.to_numeric(fighters['Reach'], errors='coerce').fillna(0)
    fighters.loc[fighters['Reach_cm'] == 0, 'Reach_cm'] = fighters.loc[fighters['Reach_cm'] == 0, 'Height_cm'] * 1.025
    
    # Build comprehensive history
    print("\nBuilding comprehensive fighter histories...")
    
    fighter_history = {}
    rows = []
    
    for idx, row in fights.iterrows():
        f1 = row['Fighter_1']
        f2 = row['Fighter_2']
        fight_date = row['Date']
        
        # Initialize history
        def get_history(name):
            if name not in fighter_history:
                fighter_history[name] = {
                    'wins': 0, 'losses': 0, 
                    'str_landed': 0, 'str_absorbed': 0,
                    'kd': 0, 'td': 0, 'ctrl': 0, 'sub': 0,
                    'time': 0, 'fights': 0,
                    'finishes': 0, 'decisions': 0,
                    'late_rounds': 0,  # Fights that went to round 3+
                    'distance_pct': [], 'clinch_pct': [], 'ground_pct': [],
                    'fight_dates': [],
                    'streak': 0
                }
            return fighter_history[name]
        
        h1 = get_history(f1)
        h2 = get_history(f2)
        
        # Get physical stats
        f1_match = fighters[fighters['Full Name'] == f1]
        f2_match = fighters[fighters['Full Name'] == f2]
        
        h1_cm = f1_match['Height_cm'].values[0] if len(f1_match) > 0 else 175.0
        r1_cm = f1_match['Reach_cm'].values[0] if len(f1_match) > 0 else 180.0
        h2_cm = f2_match['Height_cm'].values[0] if len(f2_match) > 0 else 175.0
        r2_cm = f2_match['Reach_cm'].values[0] if len(f2_match) > 0 else 180.0
        
        # PRE-FIGHT CALCULATIONS
        t1 = max(h1['time'], 60) / 60.0  # minutes
        t2 = max(h2['time'], 60) / 60.0
        
        # === 1. OFFENSIVE & DEFENSIVE METRICS ===
        slpm_1 = h1['str_landed'] / t1 if t1 > 0 else 0
        slpm_2 = h2['str_landed'] / t2 if t2 > 0 else 0
        sapm_1 = h1['str_absorbed'] / t1 if t1 > 0 else 0  # NEW: Strikes absorbed
        sapm_2 = h2['str_absorbed'] / t2 if t2 > 0 else 0
        str_diff_1 = slpm_1 - sapm_1  # NEW: Strike differential
        str_diff_2 = slpm_2 - sapm_2
        
        kd_rate_1 = h1['kd'] / (t1/15) if t1 > 0 else 0
        kd_rate_2 = h2['kd'] / (t2/15) if t2 > 0 else 0
        td_rate_1 = h1['td'] / (t1/15) if t1 > 0 else 0
        td_rate_2 = h2['td'] / (t2/15) if t2 > 0 else 0
        ctrl_rate_1 = h1['ctrl'] / (t1/15) if t1 > 0 else 0
        ctrl_rate_2 = h2['ctrl'] / (t2/15) if t2 > 0 else 0
        sub_rate_1 = h1['sub'] / (t1/15) if t1 > 0 else 0
        sub_rate_2 = h2['sub'] / (t2/15) if t2 > 0 else 0
        
        # === 2. LAYOFF & ACTIVITY ===
        def calc_layoff(fight_dates, current_date):
            if not fight_dates:
                return 365  # Default 1 year for debut
            last_fight = max(fight_dates)
            return (current_date - last_fight).days
        
        def count_recent_fights(fight_dates, current_date, months):
            if not fight_dates:
                return 0
            cutoff = current_date - pd.Timedelta(days=months*30)
            return sum(1 for d in fight_dates if d > cutoff)
        
        layoff_1 = calc_layoff(h1['fight_dates'], fight_date)
        layoff_2 = calc_layoff(h2['fight_dates'], fight_date)
        fights_12m_1 = count_recent_fights(h1['fight_dates'], fight_date, 12)
        fights_12m_2 = count_recent_fights(h2['fight_dates'], fight_date, 12)
        fights_24m_1 = count_recent_fights(h1['fight_dates'], fight_date, 24)
        fights_24m_2 = count_recent_fights(h2['fight_dates'], fight_date, 24)
        
        # === 3. CARDIO & DURABILITY PROXIES ===
        avg_duration_1 = h1['time'] / max(h1['fights'], 1)
        avg_duration_2 = h2['time'] / max(h2['fights'], 1)
        finish_rate_1 = h1['finishes'] / max(h1['wins'], 1) if h1['wins'] > 0 else 0.5
        finish_rate_2 = h2['finishes'] / max(h2['wins'], 1) if h2['wins'] > 0 else 0.5
        late_round_pct_1 = h1['late_rounds'] / max(h1['fights'], 1)
        late_round_pct_2 = h2['late_rounds'] / max(h2['fights'], 1)
        
        # === 4. STYLE ENCODING ===
        def get_avg_pct(pct_list):
            return np.mean(pct_list) if pct_list else 0.33
        
        distance_pct_1 = get_avg_pct(h1['distance_pct'])
        distance_pct_2 = get_avg_pct(h2['distance_pct'])
        clinch_pct_1 = get_avg_pct(h1['clinch_pct'])
        clinch_pct_2 = get_avg_pct(h2['clinch_pct'])
        ground_pct_1 = get_avg_pct(h1['ground_pct'])
        ground_pct_2 = get_avg_pct(h2['ground_pct'])
        
        # === 5. REACH AS MODIFIER ===
        reach_diff = r1_cm - r2_cm
        # Reach * distance fighting = amplified reach advantage for strikers
        reach_x_distance = reach_diff * distance_pct_1
        # Reach * strike rate = reach helps volume strikers
        reach_x_volume = reach_diff * slpm_1 / 10 if slpm_1 > 0 else 0
        
        # === CAREER RECORD ===
        wins_1 = h1['wins']
        wins_2 = h2['wins']
        losses_1 = h1['losses']
        losses_2 = h2['losses']
        winrate_1 = wins_1 / max(wins_1 + losses_1, 1)
        winrate_2 = wins_2 / max(wins_2 + losses_2, 1)
        
        # === BUILD FEATURE ROW (all differentials) ===
        pre_fight = {
            # Career Record
            'winrate_diff': winrate_1 - winrate_2,
            'experience_diff': (wins_1 + losses_1) - (wins_2 + losses_2),
            'streak_diff': h1['streak'] - h2['streak'],
            
            # Offensive
            'slpm_diff': slpm_1 - slpm_2,
            'kd_rate_diff': kd_rate_1 - kd_rate_2,
            'td_rate_diff': td_rate_1 - td_rate_2,
            'ctrl_rate_diff': ctrl_rate_1 - ctrl_rate_2,
            'sub_rate_diff': sub_rate_1 - sub_rate_2,
            
            # Defensive (NEW)
            'sapm_diff': sapm_2 - sapm_1,  # Lower is better, so reversed
            'str_differential_diff': str_diff_1 - str_diff_2,
            
            # Activity (NEW)
            'layoff_diff': layoff_2 - layoff_1,  # Lower layoff is better, so reversed
            'activity_12m_diff': fights_12m_1 - fights_12m_2,
            
            # Cardio (NEW)
            'finish_rate_diff': finish_rate_1 - finish_rate_2,
            'late_round_pct_diff': late_round_pct_1 - late_round_pct_2,
            
            # Style (NEW)
            'distance_style_diff': distance_pct_1 - distance_pct_2,
            'ground_style_diff': ground_pct_1 - ground_pct_2,
            
            # Physical (MODIFIED - reach as modifiers)
            'height_diff': h1_cm - h2_cm,
            'reach_x_distance': reach_x_distance,  # Reach amplified by distance style
            'reach_x_volume': reach_x_volume,      # Reach amplified by volume
            
            # Target
            'target': 1 if row['Result_1'] == 'W' else 0,
            'weight_class': row.get('Weight_Class', 'Unknown')
        }
        
        rows.append(pre_fight)
        
        # === DATA BALANCING: Create flipped version (F2 as "F1") ===
        # This removes positional bias by having 50/50 F1/F2 distribution
        reach_diff_flipped = r2_cm - r1_cm
        reach_x_distance_flipped = reach_diff_flipped * distance_pct_2
        reach_x_volume_flipped = reach_diff_flipped * slpm_2 / 10 if slpm_2 > 0 else 0
        
        pre_fight_flipped = {
            # Career Record (flipped)
            'winrate_diff': winrate_2 - winrate_1,
            'experience_diff': (wins_2 + losses_2) - (wins_1 + losses_1),
            'streak_diff': h2['streak'] - h1['streak'],
            
            # Offensive (flipped)
            'slpm_diff': slpm_2 - slpm_1,
            'kd_rate_diff': kd_rate_2 - kd_rate_1,
            'td_rate_diff': td_rate_2 - td_rate_1,
            'ctrl_rate_diff': ctrl_rate_2 - ctrl_rate_1,
            'sub_rate_diff': sub_rate_2 - sub_rate_1,
            
            # Defensive (flipped)
            'sapm_diff': sapm_1 - sapm_2,
            'str_differential_diff': str_diff_2 - str_diff_1,
            
            # Activity (flipped)
            'layoff_diff': layoff_1 - layoff_2,
            'activity_12m_diff': fights_12m_2 - fights_12m_1,
            
            # Cardio (flipped)
            'finish_rate_diff': finish_rate_2 - finish_rate_1,
            'late_round_pct_diff': late_round_pct_2 - late_round_pct_1,
            
            # Style (flipped)
            'distance_style_diff': distance_pct_2 - distance_pct_1,
            'ground_style_diff': ground_pct_2 - ground_pct_1,
            
            # Physical (flipped)
            'height_diff': h2_cm - h1_cm,
            'reach_x_distance': reach_x_distance_flipped,
            'reach_x_volume': reach_x_volume_flipped,
            
            # Target (flipped - if F1 won original, F2 "wins" the flipped version means target=0)
            'target': 1 if row['Result_2'] == 'W' else 0,
            'weight_class': row.get('Weight_Class', 'Unknown')
        }
        
        rows.append(pre_fight_flipped)
        
        # === UPDATE HISTORY AFTER RECORDING ===
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
            clinch1 = float(row.get('Clinch_%_1', 0.33) or 0.33)
            ground1 = float(row.get('Ground_%_1', 0.33) or 0.33)
            dist2 = float(row.get('Distance_%_2', 0.33) or 0.33)
            clinch2 = float(row.get('Clinch_%_2', 0.33) or 0.33)
            ground2 = float(row.get('Ground_%_2', 0.33) or 0.33)
            
            fight_sec = parse_fight_time(row.get('Fight_Time', '5:00'), row.get('Round', 3))
            fight_round = int(row.get('Round', 3))
            method = str(row.get('Method', ''))
        except:
            str1 = str2 = kd1 = kd2 = td1 = td2 = ctrl1 = ctrl2 = sub1 = sub2 = 0
            dist1 = dist2 = clinch1 = clinch2 = ground1 = ground2 = 0.33
            fight_sec = 900
            fight_round = 3
            method = ''
        
        # Update Fighter 1
        h1['str_landed'] += str1
        h1['str_absorbed'] += str2  # Opponent's strikes = absorbed
        h1['kd'] += kd1
        h1['td'] += td1
        h1['ctrl'] += ctrl1
        h1['sub'] += sub1
        h1['time'] += fight_sec
        h1['fights'] += 1
        h1['distance_pct'].append(dist1)
        h1['clinch_pct'].append(clinch1)
        h1['ground_pct'].append(ground1)
        h1['fight_dates'].append(fight_date)
        if fight_round >= 3:
            h1['late_rounds'] += 1
        
        if row['Result_1'] == 'W':
            h1['wins'] += 1
            h1['streak'] = max(1, h1['streak'] + 1) if h1['streak'] >= 0 else 1
            if 'KO' in method or 'TKO' in method or 'SUB' in method:
                h1['finishes'] += 1
        else:
            h1['losses'] += 1
            h1['streak'] = min(-1, h1['streak'] - 1) if h1['streak'] <= 0 else -1
        
        # Update Fighter 2
        h2['str_landed'] += str2
        h2['str_absorbed'] += str1
        h2['kd'] += kd2
        h2['td'] += td2
        h2['ctrl'] += ctrl2
        h2['sub'] += sub2
        h2['time'] += fight_sec
        h2['fights'] += 1
        h2['distance_pct'].append(dist2)
        h2['clinch_pct'].append(clinch2)
        h2['ground_pct'].append(ground2)
        h2['fight_dates'].append(fight_date)
        if fight_round >= 3:
            h2['late_rounds'] += 1
        
        if row['Result_2'] == 'W':
            h2['wins'] += 1
            h2['streak'] = max(1, h2['streak'] + 1) if h2['streak'] >= 0 else 1
            if 'KO' in method or 'TKO' in method or 'SUB' in method:
                h2['finishes'] += 1
        else:
            h2['losses'] += 1
            h2['streak'] = min(-1, h2['streak'] - 1) if h2['streak'] <= 0 else -1
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    print(f"Built {len(df)} fight records")
    
    # Feature columns
    feature_cols = [
        # Career
        'winrate_diff', 'experience_diff', 'streak_diff',
        # Offensive
        'slpm_diff', 'kd_rate_diff', 'td_rate_diff', 'ctrl_rate_diff', 'sub_rate_diff',
        # Defensive
        'sapm_diff', 'str_differential_diff',
        # Activity
        'layoff_diff', 'activity_12m_diff',
        # Cardio
        'finish_rate_diff', 'late_round_pct_diff',
        # Style
        'distance_style_diff', 'ground_style_diff',
        # Physical
        'height_diff', 'reach_x_distance', 'reach_x_volume'
    ]
    
    nice_names = {
        'winrate_diff': 'Win Rate',
        'experience_diff': 'Experience',
        'streak_diff': 'Current Streak',
        'slpm_diff': 'Strike Volume (SLpM)',
        'kd_rate_diff': 'KO Power',
        'td_rate_diff': 'Takedown Rate',
        'ctrl_rate_diff': 'Control Time',
        'sub_rate_diff': 'Submission Rate',
        'sapm_diff': 'Defense (Low SApM)',
        'str_differential_diff': 'Strike Differential',
        'layoff_diff': 'Ring Rust (Layoff)',
        'activity_12m_diff': 'Recent Activity',
        'finish_rate_diff': 'Finishing Ability',
        'late_round_pct_diff': 'Cardio (Late Rounds)',
        'distance_style_diff': 'Distance Fighter',
        'ground_style_diff': 'Ground Fighter',
        'height_diff': 'Height',
        'reach_x_distance': 'Reach × Distance Style',
        'reach_x_volume': 'Reach × Volume'
    }
    
    # === TRAIN GLOBAL MODEL ===
    print("\n" + "="*70)
    print("GLOBAL MODEL")
    print("="*70)
    
    X = df[feature_cols].fillna(0)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cv_scores = cross_val_score(rf, X, y, cv=5)
    
    print(f"\nTest Accuracy:  {acc:.1%}")
    print(f"CV Accuracy:    {cv_scores.mean():.1%} (+/- {cv_scores.std()*2:.1%})")
    
    imps = rf.feature_importances_
    sorted_idx = np.argsort(imps)[::-1]
    
    print(f"\nGLOBAL FEATURE IMPORTANCE:")
    print("-"*50)
    for i, idx in enumerate(sorted_idx):
        col = feature_cols[idx]
        nice = nice_names.get(col, col)
        print(f"  {i+1:2d}. {nice:28s}: {imps[idx]*100:5.1f}%")
    
    joblib.dump(rf, '../models/ufc_v10_global.pkl')
    
    # === TRAIN WEIGHT-CLASS SPECIFIC MODELS ===
    print("\n" + "="*70)
    print("WEIGHT CLASS SPECIFIC ANALYSIS")
    print("="*70)
    
    weight_groups = {
        'heavyweight': ['Heavyweight', 'Light Heavyweight'],
        'middleweight': ['Middleweight'],
        'welterweight': ['Welterweight'],
        'lightweight': ['Lightweight'],
        'featherweight': ['Featherweight'],
        'bantamweight': ['Bantamweight'],
        'flyweight': ['Flyweight', 'Strawweight', "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight"]
    }
    
    weight_results = {}
    
    for group_name, classes in weight_groups.items():
        mask = df['weight_class'].astype(str).apply(lambda x: any(c in x for c in classes))
        subset = df[mask]
        
        if len(subset) < 200:
            print(f"\n{group_name.upper()}: Skipped (only {len(subset)} fights)")
            continue
        
        X_sub = subset[feature_cols].fillna(0)
        y_sub = subset['target']
        
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2, random_state=42)
        
        rf_sub = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
        rf_sub.fit(X_train, y_train)
        
        acc_sub = accuracy_score(y_test, rf_sub.predict(X_test))
        
        print(f"\n{group_name.upper()} ({len(subset)} fights)")
        print(f"  Accuracy: {acc_sub:.1%}")
        print(f"  Top 5 Factors:")
        
        imps_sub = rf_sub.feature_importances_
        sorted_sub = np.argsort(imps_sub)[::-1]
        
        top_5 = []
        for i, idx in enumerate(sorted_sub[:5]):
            col = feature_cols[idx]
            nice = nice_names.get(col, col)
            print(f"    {i+1}. {nice:25s}: {imps_sub[idx]*100:5.1f}%")
            top_5.append((nice, imps_sub[idx]*100))
        
        weight_results[group_name] = {
            'accuracy': acc_sub,
            'fights': len(subset),
            'top_5': top_5
        }
        
        joblib.dump(rf_sub, f'../models/ufc_v10_{group_name}.pkl')
    
    # === SAVE COMPREHENSIVE REPORT ===
    with open('../feature_importance_v10.txt', 'w') as f:
        f.write("v10 COMPREHENSIVE PRE-FIGHT ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write("NEW FEATURES ADDED:\n")
        f.write("-"*40 + "\n")
        f.write("• Defense: SApM, Strike Differential\n")
        f.write("• Activity: Layoff, Fights in 12 months\n")
        f.write("• Cardio: Finish Rate, Late Round %\n")
        f.write("• Style: Distance/Ground fighter encoding\n")
        f.write("• Reach: Modified to Reach×Distance, Reach×Volume\n\n")
        
        f.write(f"GLOBAL MODEL: {cv_scores.mean():.1%} accuracy\n\n")
        
        f.write("GLOBAL FEATURE IMPORTANCE:\n")
        f.write("-"*40 + "\n")
        for i, idx in enumerate(sorted_idx):
            col = feature_cols[idx]
            nice = nice_names.get(col, col)
            f.write(f"{i+1:2d}. {nice:28s}: {imps[idx]*100:5.1f}%\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("WEIGHT CLASS BREAKDOWN:\n")
        f.write("="*60 + "\n")
        
        for group_name, data in weight_results.items():
            f.write(f"\n{group_name.upper()} ({data['fights']} fights, {data['accuracy']:.1%})\n")
            for i, (feat, pct) in enumerate(data['top_5']):
                f.write(f"  {i+1}. {feat:25s}: {pct:5.1f}%\n")
    
    print("\n" + "="*70)
    print("SAVED:")
    print("  • models/ufc_v10_global.pkl")
    print("  • models/ufc_v10_[weight_class].pkl")
    print("  • feature_importance_v10.txt")
    print("="*70)

if __name__ == "__main__":
    train_v10()
