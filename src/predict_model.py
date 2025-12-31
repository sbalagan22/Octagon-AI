import pandas as pd
import numpy as np
import os
import joblib
import json
import unicodedata
from datetime import datetime
import re
from catboost import CatBoostClassifier

"""
UFC PREDICTION ENGINE
1. Re-builds fighter state (Stats + Glicko).
2. Generates features for upcoming fights.
3. Predicts using trained CatBoost model.
"""

def normalize_name(name):
    if not name: return ""
    import re
    name = str(name)
    nfkd_form = unicodedata.normalize('NFKD', name)
    name = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    name = name.lower().replace("-", " ")
    name = re.sub(r"[^a-zA-Z0-9\s]", "", name)
    name = name.replace(" saint ", " st ").replace(" saint", " st").replace("saint ", "st ")
    return " ".join(name.split())

def parse_height(val):
    try:
        if pd.isnull(val): return 175.0
        s = str(val)
        if "'" in s:
            parts = s.split("'")
            feet = int(parts[0])
            inches = int(parts[1].replace('"', '')) if len(parts) > 1 and parts[1] else 0
            return feet * 30.48 + inches * 2.54
        return float(val) if float(val) > 100 else 175.0
    except: return 175.0

def parse_reach(val):
    try:
        if pd.isnull(val): return 175.0
        return float(val) * 2.54
    except: return 175.0

def get_age(dob_val, fight_date):
    if not dob_val or pd.isnull(dob_val) or dob_val == "": return 29.0
    try:
        dob = pd.to_datetime(dob_val)
        return (fight_date - dob).days / 365.25
    except:
        return 29.0

def is_altitude_location(location):
    if not location: return 0
    high_alt_cities = [
        'salt lake city', 'mexico city', 'denver', 'albuquerque', 
        'bogota', 'quito', 'johannesburg', 'city of mexico'
    ]
    loc_lower = str(location).lower()
    for city in high_alt_cities:
        if city in loc_lower:
            return 1
    return 0

class FighterStats:
    def __init__(self):
        self.total_time_sec = 0
        self.first_fight_date = None
        self.ema_slpm = 0
        self.ema_sapm = 0
        self.ema_td_acc = 0.4
        self.ema_td_avg = 1.0
        self.ema_td_def = 0.5
        self.ema_kd_rate = 0.2
        self.ema_sub_rate = 0.2
        self.ema_ctrl_pct = 10.0
        self.ema_sig_str_acc = 0.45
        self.ema_head_pct = 0.7
        self.ema_body_pct = 0.15
        self.ema_leg_pct = 0.15
        self.ema_dist_pct = 0.8
        self.ema_clinch_pct = 0.1
        self.ema_ground_pct = 0.1
        self.total_kd = 0
        self.total_sub_att = 0
        self.total_ctrl_sec = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.total_fights = 0
        self.streak = 0
        self.recent_form = [] # List of 'W', 'L', 'D'
        self.last_fight_date = None
        self.fight_dates = []
        
    def update(self, result, fight_date, f_time, s_landed, s_absorbed, td_landed, td_att, opp_td_att, opp_td_landed, kd, sub, ctrl,
               sig_acc, head_p, body_p, leg_p, dist_p, clin_p, grou_p):
        self.total_fights += 1
        self.total_time_sec += f_time
        if self.first_fight_date is None:
            self.first_fight_date = fight_date
        self.last_fight_date = fight_date
        self.fight_dates.append(fight_date)
        
        # Current fight metrics
        t_min = f_time / 60.0 if f_time > 0 else 1.0
        f_slpm = s_landed / t_min
        f_sapm = s_absorbed / t_min
        f_td_acc = td_landed / td_att if td_att > 0 else 0.4
        f_td_avg = (td_landed / t_min) * 15.0
        f_td_def = 1.0 - (opp_td_landed / opp_td_att) if opp_td_att > 0 else 0.5
        
        # EMA Update (alpha=0.3)
        alpha = 0.3
        # Rates
        f_kd = (kd / t_min) * 15.0
        f_sub = (sub / t_min) * 15.0
        f_ctrl = (ctrl / f_time) * 100.0 if f_time > 0 else 0
        
        if self.total_fights == 1:
            self.ema_slpm = f_slpm
            self.ema_sapm = f_sapm
            self.ema_td_acc = f_td_acc
            self.ema_td_avg = f_td_avg
            self.ema_td_def = f_td_def
            self.ema_kd_rate = f_kd
            self.ema_sub_rate = f_sub
            self.ema_ctrl_pct = f_ctrl
            self.ema_sig_str_acc = sig_acc
            self.ema_head_pct = head_p
            self.ema_body_pct = body_p
            self.ema_leg_pct = leg_p
            self.ema_dist_pct = dist_p
            self.ema_clinch_pct = clin_p
            self.ema_ground_pct = grou_p
        else:
            self.ema_slpm = alpha * f_slpm + (1 - alpha) * self.ema_slpm
            self.ema_sapm = alpha * f_sapm + (1 - alpha) * self.ema_sapm
            self.ema_td_acc = alpha * f_td_acc + (1 - alpha) * self.ema_td_acc
            self.ema_td_avg = alpha * f_td_avg + (1 - alpha) * self.ema_td_avg
            self.ema_td_def = alpha * f_td_def + (1 - alpha) * self.ema_td_def
            self.ema_kd_rate = alpha * f_kd + (1 - alpha) * self.ema_kd_rate
            self.ema_sub_rate = alpha * f_sub + (1 - alpha) * self.ema_sub_rate
            self.ema_ctrl_pct = alpha * f_ctrl + (1 - alpha) * self.ema_ctrl_pct
            self.ema_sig_str_acc = alpha * sig_acc + (1 - alpha) * self.ema_sig_str_acc
            self.ema_head_pct = alpha * head_p + (1 - alpha) * self.ema_head_pct
            self.ema_body_pct = alpha * body_p + (1 - alpha) * self.ema_body_pct
            self.ema_leg_pct = alpha * leg_p + (1 - alpha) * self.ema_leg_pct
            self.ema_dist_pct = alpha * dist_p + (1 - alpha) * self.ema_dist_pct
            self.ema_clinch_pct = alpha * clin_p + (1 - alpha) * self.ema_clinch_pct
            self.ema_ground_pct = alpha * grou_p + (1 - alpha) * self.ema_ground_pct

        self.total_kd += kd
        self.total_sub_att += sub
        self.total_ctrl_sec += ctrl
        
        if result == 'W':
            self.wins += 1
            self.streak = (self.streak + 1) if self.streak >= 0 else 1
        elif result == 'L':
            self.losses += 1
            self.streak = (self.streak - 1) if self.streak <= 0 else -1
        else: # Draw
            self.draws += 1
            self.streak = 0
        
        self.recent_form.append(result)
        if len(self.recent_form) > 5: self.recent_form.pop(0)

    def get_stat_vector(self, current_date):
        t_min = self.total_time_sec / 60.0
        if t_min < 1: t_min = 1.0
        
        # Derived values for UI/Factors (Can keep career-based for factors, or use EMA)
        # User requested "trajectory", so EMA for everything is better.
        slpm = self.ema_slpm
        sapm = self.ema_sapm
        td_acc = self.ema_td_acc
        td_avg = self.ema_td_avg
        td_def = self.ema_td_def
        
        kd_rate = (self.total_kd / t_min) * 15.0 # Keep these as rates for now
        sub_rate = (self.total_sub_att / t_min) * 15.0
        ctrl_rate = (self.total_ctrl_sec / self.total_time_sec) * 100 if self.total_time_sec > 0 else 0
        
        win_rate = self.wins / self.total_fights if self.total_fights > 0 else 0.5
        
        # Ring Rust
        rust_days = (current_date - self.last_fight_date).days if self.last_fight_date else 365
        
        # Activity
        two_years_ago = current_date - pd.Timedelta(days=730)
        recent_fights = len([d for d in self.fight_dates if d > two_years_ago])
        
        return {
            'slpm': slpm, 'sapm': sapm, 
            'td_acc': self.ema_td_acc, 'td_avg': self.ema_td_avg, 'td_def': self.ema_td_def, 
            'sig_str_acc': self.ema_sig_str_acc,
            'head_pct': self.ema_head_pct, 'body_pct': self.ema_body_pct, 'leg_pct': self.ema_leg_pct,
            'dist_pct': self.ema_dist_pct, 'clinch_pct': self.ema_clinch_pct, 'ground_pct': self.ema_ground_pct,
            'kd_rate': self.ema_kd_rate, 'sub_rate': self.ema_sub_rate, 'ctrl_rate': self.ema_ctrl_pct,
            'exp_time': self.total_time_sec,
            'wins': self.wins, 'losses': self.losses, 'draws': self.draws,
            'streak': self.streak, 'win_rate': win_rate,
            'rust_days': rust_days, 'recent_fights_count': recent_fights,
            'recent_form': "-".join(self.recent_form[::-1]) if self.recent_form else "N/A",
            'recent_win_rate': len([f for f in self.recent_form if f == 'W']) / len(self.recent_form) if self.recent_form else 0.5
        }

def predict_upcoming():
    print("="*60)
    print("STARTING PREDICTION ENGINE (w/ Name Bridge & Advanced Stats)")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../newdata')
    model_dir = os.path.join(base_dir, '../models')
    
    # 1. Load Resources
    print("Loading resources...")
    model = joblib.load(os.path.join(model_dir, 'catboost_ufc_model.pkl'))
    fights = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    fighters_df = pd.read_csv(os.path.join(data_dir, 'Fighters.csv'))
    
    current_glicko = pd.read_csv(os.path.join(data_dir, 'current_glicko.csv')).set_index('Fighter_Id')['Rating'].to_dict()
    current_rd = pd.read_csv(os.path.join(data_dir, 'current_glicko.csv')).set_index('Fighter_Id')['RD'].to_dict()
    
    # Name Bridge (Normalize -> Hashed ID) + Bio Bridge
    name_to_id = {}
    fighter_exp = {}
    fighter_bio = {} # fid -> {height, reach, stance}
    for _, row in fighters_df.iterrows():
        n = normalize_name(row['Full Name'])
        fid = row['Fighter_Id']
        exp = int(row.get('W', 0)) + int(row.get('L', 0))
        if n not in name_to_id or exp > fighter_exp.get(n, -1):
            name_to_id[n] = fid
            fighter_exp[n] = exp
            fighter_bio[fid] = {
                'Height': parse_height(row.get('Height')),
                'Reach': parse_reach(row.get('Reach')),
                'Stance': row.get('Stance', 'Orthodox')
            }
    
    # 2. Re-calculate Stats
    print("Re-calculating current stats...")
    stats_tracker = {}
    
    def pars_time(t_str, r_num):
        try:
            m, s = map(int, t_str.split(':'))
            return (int(r_num)-1)*300 + m*60 + s
        except: return 0
        
    # Load Events to get Dates
    events_df = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    
    # Merge Date into Fights
    fights = fights.merge(events_df[['Event_Id', 'Date']], on='Event_Id', how='left')
    
    # Sort fights chronologically to ensure correct record/history tracking
    fights['Date'] = pd.to_datetime(fights['Date'])
    fights = fights.sort_values('Date').reset_index(drop=True)
    
    for idx, row in fights.iterrows():
        fid1, fid2 = row.get('Fighter_Id_1'), row.get('Fighter_Id_2')
        if pd.isna(fid1) or pd.isna(fid2): continue
        if row['Result_1'] not in ['W', 'L', 'D']: continue
        try:
            time_sec = pars_time(row['Fight_Time'], row['Round'])
            str1_l = float(row.get('STR_1', 0) if pd.notnull(row.get('STR_1')) else 0)
            str2_l = float(row.get('STR_2', 0) if pd.notnull(row.get('STR_2')) else 0)
            
            pct1 = float(row.get('Sig. Str. %_1', 0) if pd.notnull(row.get('Sig. Str. %_1')) else 0.5)
            pct2 = float(row.get('Sig. Str. %_2', 0) if pd.notnull(row.get('Sig. Str. %_2')) else 0.5)
            if pct1 == 0: pct1 = 0.01
            if pct2 == 0: pct2 = 0.01
            str1_a = str1_l / pct1
            str2_a = str2_l / pct2

            td1_l = float(row.get('TD_1', 0) if pd.notnull(row.get('TD_1')) else 0)
            td2_l = float(row.get('TD_2', 0) if pd.notnull(row.get('TD_2')) else 0)
            
            # Additional Stats for Radar
            kd1 = float(row.get('KD_1', 0) if pd.notnull(row.get('KD_1')) else 0)
            kd2 = float(row.get('KD_2', 0) if pd.notnull(row.get('KD_2')) else 0)
            
            sub1 = float(row.get('SUB_1', 0) if pd.notnull(row.get('SUB_1')) else 0)
            sub2 = float(row.get('SUB_2', 0) if pd.notnull(row.get('SUB_2')) else 0)
            
            # Control Time Parsing
            # Format usually "MM:SS" or just "SS"
            def parse_ctrl(val):
                if pd.isnull(val) or val == '--': return 0
                try:
                    if ':' in str(val):
                        m, s = map(int, str(val).split(':'))
                        return m*60 + s
                    return int(val)
                except: return 0
                
            ctrl1 = parse_ctrl(row.get('Ctrl_1'))
            ctrl2 = parse_ctrl(row.get('Ctrl_2'))
            
            # Accuracy & Position
            sig_acc1 = float(row.get('Sig. Str._%_1', 0.45) if pd.notnull(row.get('Sig. Str._%_1')) else 0.45)
            sig_acc2 = float(row.get('Sig. Str._%_2', 0.45) if pd.notnull(row.get('Sig. Str._%_2')) else 0.45)
            
            head1 = float(row.get('Head_%_1', 0.7) if pd.notnull(row.get('Head_%_1')) else 0.7)
            head2 = float(row.get('Head_%_2', 0.7) if pd.notnull(row.get('Head_%_2')) else 0.7)
            body1 = float(row.get('Body_%_1', 0.15) if pd.notnull(row.get('Body_%_1')) else 0.15)
            body2 = float(row.get('Body_%_2', 0.15) if pd.notnull(row.get('Body_%_2')) else 0.15)
            leg1 = float(row.get('Leg_%_1', 0.15) if pd.notnull(row.get('Leg_%_1')) else 0.15)
            leg2 = float(row.get('Leg_%_2', 0.15) if pd.notnull(row.get('Leg_%_2')) else 0.15)
            
            dist1 = float(row.get('Distance_%_1', 0.8) if pd.notnull(row.get('Distance_%_1')) else 0.8)
            dist2 = float(row.get('Distance_%_2', 0.8) if pd.notnull(row.get('Distance_%_2')) else 0.8)
            clin1 = float(row.get('Clinch_%_1', 0.1) if pd.notnull(row.get('Clinch_%_1')) else 0.1)
            clin2 = float(row.get('Clinch_%_2', 0.1) if pd.notnull(row.get('Clinch_%_2')) else 0.1)
            grou1 = float(row.get('Ground_%_1', 0.1) if pd.notnull(row.get('Ground_%_1')) else 0.1)
            grou2 = float(row.get('Ground_%_2', 0.1) if pd.notnull(row.get('Ground_%_2')) else 0.1)

            # KD, SUB, Ctrl
            kd1 = float(row.get('KD_1', 0) if pd.notnull(row.get('KD_1')) else 0)
            kd2 = float(row.get('KD_2', 0) if pd.notnull(row.get('KD_2')) else 0)
            sub1 = float(row.get('SUB_1', 0) if pd.notnull(row.get('SUB_1')) else 0)
            sub2 = float(row.get('SUB_2', 0) if pd.notnull(row.get('SUB_2')) else 0)
            
            def parse_ctrl(val):
                if pd.isnull(val) or val == '--': return 0
                try:
                    if ':' in str(val):
                        m, s = map(int, str(val).split(':'))
                        return m*60 + s
                    return int(val)
                except: return 0
            ctrl1 = parse_ctrl(row.get('Ctrl_1'))
            ctrl2 = parse_ctrl(row.get('Ctrl_2'))
            
            # Using 40% TD accuracy assumption for attempts if missing
            td1_a = td1_l / 0.4 if td1_l > 0 else 0 
            td2_a = td2_l / 0.4 if td2_l > 0 else 0

            if fid1 not in stats_tracker: stats_tracker[fid1] = FighterStats()
            if fid2 not in stats_tracker: stats_tracker[fid2] = FighterStats()
            
            # stats for p1: landed s1, abs s2
            stats_tracker[fid1].update(row['Result_1'], row['Date'], time_sec, str1_l, str2_l, td1_l, td1_a, td2_a, td2_l, kd1, sub1, ctrl1,
                                       sig_acc1, head1, body1, leg1, dist1, clin1, grou1)
            stats_tracker[fid2].update(row['Result_2'], row['Date'], time_sec, str2_l, str1_l, td2_l, td2_a, td1_a, td1_l, kd2, sub2, ctrl2,
                                       sig_acc2, head2, body2, leg2, dist2, clin2, grou2)
        except Exception as e: 
            # print(f"DEBUG: Error at row {idx}: {e}")
            pass

    # 3. Process Upcoming Events
    json_path = os.path.join(base_dir, '../upcoming_events.json')
    if not os.path.exists(json_path):
        print("No upcoming_events.json found.")
        return

    with open(json_path, 'r') as f:
        events = json.load(f)
        
    predictions_out = []
    
    print("Predicting fights...")
    today = datetime.now() # Moved today definition here, outside the event loop
    
    for event in events:
        event_name = event.get('event_name', 'Unknown Event') # use event_name key
        date_str = event.get('date', 'Unknown Date')
        
        is_apex = 1 if ('Fight Night' in str(event_name) and 'Las Vegas' in str(event.get('location', ''))) or 'Apex' in str(event.get('location', '')) else 0
        is_altitude = is_altitude_location(event.get('location', ''))
        
        for fight in event.get('fights', []):
            f1_name = fight.get('fighter_1')
            f2_name = fight.get('fighter_2')
            
            # Try to find IDs
            fid1 = name_to_id.get(normalize_name(f1_name))
            fid2 = name_to_id.get(normalize_name(f2_name))
            
            if not fid1 or not fid2:
                # print(f"Skipping {f1_name} vs {f2_name} - Not in history")
                # Even if not in history, we can predict using Bio features + Base Stats?
                # For high quality, we usually skip, but for app user experience, providing a 50/50 or bio-based prediction is better?
                # Glicko default 1500.
                if not fid1: fid1 = "UNKNOWN_1"
                if not fid2: fid2 = "UNKNOWN_2"
                # continue
                
            # Get Features
            g1 = {'Rating': current_glicko.get(fid1, 1500), 'RD': current_rd.get(fid1, 350)}
            g2 = {'Rating': current_glicko.get(fid2, 1500), 'RD': current_rd.get(fid2, 350)}
            
            # Bio (Prioritize CSV over scraper)
            b1 = fighter_bio.get(fid1, {})
            b2 = fighter_bio.get(fid2, {})
            
            h1 = b1.get('Height', parse_height(fight.get('f1_stats', {}).get('Height', '5.9')))
            h2 = b2.get('Height', parse_height(fight.get('f2_stats', {}).get('Height', '5.9')))
            r1 = b1.get('Reach', parse_reach(fight.get('f1_stats', {}).get('Reach', '70')))
            r2 = b2.get('Reach', parse_reach(fight.get('f2_stats', {}).get('Reach', '70')))
            s1_stance = b1.get('Stance', fight.get('f1_stats', {}).get('Stance', 'Orthodox'))
            s2_stance = b2.get('Stance', fight.get('f2_stats', {}).get('Stance', 'Orthodox'))
            
            st1 = stats_tracker.get(fid1, FighterStats()).get_stat_vector(today)
            st2 = stats_tracker.get(fid2, FighterStats()).get_stat_vector(today)
            
            # Scraper stats for fallback/additional info
            f1_stats_scr = fight.get('f1_stats', {})
            f2_stats_scr = fight.get('f2_stats', {})
            
            # Athletic Age: Years in UFC (to match training proxy)
            f1_debut = stats_tracker.get(fid1, FighterStats()).first_fight_date
            f2_debut = stats_tracker.get(fid2, FighterStats()).first_fight_date
            f1_ath_age = (today - f1_debut).days / 365.25 if f1_debut else 0
            f2_ath_age = (today - f2_debut).days / 365.25 if f2_debut else 0

            # E. BUILD FEATURE ROW (v16: Capped Glicko)
            g_diff = np.clip(g1['Rating'] - g2['Rating'], -250, 250)
            
            features = {
                'glicko_diff': g_diff,
                'glicko_rd_diff': g1['RD'] - g2['RD'],
                'age_diff': f1_ath_age - f2_ath_age,
                'height_diff': h1 - h2,
                'reach_diff': r1 - r2,
                'slpm_diff': st1['slpm'] - st2['slpm'],
                'sapm_diff': st1['sapm'] - st2['sapm'],
                'td_avg_diff': st1['td_avg'] - st2['td_avg'],
                'td_acc_diff': st1['td_acc'] - st2['td_acc'],
                'td_def_diff': st1['td_def'] - st2['td_def'],
                'kd_diff': st1['kd_rate'] - st2['kd_rate'],
                'sub_diff': st1['sub_rate'] - st2['sub_rate'],
                'ctrl_diff': st1['ctrl_rate'] - st2['ctrl_rate'],
                'sig_acc_diff': st1['sig_str_acc'] - st2['sig_str_acc'],
                'head_pct_diff': st1['head_pct'] - st2['head_pct'],
                'body_pct_diff': st1['body_pct'] - st2['body_pct'],
                'leg_pct_diff': st1['leg_pct'] - st2['leg_pct'],
                'dist_pct_diff': st1['dist_pct'] - st2['dist_pct'],
                'clinch_pct_diff': st1['clinch_pct'] - st2['clinch_pct'],
                'ground_pct_diff': st1['ground_pct'] - st2['ground_pct'],
                'ground_pct_diff': st1['ground_pct'] - st2['ground_pct'],
                'exp_diff': (st1['exp_time'] - st2['exp_time']) / 60.0,
                'streak_diff': st1['streak'] - st2['streak'],
                'win_rate_diff': st1['recent_win_rate'] - st2['recent_win_rate'], # Recent over career
                'rust_diff': st1['rust_days'] - st2['rust_days'],
                'activity_diff': st1['recent_fights_count'] - st2['recent_fights_count'],
                'is_apex': is_apex,
                'is_altitude': is_altitude,
                'stance_1': s1_stance,
                'stance_2': s2_stance,
                'weight_class': f1_stats_scr.get('Weight', '155 lbs')
            }
            
            # Fill Nans (Should be handled by logic above, but safety)
            # CatBoost handles NaNs, but passing dict to DataFrame might induce them if keys missing
            
            if fid1 == "UNKNOWN_1" and fid2 == "UNKNOWN_2":
                prob = 0.5
            else:
                df_feat = pd.DataFrame([features])
                prob = model.predict_proba(df_feat)[0][1] # Prob of F1 winning
            
            # 1. Update Fight Object (for JSON)
            # Schema must match FighterModal.tsx: factors, mov, odds
            
            
            # Prioritize Scraped Record if available
            def parse_record(rec_str):
                try:
                    # "25-5-0" or "25-5-0 (1 NC)"
                    base = rec_str.split(' ')[0]
                    parts = base.split('-')
                    if len(parts) >= 2:
                        return int(parts[0]), int(parts[1]) # W, L
                    return None
                except: return None
            
            f1_rec = f1_stats_scr.get('Record')
            f2_rec = f2_stats_scr.get('Record')
            
            w1, l1 = st1['wins'], st1['losses']
            if f1_rec:
                parsed = parse_record(f1_rec)
                if parsed: w1, l1 = parsed
                
            w2, l2 = st2['wins'], st2['losses']
            if f2_rec:
                parsed = parse_record(f2_rec)
                if parsed: w2, l2 = parsed

            # Use these overridden W/L for display
            fight['prediction'] = {
                'winner': f1_name if prob > 0.5 else f2_name,
                'confidence': f"{max(prob, 1-prob)*100:.1f}%",
                'odds': {
                    f1_name: f"{prob*100:.1f}%",
                    f2_name: f"{(1-prob)*100:.1f}%"
                },
                'factors': {
                    f1_name: {
                        'slpm': st1['slpm'],
                        'td_rate': st1['td_avg'],
                        'ctrl_rate': st1['ctrl_rate'],
                        'kd_rate': st1['kd_rate'],
                        'sub_rate': st1['sub_rate'],
                        'wins': w1,
                        'losses': l1,
                        'recent_form': st1['recent_form'],
                        'height': h1,
                        'reach': r1
                    },
                    f2_name: {
                        'slpm': st2['slpm'],
                        'td_rate': st2['td_avg'],
                        'ctrl_rate': st2['ctrl_rate'],
                        'kd_rate': st2['kd_rate'],
                        'sub_rate': st2['sub_rate'],
                        'wins': w2,
                        'losses': l2,
                        'recent_form': st2['recent_form'],
                        'height': h2,
                        'reach': r2
                    }
                }
            }
            
            # 2. Add to Flat List (for CSV)
            pred_row = {
                'Event': event_name,
                'Date': date_str,
                'Fighter_1': f1_name,
                'Fighter_2': f2_name,
                'Prob_1': float(prob),
                'Prob_2': 1.0 - float(prob),
                'ELO_1': int(g1['Rating']),
                'ELO_2': int(g2['Rating']),
                'Odds_1': fight.get('market_odds', {}).get(f1_name, 'N/A'),
                'Odds_2': fight.get('market_odds', {}).get(f2_name, 'N/A')
            }
            predictions_out.append(pred_row)
            
    # Define output paths
    root_path = os.path.join(base_dir, '../upcoming_predictions.json')
    dashboard_path = os.path.join(base_dir, '../dashboard/public/upcoming_events.json')
    dashboard_pred_path = os.path.join(base_dir, '../dashboard/public/upcoming_predictions.json')
    
    # Save to all locations
    for path in [root_path, dashboard_path, dashboard_pred_path]:
        with open(path, 'w') as f:
            json.dump(events, f, indent=2)
        print(f"  -> Saved combined JSON to {path}")
    
    # Save Flat CSV to root
    pd.DataFrame(predictions_out).to_csv(root_path.replace('.json', '.csv'), index=False)

if __name__ == "__main__":
    predict_upcoming()
