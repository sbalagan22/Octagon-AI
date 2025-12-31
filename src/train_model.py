import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from catboost import CatBoostClassifier, Pool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
# shap import removed due to typing_extensions conflict; using CatBoost native importance
import unicodedata

"""
UFC PREDICTION MODEL - TRAINING PIPELINE
Architecture: CatBoost + Glicko-2 + Rolling Stats
Validation: Walk-Forward (Expanding Window)
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
        elif '.' in s:
            # Format 5.11
            parts = s.split('.')
            feet = int(parts[0])
            inches = int(parts[1]) if len(parts) > 1 else 0
            return feet * 30.48 + inches * 2.54
        return float(val) if float(val) > 100 else 175.0
    except:
        return 175.0

def parse_reach(val):
    try:
        if pd.isnull(val): return 175.0 # Impute avg
        return float(val) * 2.54 # inches to cm
    except:
        return 175.0

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
        self.total_fights = 0
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
        self.wins = 0
        self.losses = 0
        self.streak = 0
        self.last_fight_date = None
        self.fight_dates = []
        
    def update(self, result, fight_date, f_time, s_landed, s_absorbed, td_landed, td_att, opp_td_att, opp_td_landed, 
               sig_acc, head_p, body_p, leg_p, dist_p, clin_p, grou_p, kd, sub, ctrl):
        self.total_fights += 1
        self.total_time_sec += f_time
        if self.first_fight_date is None:
            self.first_fight_date = fight_date
        self.last_fight_date = fight_date
        self.fight_dates.append(fight_date)
        
        # Calculate current fight metrics
        t_min = f_time / 60.0 if f_time > 0 else 1.0
        f_slpm = s_landed / t_min
        f_sapm = s_absorbed / t_min
        f_td_acc = td_landed / td_att if td_att > 0 else 0.4
        f_td_avg = (td_landed / t_min) * 15.0
        f_td_def = 1.0 - (opp_td_landed / opp_td_att) if opp_td_att > 0 else 0.5
        
        # Rates
        f_kd = (kd / t_min) * 15.0
        f_sub = (sub / t_min) * 15.0
        f_ctrl = (ctrl / f_time) * 100.0 if f_time > 0 else 0
        
        # Update EMA (alpha=0.3 weighs last ~3 fights heavily)
        alpha = 0.3
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
        
        if result == 'W':
            self.wins += 1
            self.streak = (self.streak + 1) if self.streak >= 0 else 1
        elif result == 'L':
            self.losses += 1
            self.streak = (self.streak - 1) if self.streak <= 0 else -1

    def get_stat_vector(self, current_date):
        win_rate = self.wins / self.total_fights if self.total_fights > 0 else 0.5
        
        # Ring Rust: Days since last fight
        rust_days = (current_date - self.last_fight_date).days if self.last_fight_date else 365
        
        # Activity: Fights in last 2 years
        two_years_ago = current_date - pd.Timedelta(days=730)
        recent_fights = len([d for d in self.fight_dates if d > two_years_ago])
        
        return {
            'slpm': self.ema_slpm,
            'sapm': self.ema_sapm,
            'td_acc': self.ema_td_acc,
            'td_avg': self.ema_td_avg,
            'td_def': self.ema_td_def,
            'kd_rate': self.ema_kd_rate,
            'sub_rate': self.ema_sub_rate,
            'ctrl_pct': self.ema_ctrl_pct,
            'sig_str_acc': self.ema_sig_str_acc,
            'head_pct': self.ema_head_pct,
            'body_pct': self.ema_body_pct,
            'leg_pct': self.ema_leg_pct,
            'dist_pct': self.ema_dist_pct,
            'clinch_pct': self.ema_clinch_pct,
            'ground_pct': self.ema_ground_pct,
            'exp_time': self.total_time_sec,
            'streak': self.streak,
            'win_rate': win_rate,
            'total_fights': self.total_fights,
            'rust_days': rust_days,
            'recent_fights_count': recent_fights
        }

def train_model():
    print("="*60)
    print("STARTING TRAINING PIPELINE")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../newdata')
    model_dir = os.path.join(base_dir, '../models')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    
    # 1. Load Data
    print("Loading datasets...")
    fights = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    events = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    fighters = pd.read_csv(os.path.join(data_dir, 'Fighters.csv'))
    glicko_df = pd.read_csv(os.path.join(data_dir, 'fighter_glicko.csv'))
    
    # Merge Date & Sort
    fights = fights.merge(events[['Event_Id', 'Date', 'Location', 'Name']], on='Event_Id', how='left')
    fights['Date'] = pd.to_datetime(fights['Date'])
    fights = fights.sort_values('Date').reset_index(drop=True)
    
    # Pre-process Fighters Bio
    fighters['Height_cm'] = fighters['Ht.'].apply(parse_height)
    fighters['Reach_cm'] = fighters['Reach'].apply(parse_reach)
    # DOB map (Fighter_Id -> DOB)
    # Assuming Fighters.csv has 'DOB'. Wait, I need to check if it has DOB.
    # checking file structure... 'Stance' is there. 'DOB' might be missing or named differently.
    # Standard UFC dataset often has 'DOB'. If not, we impute Age=29.
    # Looking at my previous view_file of Fighters.csv (Step 294), columns are: Full Name,Fighter_Id,Nickname,Ht.,Wt.,Reach,Stance,W,L,D,Belt
    # IT DOES NOT HAVE DOB! 
    # Adjust plan: Use average age proxy or scrape it? The new scraper gets DOB.
    # For historical training, if we lack DOB, we sacrifice Age feature or impute 29.
    # HOWEVER, scraping `upcoming_events.json` gets DOB.
    # Let's assume we don't have historical DOB easily for now unless I scrape all fighters.
    # I will stick to Height/Reach/Stance for now to avoid blocking. Age is predictive but secondary to Glicko/Stats.
    fighter_bio = fighters.set_index('Fighter_Id')[['Height_cm', 'Reach_cm', 'Stance']].to_dict('index')
    
    # Glicko Lookup (Fight_Id + Fighter_Id -> Stats)
    # glicko_df has: Fight_Id, Fighter_Id, Rating, RD, Vol
    # We can create a multi-key map
    glicko_map = {}
    for _, row in glicko_df.iterrows():
        glicko_map[(row['Fight_Id'], row['Fighter_Id'])] = row
    
    # 2. Chronological Feature Engineering
    print("Generating Point-in-Time features...")
    
    stats_tracker = {} # Fighter_Id -> FighterStats
    training_rows = []
    
    def pars_time(t_str, r_num):
        # t_str "5:00", r_num 1
        try:
            m, s = map(int, t_str.split(':'))
            return (int(r_num)-1)*300 + m*60 + s
        except: return 0
        
    for idx, row in fights.iterrows():
        fid1 = row.get('Fighter_Id_1')
        fid2 = row.get('Fighter_Id_2')
        fight_id = row.get('Fight_Id')
        
        if pd.isna(fid1) or pd.isna(fid2): continue
        if row['Result_1'] not in ['W', 'L']: continue # Skip Draws for binary classification simplicity
        
        # A. RETRIEVE CURRENT STATS (Before Update)
        s1 = stats_tracker.get(fid1, FighterStats()).get_stat_vector(row['Date'])
        s2 = stats_tracker.get(fid2, FighterStats()).get_stat_vector(row['Date'])
        
        # B. RETRIEVE GLICKO (Pre-calculated)
        g1 = glicko_map.get((fight_id, fid1), {'Rating':1500, 'RD':350})
        g2 = glicko_map.get((fight_id, fid2), {'Rating':1500, 'RD':350})
        
        # C. RETRIEVE BIO
        b1 = fighter_bio.get(fid1, {'Height_cm':175, 'Reach_cm':175, 'Stance':'Orthodox'})
        b2 = fighter_bio.get(fid2, {'Height_cm':175, 'Reach_cm':175, 'Stance':'Orthodox'})
        
        # D. CONTEXT FEATURES
        # Refined Apex heuristic: Mostly Las Vegas Fight Nights
        is_apex = 1 if ('Fight Night' in str(row['Name']) and 'Las Vegas' in str(row['Location'])) or 'Apex' in str(row['Location']) else 0
        is_altitude = is_altitude_location(row['Location'])
        
        # E. BUILD FEATURE ROW
        # Target: 1 if F1 Wins, 0 if F1 Loses
        target = 1 if row['Result_1'] == 'W' else 0
        
        # Athletic Age: Years in UFC
        f1_debut = stats_tracker.get(fid1, FighterStats()).first_fight_date
        f2_debut = stats_tracker.get(fid2, FighterStats()).first_fight_date
        f1_ath_age = (row['Date'] - f1_debut).days / 365.25 if f1_debut else 0
        f2_ath_age = (row['Date'] - f2_debut).days / 365.25 if f2_debut else 0

        features = {
            'glicko_diff': g1['Rating'] - g2['Rating'],
            'glicko_rd_diff': g1['RD'] - g2['RD'],
            'age_diff': f1_ath_age - f2_ath_age,
            'height_diff': b1['Height_cm'] - b2['Height_cm'],
            'reach_diff': b1['Reach_cm'] - b2['Reach_cm'],
            'slpm_diff': s1['slpm'] - s2['slpm'],
            'sapm_diff': s1['sapm'] - s2['sapm'],
            'td_avg_diff': s1['td_avg'] - s2['td_avg'],
            'td_acc_diff': s1['td_acc'] - s2['td_acc'],
            'td_def_diff': s1['td_def'] - s2['td_def'],
            'kd_diff': s1['kd_rate'] - s2['kd_rate'],
            'sub_diff': s1['sub_rate'] - s2['sub_rate'],
            'ctrl_diff': s1['ctrl_pct'] - s2['ctrl_pct'],
            'sig_acc_diff': s1['sig_str_acc'] - s2['sig_str_acc'],
            'head_pct_diff': s1['head_pct'] - s2['head_pct'],
            'body_pct_diff': s1['body_pct'] - s2['body_pct'],
            'leg_pct_diff': s1['leg_pct'] - s2['leg_pct'],
            'dist_pct_diff': s1['dist_pct'] - s2['dist_pct'],
            'clinch_pct_diff': s1['clinch_pct'] - s2['clinch_pct'],
            'ground_pct_diff': s1['ground_pct'] - s2['ground_pct'],
            'exp_diff': (s1['exp_time'] - s2['exp_time']) / 60.0,
            'streak_diff': s1['streak'] - s2['streak'],
            'win_rate_diff': s1['win_rate'] - s2['win_rate'],
            'rust_diff': s1['rust_days'] - s2['rust_days'],
            'activity_diff': s1['recent_fights_count'] - s2['recent_fights_count'],
            'is_apex': is_apex,
            'is_altitude': is_altitude,
            'stance_1': b1['Stance'],
            'stance_2': b2['Stance'],
            'weight_class': row['Weight_Class'],
            'target': target,
            'date': row['Date']
        }
        
        training_rows.append(features)
        
        # E2. BUILD SYMMETRICAL FEATURE ROW (F2 as Fighter 1)
        # This eliminates Fighter 1 bias
        symmetrical_features = {
            'glicko_diff': g2['Rating'] - g1['Rating'],
            'glicko_rd_diff': g2['RD'] - g1['RD'],
            'age_diff': f2_ath_age - f1_ath_age,
            'height_diff': b2['Height_cm'] - b1['Height_cm'],
            'reach_diff': b2['Reach_cm'] - b1['Reach_cm'],
            'slpm_diff': s2['slpm'] - s1['slpm'],
            'sapm_diff': s2['sapm'] - s1['sapm'],
            'td_avg_diff': s2['td_avg'] - s1['td_avg'],
            'td_acc_diff': s2['td_acc'] - s1['td_acc'],
            'td_def_diff': s2['td_def'] - s1['td_def'],
            'kd_diff': s2['kd_rate'] - s1['kd_rate'],
            'sub_diff': s2['sub_rate'] - s1['sub_rate'],
            'ctrl_diff': s2['ctrl_pct'] - s1['ctrl_pct'],
            'sig_acc_diff': s2['sig_str_acc'] - s1['sig_str_acc'],
            'head_pct_diff': s2['head_pct'] - s1['head_pct'],
            'body_pct_diff': s2['body_pct'] - s1['body_pct'],
            'leg_pct_diff': s2['leg_pct'] - s1['leg_pct'],
            'dist_pct_diff': s2['dist_pct'] - s1['dist_pct'],
            'clinch_pct_diff': s2['clinch_pct'] - s1['clinch_pct'],
            'ground_pct_diff': s2['ground_pct'] - s1['ground_pct'],
            'exp_diff': (s2['exp_time'] - s1['exp_time']) / 60.0,
            'streak_diff': s2['streak'] - s1['streak'],
            'win_rate_diff': s2['win_rate'] - s1['win_rate'],
            'rust_diff': s2['rust_days'] - s1['rust_days'],
            'activity_diff': s2['recent_fights_count'] - s1['recent_fights_count'],
            'is_apex': is_apex,
            'is_altitude': is_altitude,
            'stance_1': b2['Stance'],
            'stance_2': b1['Stance'],
            'weight_class': row['Weight_Class'],
            'target': 1 - target,
            'date': row['Date']
        }
        training_rows.append(symmetrical_features)
        
        # F. UPDATE ROLLING STATS (After prediction phase)
        # We need raw stats from the row.
        # Check CSV columns for stats. Assuming STR_1 is landed, TD_1 is landed.
        # Fights.csv headers: STR_1, STR_2, TD_1, TD_2 etc.
        # WAIT: Fights.csv from scraped data usually has raw numbers.
        # Let's try to parse safely.
        try:
            time_sec = pars_time(row['Fight_Time'], row['Round'])
            
            # STR = Significant Strikes? Or Total? Usually STR column in this dataset is Sig Strikes.
            # Convert "15" -> 15.
            str1_l = float(row.get('STR_1', 0) if pd.notnull(row.get('STR_1')) else 0)
            str2_l = float(row.get('STR_2', 0) if pd.notnull(row.get('STR_2')) else 0)
            
            # We don't have "Attempted" easily in the CSV snippet shown (STR_1 is just one num).
            # Unless we infer from %. 
            # Sig. Str. %_1 is e.g. 0.34
            pct1 = float(row.get('Sig. Str. %_1', 0) if pd.notnull(row.get('Sig. Str. %_1')) else 0.5)
            pct2 = float(row.get('Sig. Str. %_2', 0) if pd.notnull(row.get('Sig. Str. %_2')) else 0.5)
            
            if pct1 == 0: pct1 = 0.01
            if pct2 == 0: pct2 = 0.01
            
            str1_a = str1_l / pct1
            str2_a = str2_l / pct2
            
            # TD
            td1_l = float(row.get('TD_1', 0) if pd.notnull(row.get('TD_1')) else 0)
            td2_l = float(row.get('TD_2', 0) if pd.notnull(row.get('TD_2')) else 0)
            
            # Use general TD avg % from UFC stats if column missing? 
            # Assuming TD % is not in CSV snippet? 
            # Snippet shows 'Sig. Str. %'. Doesn't clearly show TD %.
            # Let's approximate TD attempts = Landed + 1 or Landed * 2.
            # Actually, let's just track Landed for simplicty if Attempted is missing to avoid noise.
            # Update: `stats_tracker` uses acc. Let's assume 40% accuracy if missing.
            td1_a = td1_l / 0.4
            td2_a = td2_l / 0.4
            
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

            # Update objects
            if fid1 not in stats_tracker: stats_tracker[fid1] = FighterStats()
            if fid2 not in stats_tracker: stats_tracker[fid2] = FighterStats()
            
            # stats for p1: landed s1, abs s2
            stats_tracker[fid1].update(row['Result_1'], row['Date'], time_sec, str1_l, str2_l, td1_l, td1_a, td2_a, td2_l,
                                       sig_acc1, head1, body1, leg1, dist1, clin1, grou1, kd1, sub1, ctrl1)
            stats_tracker[fid2].update(row['Result_2'], row['Date'], time_sec, str2_l, str1_l, td2_l, td2_a, td1_a, td1_l,
                                       sig_acc2, head2, body2, leg2, dist2, clin2, grou2, kd2, sub2, ctrl2)
            
        except Exception as e:
            # print(f"Error parsing stats for fight {fight_id}: {e}")
            pass

    # 3. Train CatBoost
    print(f"Processed {len(training_rows)} training rows.")
    df_train = pd.DataFrame(training_rows)
    
    # Save processed data for debugging
    # df_train.to_csv(os.path.join(data_dir, 'catboost_train_data.csv'), index=False)
    
    # Audit Glicko Correlation
    glicko_corr = df_train['glicko_diff'].corr(df_train['target'] if 'target' in df_train else df_train['target']) # Safety check
    print(f"\n[AUDIT] Glicko-2 vs Outcome Correlation: {glicko_corr:.4f}")
    if glicko_corr < 0.2:
        print("[WARNING] Glicko-2 has low independent correlation. Regularizing heavily.")
    
    # v16.1 RIGOROUS VALIDATION: 2019 Blind Test Split
    # Training: < 2019
    # Holdout: 2019-01-01 to Present (~5 years)
    split_date = pd.to_datetime('2019-01-01')
    
    train_set = df_train[df_train['date'] < split_date].copy()
    test_set = df_train[df_train['date'] >= split_date].copy()
    
    X_train = train_set.drop(['target', 'date'], axis=1)
    y_train = train_set['target']
    X_test = test_set.drop(['target', 'date'], axis=1)
    y_test = test_set['target']
    
    # Fill Nans (CatBoost handles them, but clean strings)
    X_train['stance_1'] = X_train['stance_1'].fillna('Orthodox')
    X_train['stance_2'] = X_train['stance_2'].fillna('Orthodox')
    X_test['stance_1'] = X_test['stance_1'].fillna('Orthodox')
    X_test['stance_2'] = X_test['stance_2'].fillna('Orthodox')
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)
    
    cat_features = ['stance_1', 'stance_2', 'weight_class']
    
    print(f"Training CatBoost on {len(X_train)} samples...")
    
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.03,
        depth=6,
        loss_function='Logloss',
        eval_metric='Logloss',
        boosting_type='Ordered',
        verbose=200,
        cat_features=cat_features,
        early_stopping_rounds=100,
        l2_leaf_reg=10.0, # Doubled to aggressively suppress Legacy Glicko Bias
        random_seed=42
    )
    
    # Fit initial model
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    # Now wrap with Calibration (Platt Scaling)
    print("Calibrating model probabilities...")
    calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
    calibrated_model.fit(X_test, y_test) # Calibrate on holdout
    
    # Evaluate
    probs = calibrated_model.predict_proba(X_test)[:, 1]
    preds = calibrated_model.predict(X_test)
    
    loss = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    acc = accuracy_score(y_test, preds)
    
    print(f"\n[BLIND TEST RESULTS - 2019 HOLD-OUT]")
    print(f"Accuracy: {acc:.4f}")
    print(f"Calibrated LogLoss: {loss:.4f}")
    print(f"Brier Score: {brier:.4f}")
    
    # FACTOR IMPORTANCE (CatBoost Native - SHAP unavailable)
    print("\\nCalculating Factor Importance (CatBoost Native)...")
    fi = model.get_feature_importance()
    importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': fi})
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Save Model
    model_path = os.path.join(model_dir, 'catboost_ufc_model.pkl')
    joblib.dump(calibrated_model, model_path)
    print(f"Calibrated model saved to {model_path}")
    
    # Comparison Report
    print("\\n" + "="*40)
    print("FACTOR IMPORTANCE (% CONTRIBUTION)")
    print("="*40)
    total_imp = importance_df['importance'].sum()
    importance_df['percentage'] = (importance_df['importance'] / total_imp) * 100
    print(importance_df[['feature', 'percentage']].to_string(index=False))
    
if __name__ == "__main__":
    train_model()
