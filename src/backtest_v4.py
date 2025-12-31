"""
OCTAGON AI - ZERO-LEAKAGE BACKTESTER (v4)
==========================================
Industry-standard time-series Walk-Forward Validation.
NO future information is used at any prediction point.

Protocol:
1. Pre-Test Baseline: Build stats_tracker from 2010-2022 ONLY
2. Predict-then-Update Loop: For each 2023-2024 fight:
   - EXTRACT: Get fighter's pre-fight stats
   - PREDICT: Model makes prediction
   - BET: Log result against market odds
   - UPDATE: Only AFTER prediction, update stats with fight outcome
3. Debutant Handling: Newcomers get baseline stats until first UFC fight
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from copy import deepcopy
import unicodedata

# --- Helpers ---
def normalize_name(name):
    if not name: return ""
    name = unicodedata.normalize('NFD', str(name))
    name = name.encode('ascii', 'ignore').decode('utf-8')
    name = name.lower().strip()
    name = name.replace('.', '').replace("'", "").replace("-", " ")
    return ' '.join(name.split())

def decimal_to_implied_prob(odds):
    if odds <= 1.0: return 0.5
    return 1.0 / odds

def calculate_kelly_stake(model_prob, market_prob, decimal_odds, bankroll, fraction=0.25):
    edge = model_prob - market_prob
    if edge <= 0.10:  # 10% minimum edge
        return 0.0
    
    q = 1 - model_prob
    b = decimal_odds - 1
    if b <= 0: return 0.0
    kelly = (model_prob * b - q) / b
    
    stake = max(0, kelly * fraction * bankroll)
    stake = min(stake, bankroll * 0.05)  # 5% max per bet
    
    return round(stake, 2)

def parse_height(h):
    if not h or pd.isnull(h): return 175.0
    try:
        if "'" in str(h):
            parts = str(h).replace('"', '').split("'")
            return float(parts[0]) * 30.48 + float(parts[1].strip() or 0) * 2.54
        return float(h)
    except: return 175.0

def parse_reach(r):
    if not r or pd.isnull(r): return 175.0
    try:
        r_str = str(r).replace('"', '').replace("'", "").strip()
        if r_str.endswith('cm'): return float(r_str.replace('cm', ''))
        return float(r_str) * 2.54
    except: return 175.0

class FighterStats:
    """Point-in-time fighter stats with EMA tracking."""
    def __init__(self):
        self.total_fights = 0
        self.wins = 0
        self.losses = 0
        self.total_time_sec = 0
        self.first_fight_date = None
        self.last_fight_date = None
        self.recent_form = []
        
        # EMA defaults (newcomer baseline)
        self.ema_slpm = 4.0
        self.ema_sapm = 3.0
        self.ema_td_avg = 1.0
        self.ema_td_acc = 0.4
        self.ema_td_def = 0.5
        self.ema_ctrl_pct = 10.0
        self.ema_sig_acc = 0.45
        self.ema_kd_rate = 0.2
        self.ema_sub_rate = 0.2
        
    def update(self, result, fight_date, time_sec, slpm, sapm, td_l, td_a, opp_td_a, opp_td_l, 
               kd, sub, ctrl, sig_acc):
        self.total_fights += 1
        self.total_time_sec += time_sec
        if self.first_fight_date is None:
            self.first_fight_date = fight_date
        self.last_fight_date = fight_date
        
        if result == 'W': 
            self.wins += 1
            self.recent_form.append('W')
        elif result == 'L': 
            self.losses += 1
            self.recent_form.append('L')
        self.recent_form = self.recent_form[-5:]
        
        t_min = time_sec / 60.0 if time_sec > 0 else 1.0
        f_slpm = slpm / t_min if slpm else 4.0
        f_sapm = sapm / t_min if sapm else 3.0
        f_td = (td_l / t_min) * 15.0 if td_l else 1.0
        f_td_acc = td_l / td_a if td_a > 0 else 0.4
        f_td_def = 1.0 - (opp_td_l / opp_td_a) if opp_td_a > 0 else 0.5
        f_kd = (kd / t_min) * 15.0
        f_sub = (sub / t_min) * 15.0
        f_ctrl = (ctrl / time_sec) * 100 if time_sec > 0 else 10.0
        
        alpha = 0.3
        if self.total_fights == 1:
            self.ema_slpm, self.ema_sapm = f_slpm, f_sapm
            self.ema_td_avg, self.ema_td_acc, self.ema_td_def = f_td, f_td_acc, f_td_def
            self.ema_kd_rate, self.ema_sub_rate, self.ema_ctrl_pct = f_kd, f_sub, f_ctrl
            self.ema_sig_acc = sig_acc
        else:
            self.ema_slpm = alpha * f_slpm + (1 - alpha) * self.ema_slpm
            self.ema_sapm = alpha * f_sapm + (1 - alpha) * self.ema_sapm
            self.ema_td_avg = alpha * f_td + (1 - alpha) * self.ema_td_avg
            self.ema_td_acc = alpha * f_td_acc + (1 - alpha) * self.ema_td_acc
            self.ema_td_def = alpha * f_td_def + (1 - alpha) * self.ema_td_def
            self.ema_kd_rate = alpha * f_kd + (1 - alpha) * self.ema_kd_rate
            self.ema_sub_rate = alpha * f_sub + (1 - alpha) * self.ema_sub_rate
            self.ema_ctrl_pct = alpha * f_ctrl + (1 - alpha) * self.ema_ctrl_pct
            self.ema_sig_acc = alpha * sig_acc + (1 - alpha) * self.ema_sig_acc
    
    def get_stats(self, current_date):
        rust_days = (current_date - self.last_fight_date).days if self.last_fight_date else 365
        win_rate = self.wins / self.total_fights if self.total_fights > 0 else 0.5
        ath_age = (current_date - self.first_fight_date).days / 365.25 if self.first_fight_date else 0
        recent_wr = len([f for f in self.recent_form if f == 'W']) / len(self.recent_form) if self.recent_form else 0.5
        streak = 0
        for r in reversed(self.recent_form):
            if r == self.recent_form[-1] if self.recent_form else True:
                streak += 1 if r == 'W' else -1
            else:
                break
        
        return {
            'slpm': self.ema_slpm, 'sapm': self.ema_sapm,
            'td_avg': self.ema_td_avg, 'td_acc': self.ema_td_acc, 'td_def': self.ema_td_def,
            'ctrl_pct': self.ema_ctrl_pct, 'sig_acc': self.ema_sig_acc,
            'kd_rate': self.ema_kd_rate, 'sub_rate': self.ema_sub_rate,
            'win_rate': win_rate, 'recent_win_rate': recent_wr,
            'rust_days': rust_days, 'ath_age': ath_age,
            'streak': streak, 'total_fights': self.total_fights,
            'exp_time': self.total_time_sec
        }

def parse_ctrl(val):
    if pd.isnull(val) or val == '--': return 0
    try:
        if ':' in str(val):
            m, s = map(int, str(val).split(':'))
            return m*60 + s
        return int(val)
    except: return 0

def parse_time(t_str, r_num):
    try:
        m, s = map(int, t_str.split(':'))
        return (int(r_num)-1)*300 + m*60 + s
    except: return 0

def run_backtest():
    print("="*60)
    print("OCTAGON AI - ZERO-LEAKAGE BACKTEST (v4)")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../newdata')
    model_dir = os.path.join(base_dir, '../models')
    
    print("Loading model and data...")
    model = joblib.load(os.path.join(model_dir, 'catboost_ufc_model.pkl'))
    
    odds_df = pd.read_csv(os.path.join(data_dir, 'UFC_betting_odds.csv'), low_memory=False)
    odds_df['event_date'] = pd.to_datetime(odds_df['event_date'])
    
    fighters_df = pd.read_csv(os.path.join(data_dir, 'Fighters.csv'))
    fights_df = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    events_df = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    fights_df = fights_df.merge(events_df[['Event_Id', 'Date']], on='Event_Id')
    fights_df['Date'] = pd.to_datetime(fights_df['Date'])
    fights_df = fights_df.sort_values('Date').reset_index(drop=True)
    
    glicko_df = pd.read_csv(os.path.join(data_dir, 'fighter_glicko.csv'))
    glicko_df['Date'] = pd.to_datetime(glicko_df['Date'])
    
    # Build name->id and bio maps
    name_to_id = {}
    fighter_bio = {}
    for _, row in fighters_df.iterrows():
        n = normalize_name(row['Full Name'])
        fid = row['Fighter_Id']
        name_to_id[n] = fid
        fighter_bio[fid] = {
            'height': parse_height(row.get('Height')),
            'reach': parse_reach(row.get('Reach')),
            'stance': row.get('Stance', 'Orthodox')
        }
    
    # === STEP 1: BUILD PRE-TEST BASELINE (2010-2022 ONLY) ===
    test_start = pd.to_datetime('2023-01-01')
    test_end = pd.to_datetime('2024-12-31')
    
    print(f"Building pre-test baseline (fights before {test_start.date()})...")
    stats_tracker = {}
    
    pre_test_fights = fights_df[fights_df['Date'] < test_start].copy()
    
    for _, row in pre_test_fights.iterrows():
        fid1, fid2 = row.get('Fighter_Id_1'), row.get('Fighter_Id_2')
        if pd.isna(fid1) or pd.isna(fid2): continue
        if row['Result_1'] not in ['W', 'L', 'D']: continue
        
        try:
            time_sec = parse_time(row['Fight_Time'], row['Round'])
            str1 = float(row.get('STR_1', 0) or 0)
            str2 = float(row.get('STR_2', 0) or 0)
            td1 = float(row.get('TD_1', 0) or 0)
            td2 = float(row.get('TD_2', 0) or 0)
            ctrl1 = parse_ctrl(row.get('Ctrl_1'))
            ctrl2 = parse_ctrl(row.get('Ctrl_2'))
            kd1 = float(row.get('KD_1', 0) or 0)
            kd2 = float(row.get('KD_2', 0) or 0)
            sub1 = float(row.get('SUB_1', 0) or 0)
            sub2 = float(row.get('SUB_2', 0) or 0)
            sig_acc1 = float(row.get('Sig. Str._%_1', 0.45) or 0.45)
            sig_acc2 = float(row.get('Sig. Str._%_2', 0.45) or 0.45)
            
            td1_a = td1 / 0.4 if td1 > 0 else 0
            td2_a = td2 / 0.4 if td2 > 0 else 0
            
            if fid1 not in stats_tracker: stats_tracker[fid1] = FighterStats()
            if fid2 not in stats_tracker: stats_tracker[fid2] = FighterStats()
            
            stats_tracker[fid1].update(row['Result_1'], row['Date'], time_sec, str1, str2, 
                                       td1, td1_a, td2_a, td2, kd1, sub1, ctrl1, sig_acc1)
            stats_tracker[fid2].update(row['Result_2'], row['Date'], time_sec, str2, str1,
                                       td2, td2_a, td1_a, td1, kd2, sub2, ctrl2, sig_acc2)
        except: pass
    
    print(f"  -> Baseline built from {len(pre_test_fights)} pre-2023 fights")
    print(f"  -> {len(stats_tracker)} fighters in tracker")
    
    # === STEP 2: TEST PERIOD FIGHTS ===
    test_fights = fights_df[(fights_df['Date'] >= test_start) & (fights_df['Date'] <= test_end)].copy()
    test_fights = test_fights.sort_values('Date').reset_index(drop=True)
    
    # Build a lookup for odds by fight
    fights_with_odds = []
    for _, fight_row in test_fights.iterrows():
        fid1, fid2 = fight_row.get('Fighter_Id_1'), fight_row.get('Fighter_Id_2')
        fight_date = fight_row['Date']
        
        # Find matching odds row
        odds_match = odds_df[
            (odds_df['event_date'].dt.date == fight_date.date())
        ]
        
        for _, odds_row in odds_match.iterrows():
            ofid1 = name_to_id.get(normalize_name(odds_row['fighter_1']))
            ofid2 = name_to_id.get(normalize_name(odds_row['fighter_2']))
            
            if (ofid1 == fid1 and ofid2 == fid2) or (ofid1 == fid2 and ofid2 == fid1):
                # Matched!
                flipped = (ofid1 == fid2)
                fights_with_odds.append({
                    'fight_row': fight_row,
                    'odds_row': odds_row,
                    'flipped': flipped
                })
                break
    
    print(f"Testing on {len(fights_with_odds)} fights with odds from 2023-2024")
    
    # === STEP 3: PREDICT-THEN-UPDATE LOOP ===
    initial_bankroll = 1000.0
    bankroll = initial_bankroll
    stop_loss = bankroll * 0.7
    
    total_bets = 0
    wins = 0
    losses = 0
    total_wagered = 0.0
    total_pnl = 0.0
    results = []
    
    for idx, item in enumerate(fights_with_odds):
        if bankroll < stop_loss:
            print(f"\n[STOP LOSS] Bankroll: ${bankroll:.2f}")
            break
        
        fight_row = item['fight_row']
        odds_row = item['odds_row']
        flipped = item['flipped']
        
        fid1 = fight_row.get('Fighter_Id_1')
        fid2 = fight_row.get('Fighter_Id_2')
        fight_date = fight_row['Date']
        
        if flipped:
            odds_1 = odds_row['odds_2']
            odds_2 = odds_row['odds_1']
        else:
            odds_1 = odds_row['odds_1']
            odds_2 = odds_row['odds_2']
        
        # --- EXTRACT: Get PRE-FIGHT stats ---
        st1 = stats_tracker.get(fid1, FighterStats()).get_stats(fight_date)
        st2 = stats_tracker.get(fid2, FighterStats()).get_stats(fight_date)
        
        # Skip if both are newcomers (no data)
        if st1['total_fights'] < 1 and st2['total_fights'] < 1:
            continue
        
        # Glicko (point-in-time)
        g1_row = glicko_df[(glicko_df['Fighter_Id'] == fid1) & (glicko_df['Date'] < fight_date)]
        g2_row = glicko_df[(glicko_df['Fighter_Id'] == fid2) & (glicko_df['Date'] < fight_date)]
        g1 = g1_row['Rating'].iloc[-1] if len(g1_row) > 0 else 1500
        g2 = g2_row['Rating'].iloc[-1] if len(g2_row) > 0 else 1500
        g1_rd = g1_row['RD'].iloc[-1] if len(g1_row) > 0 else 350
        g2_rd = g2_row['RD'].iloc[-1] if len(g2_row) > 0 else 350
        
        b1 = fighter_bio.get(fid1, {'height': 175, 'reach': 175, 'stance': 'Orthodox'})
        b2 = fighter_bio.get(fid2, {'height': 175, 'reach': 175, 'stance': 'Orthodox'})
        
        # --- PREDICT: Build feature row and get model probability ---
        features = {
            'glicko_diff': np.clip(g1 - g2, -250, 250),
            'glicko_rd_diff': g1_rd - g2_rd,
            'age_diff': st1['ath_age'] - st2['ath_age'],
            'height_diff': b1['height'] - b2['height'],
            'reach_diff': b1['reach'] - b2['reach'],
            'slpm_diff': st1['slpm'] - st2['slpm'],
            'sapm_diff': st1['sapm'] - st2['sapm'],
            'td_avg_diff': st1['td_avg'] - st2['td_avg'],
            'td_acc_diff': st1['td_acc'] - st2['td_acc'],
            'td_def_diff': st1['td_def'] - st2['td_def'],
            'kd_diff': st1['kd_rate'] - st2['kd_rate'],
            'sub_diff': st1['sub_rate'] - st2['sub_rate'],
            'ctrl_diff': st1['ctrl_pct'] - st2['ctrl_pct'],
            'sig_acc_diff': st1['sig_acc'] - st2['sig_acc'],
            'exp_diff': (st1['exp_time'] - st2['exp_time']) / 60.0,
            'streak_diff': st1['streak'] - st2['streak'],
            'win_rate_diff': st1['recent_win_rate'] - st2['recent_win_rate'],
            'rust_diff': st1['rust_days'] - st2['rust_days'],
            'activity_diff': 0,
            'head_pct_diff': 0, 'body_pct_diff': 0, 'leg_pct_diff': 0,
            'dist_pct_diff': 0, 'clinch_pct_diff': 0, 'ground_pct_diff': 0,
            'is_apex': 0, 'is_altitude': 0,
            'stance_1': b1['stance'], 'stance_2': b2['stance'],
            'weight_class': '155 lbs'
        }
        
        try:
            df_feat = pd.DataFrame([features])
            model_prob_f1 = model.predict_proba(df_feat)[0][1]
        except:
            continue
        
        # Market probabilities
        mp1 = decimal_to_implied_prob(odds_1)
        mp2 = decimal_to_implied_prob(odds_2)
        total_mp = mp1 + mp2
        mp1 /= total_mp
        mp2 /= total_mp
        
        # Probability capping for underdogs
        if odds_1 > 2.5:
            model_prob_f1 = min(model_prob_f1, mp1 + 0.15)
        if odds_2 > 2.5:
            model_prob_f1 = max(model_prob_f1, 1 - (mp2 + 0.15))
        
        # Actual result
        actual_winner = 1 if fight_row['Result_1'] == 'W' else 2
        
        # --- BET: Confirmation strategy (favorites only) ---
        bet_side = None
        stake = 0.0
        decimal_odds_used = 0.0
        
        if odds_1 < 2.0 and model_prob_f1 > mp1 + 0.10:
            stake = calculate_kelly_stake(model_prob_f1, mp1, odds_1, bankroll)
            bet_side = 1
            decimal_odds_used = odds_1
        elif odds_2 < 2.0 and (1 - model_prob_f1) > mp2 + 0.10:
            stake = calculate_kelly_stake(1 - model_prob_f1, mp2, odds_2, bankroll)
            bet_side = 2
            decimal_odds_used = odds_2
        
        if stake > 0:
            total_bets += 1
            total_wagered += stake
            
            if bet_side == actual_winner:
                pnl = stake * (decimal_odds_used - 1)
                wins += 1
            else:
                pnl = -stake
                losses += 1
            
            bankroll += pnl
            total_pnl += pnl
            
            f1_name = odds_row['fighter_1'] if not flipped else odds_row['fighter_2']
            f2_name = odds_row['fighter_2'] if not flipped else odds_row['fighter_1']
            
            results.append({
                'date': fight_date,
                'fighter_1': f1_name,
                'fighter_2': f2_name,
                'bet_on': f1_name if bet_side == 1 else f2_name,
                'model_prob': model_prob_f1 if bet_side == 1 else 1 - model_prob_f1,
                'market_prob': mp1 if bet_side == 1 else mp2,
                'edge': (model_prob_f1 - mp1) if bet_side == 1 else ((1 - model_prob_f1) - mp2),
                'stake': stake,
                'odds': decimal_odds_used,
                'won': bet_side == actual_winner,
                'pnl': pnl,
                'bankroll': bankroll
            })
        
        # --- UPDATE: Now update stats with this fight's data ---
        try:
            time_sec = parse_time(fight_row['Fight_Time'], fight_row['Round'])
            str1 = float(fight_row.get('STR_1', 0) or 0)
            str2 = float(fight_row.get('STR_2', 0) or 0)
            td1 = float(fight_row.get('TD_1', 0) or 0)
            td2 = float(fight_row.get('TD_2', 0) or 0)
            ctrl1 = parse_ctrl(fight_row.get('Ctrl_1'))
            ctrl2 = parse_ctrl(fight_row.get('Ctrl_2'))
            kd1 = float(fight_row.get('KD_1', 0) or 0)
            kd2 = float(fight_row.get('KD_2', 0) or 0)
            sub1 = float(fight_row.get('SUB_1', 0) or 0)
            sub2 = float(fight_row.get('SUB_2', 0) or 0)
            sig_acc1 = float(fight_row.get('Sig. Str._%_1', 0.45) or 0.45)
            sig_acc2 = float(fight_row.get('Sig. Str._%_2', 0.45) or 0.45)
            
            td1_a = td1 / 0.4 if td1 > 0 else 0
            td2_a = td2 / 0.4 if td2 > 0 else 0
            
            if fid1 not in stats_tracker: stats_tracker[fid1] = FighterStats()
            if fid2 not in stats_tracker: stats_tracker[fid2] = FighterStats()
            
            stats_tracker[fid1].update(fight_row['Result_1'], fight_date, time_sec, str1, str2,
                                       td1, td1_a, td2_a, td2, kd1, sub1, ctrl1, sig_acc1)
            stats_tracker[fid2].update(fight_row['Result_2'], fight_date, time_sec, str2, str1,
                                       td2, td2_a, td1_a, td1, kd2, sub2, ctrl2, sig_acc2)
        except: pass
    
    # Report
    print("\n" + "="*60)
    print("ZERO-LEAKAGE BACKTEST RESULTS (v4)")
    print("="*60)
    print(f"Initial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll:   ${bankroll:.2f}")
    print(f"Total P/L:        ${total_pnl:.2f}")
    print(f"ROI:              {(total_pnl / initial_bankroll) * 100:.2f}%")
    print(f"\nTotal Bets:       {total_bets}")
    if total_bets > 0:
        print(f"Wins:             {wins} ({(wins/total_bets)*100:.1f}%)")
        print(f"Losses:           {losses}")
        print(f"Total Wagered:    ${total_wagered:.2f}")
        print(f"Yield:            {(total_pnl / total_wagered) * 100:.2f}%")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(data_dir, 'backtest_v4_zero_leakage.csv'), index=False)
        print(f"\nDetailed results saved to backtest_v4_zero_leakage.csv")
    
    return bankroll, total_pnl

if __name__ == "__main__":
    run_backtest()
