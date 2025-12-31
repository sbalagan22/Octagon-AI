"""
OCTAGON AI - MARKET-ALIGNED BACKTESTER (v3)
============================================
Implements "Confirmation" Strategy with CatBoost Model.
Only bets on FAVORITES where model agrees and finds additional value.

Strategy:
- Use actual trained CatBoost model (not simplified logistic)
- Probability Capping: Cap underdog probabilities at market + 15%
- Confirmation Only: Bet when market says favorite AND model agrees they're underpriced
- Value Favorite: Market -200 (66%) + Model -350 (78%) = +12% Edge = BET
"""

import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
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
    if edge <= 0.08:  # 8% minimum edge for favorites
        return 0.0
    
    q = 1 - model_prob
    b = decimal_odds - 1
    if b <= 0: return 0.0
    kelly = (model_prob * b - q) / b
    
    # Quarter-Kelly with 5% max cap
    stake = max(0, kelly * fraction * bankroll)
    stake = min(stake, bankroll * 0.05)
    
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
    def __init__(self):
        self.total_fights = 0
        self.wins = 0
        self.losses = 0
        self.total_time_sec = 0
        self.first_fight_date = None
        self.last_fight_date = None
        self.fight_dates = []
        self.recent_form = []
        
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
               kd, sub, ctrl, sig_acc, head_p, body_p, leg_p, dist_p, clin_p, grou_p):
        self.total_fights += 1
        self.total_time_sec += time_sec
        if self.first_fight_date is None:
            self.first_fight_date = fight_date
        self.last_fight_date = fight_date
        self.fight_dates.append(fight_date)
        
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

def run_backtest():
    print("="*60)
    print("OCTAGON AI - MARKET-ALIGNED BACKTEST (v3)")
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
    
    # Build point-in-time stats
    print("Building point-in-time fighter stats...")
    stats_tracker = {}
    
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
    
    for _, row in fights_df.iterrows():
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
            head1 = float(row.get('Head_%_1', 0.7) or 0.7)
            head2 = float(row.get('Head_%_2', 0.7) or 0.7)
            body1 = float(row.get('Body_%_1', 0.15) or 0.15)
            body2 = float(row.get('Body_%_2', 0.15) or 0.15)
            leg1 = float(row.get('Leg_%_1', 0.15) or 0.15)
            leg2 = float(row.get('Leg_%_2', 0.15) or 0.15)
            dist1 = float(row.get('Distance_%_1', 0.8) or 0.8)
            dist2 = float(row.get('Distance_%_2', 0.8) or 0.8)
            clin1 = float(row.get('Clinch_%_1', 0.1) or 0.1)
            clin2 = float(row.get('Clinch_%_2', 0.1) or 0.1)
            grou1 = float(row.get('Ground_%_1', 0.1) or 0.1)
            grou2 = float(row.get('Ground_%_2', 0.1) or 0.1)
            
            td1_a = td1 / 0.4 if td1 > 0 else 0
            td2_a = td2 / 0.4 if td2 > 0 else 0
            
            if fid1 not in stats_tracker: stats_tracker[fid1] = FighterStats()
            if fid2 not in stats_tracker: stats_tracker[fid2] = FighterStats()
            
            stats_tracker[fid1].update(row['Result_1'], row['Date'], time_sec, str1, str2, 
                                       td1, td1_a, td2_a, td2, kd1, sub1, ctrl1, sig_acc1,
                                       head1, body1, leg1, dist1, clin1, grou1)
            stats_tracker[fid2].update(row['Result_2'], row['Date'], time_sec, str2, str1,
                                       td2, td2_a, td1_a, td1, kd2, sub2, ctrl2, sig_acc2,
                                       head2, body2, leg2, dist2, clin2, grou2)
        except: pass
    
    # Test period
    test_start = pd.to_datetime('2023-01-01')
    test_end = pd.to_datetime('2024-12-31')
    
    test_odds = odds_df[(odds_df['event_date'] >= test_start) & (odds_df['event_date'] <= test_end)]
    test_odds = test_odds.sort_values('event_date').reset_index(drop=True)
    
    print(f"Testing on {len(test_odds)} fights from 2023-2024")
    
    # Backtest
    initial_bankroll = 1000.0
    bankroll = initial_bankroll
    stop_loss = bankroll * 0.7
    
    total_bets = 0
    wins = 0
    losses = 0
    total_wagered = 0.0
    total_pnl = 0.0
    results = []
    
    for _, row in test_odds.iterrows():
        if bankroll < stop_loss:
            print(f"\n[STOP LOSS] Bankroll: ${bankroll:.2f}")
            break
        
        f1_name = row['fighter_1']
        f2_name = row['fighter_2']
        odds_1 = row['odds_1']
        odds_2 = row['odds_2']
        event_date = row['event_date']
        
        fid1 = name_to_id.get(normalize_name(f1_name))
        fid2 = name_to_id.get(normalize_name(f2_name))
        
        if not fid1 or not fid2: continue
        
        st1 = stats_tracker.get(fid1, FighterStats()).get_stats(event_date)
        st2 = stats_tracker.get(fid2, FighterStats()).get_stats(event_date)
        
        if st1['total_fights'] < 3 or st2['total_fights'] < 3:
            continue
        
        # Glicko
        g1_row = glicko_df[(glicko_df['Fighter_Id'] == fid1) & (glicko_df['Date'] < event_date)]
        g2_row = glicko_df[(glicko_df['Fighter_Id'] == fid2) & (glicko_df['Date'] < event_date)]
        g1 = g1_row['Rating'].iloc[-1] if len(g1_row) > 0 else 1500
        g2 = g2_row['Rating'].iloc[-1] if len(g2_row) > 0 else 1500
        g1_rd = g1_row['RD'].iloc[-1] if len(g1_row) > 0 else 350
        g2_rd = g2_row['RD'].iloc[-1] if len(g2_row) > 0 else 350
        
        b1 = fighter_bio.get(fid1, {'height': 175, 'reach': 175, 'stance': 'Orthodox'})
        b2 = fighter_bio.get(fid2, {'height': 175, 'reach': 175, 'stance': 'Orthodox'})
        
        # Build feature row (MUST match train_model.py)
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
            'activity_diff': 0,  # Would need recent_fights_count
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
        
        # PROBABILITY CAPPING: Don't trust extreme underdog predictions
        # If market says underdog (odds > 2.5), cap model at market + 15%
        mp1 = decimal_to_implied_prob(odds_1)
        mp2 = decimal_to_implied_prob(odds_2)
        total_mp = mp1 + mp2
        mp1 /= total_mp
        mp2 /= total_mp
        
        if odds_1 > 2.5:  # F1 is underdog
            model_prob_f1 = min(model_prob_f1, mp1 + 0.15)
        if odds_2 > 2.5:  # F2 is underdog
            model_prob_f1 = max(model_prob_f1, 1 - (mp2 + 0.15))
        
        # Find actual result
        fight_result = fights_df[
            (fights_df['Fighter_Id_1'] == fid1) & 
            (fights_df['Fighter_Id_2'] == fid2) &
            (fights_df['Date'].dt.date == event_date.date())
        ]
        if len(fight_result) == 0:
            fight_result = fights_df[
                (fights_df['Fighter_Id_1'] == fid2) & 
                (fights_df['Fighter_Id_2'] == fid1) &
                (fights_df['Date'].dt.date == event_date.date())
            ]
            if len(fight_result) > 0:
                actual_winner = 2 if fight_result['Result_1'].iloc[0] == 'W' else 1
            else: continue
        else:
            actual_winner = 1 if fight_result['Result_1'].iloc[0] == 'W' else 2
        
        # CONFIRMATION STRATEGY: Only bet on FAVORITES when model confirms
        # Market favorite = odds < 2.0 (>50% implied)
        bet_side = None
        stake = 0.0
        decimal_odds_used = 0.0
        
        # If F1 is market favorite AND model says they should be even stronger
        if odds_1 < 2.0 and model_prob_f1 > mp1 + 0.08:
            stake = calculate_kelly_stake(model_prob_f1, mp1, odds_1, bankroll)
            bet_side = 1
            decimal_odds_used = odds_1
        # If F2 is market favorite AND model confirms
        elif odds_2 < 2.0 and (1 - model_prob_f1) > mp2 + 0.08:
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
            
            results.append({
                'date': event_date,
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
    
    # Report
    print("\n" + "="*60)
    print("MARKET-ALIGNED BACKTEST RESULTS (v3)")
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
        results_df.to_csv(os.path.join(data_dir, 'backtest_v3_results.csv'), index=False)
        print(f"\nDetailed results saved to backtest_v3_results.csv")
    
    return bankroll, total_pnl

if __name__ == "__main__":
    run_backtest()
