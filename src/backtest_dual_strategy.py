"""
OCTAGON AI - DUAL STRATEGY BACKTESTER
======================================
Strategy 1: CONSERVATIVE FAVORITES (target: 8-12% ROI)
Strategy 2: SMART UNDERDOGS (value hunting)

Both use strict n+1 protocol with variance-adjusted bet sizing.
"""

import pandas as pd
import numpy as np
import os
import joblib
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

class FighterStats:
    def __init__(self):
        self.total_fights = 0
        self.wins = 0
        self.losses = 0
        self.total_time_sec = 0
        self.first_fight_date = None
        self.last_fight_date = None
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
        alpha = 0.3
        
        f_slpm = slpm / t_min if slpm else 4.0
        f_sapm = sapm / t_min if sapm else 3.0
        f_td = (td_l / t_min) * 15.0 if td_l else 1.0
        f_td_acc = td_l / td_a if td_a > 0 else 0.4
        f_td_def = 1.0 - (opp_td_l / opp_td_a) if opp_td_a > 0 else 0.5
        f_kd = (kd / t_min) * 15.0
        f_sub = (sub / t_min) * 15.0
        f_ctrl = (ctrl / time_sec) * 100 if time_sec > 0 else 10.0
        
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
        streak = sum(1 if self.recent_form and self.recent_form[-1] == 'W' else -1 
                     for r in self.recent_form if r == (self.recent_form[-1] if self.recent_form else 'W'))
        
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

def run_backtest():
    print("="*60)
    print("OCTAGON AI - DUAL STRATEGY BACKTESTER")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../newdata')
    model_dir = os.path.join(base_dir, '../models')
    
    print("Loading data...")
    model = joblib.load(os.path.join(model_dir, 'catboost_ufc_model.pkl'))
    
    odds_df = pd.read_csv(os.path.join(data_dir, 'UFC_betting_odds (1).csv'), low_memory=False)
    odds_df['event_date'] = pd.to_datetime(odds_df['event_date'])
    odds_df = odds_df[odds_df['event_date'] < pd.to_datetime('2025-01-01')]
    odds_df = odds_df.sort_values('event_date').reset_index(drop=True)
    
    fighters_df = pd.read_csv(os.path.join(data_dir, 'Fighters.csv'))
    fights_df = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    events_df = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    fights_df = fights_df.merge(events_df[['Event_Id', 'Date']], on='Event_Id')
    fights_df['Date'] = pd.to_datetime(fights_df['Date'])
    fights_df = fights_df.sort_values('Date').reset_index(drop=True)
    
    glicko_df = pd.read_csv(os.path.join(data_dir, 'fighter_glicko.csv'))
    glicko_df['Date'] = pd.to_datetime(glicko_df['Date'])
    
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
    
    # Match fights with odds
    print("Matching fights with odds...")
    fight_odds_pairs = []
    seen_fights = set()
    
    for _, odds_row in odds_df.iterrows():
        event_date = odds_row['event_date']
        fid1 = name_to_id.get(normalize_name(odds_row['fighter_1']))
        fid2 = name_to_id.get(normalize_name(odds_row['fighter_2']))
        if not fid1 or not fid2: continue
        
        fight_match = fights_df[
            (fights_df['Date'].dt.date == event_date.date()) &
            (((fights_df['Fighter_Id_1'] == fid1) & (fights_df['Fighter_Id_2'] == fid2)) |
             ((fights_df['Fighter_Id_1'] == fid2) & (fights_df['Fighter_Id_2'] == fid1)))
        ]
        if len(fight_match) == 0: continue
        
        fight_key = (event_date.date(), frozenset([fid1, fid2]))
        if fight_key in seen_fights: continue
        seen_fights.add(fight_key)
        
        fight_row = fight_match.iloc[0]
        flipped = (fight_row['Fighter_Id_1'] == fid2)
        
        fight_odds_pairs.append({
            'fight_row': fight_row, 'odds_row': odds_row,
            'fid1': fid1, 'fid2': fid2, 'flipped': flipped, 'event_date': event_date
        })
    
    fight_odds_pairs = sorted(fight_odds_pairs, key=lambda x: x['event_date'])
    print(f"Found {len(fight_odds_pairs)} unique fights")
    
    # Initialize both strategies
    stats_tracker = {}
    
    strategies = {
        'favorites': {
            'bankroll': 1000.0, 'bets': 0, 'wins': 0, 'wagered': 0, 'pnl': 0,
            'min_edge': 0.12, 'max_edge': 0.17,  # OPTIMIZED: 12-17% edge sweet spot
            'odds_range': (1.75, 2.0),  # OPTIMIZED: 1.75-2.0 odds = best WR
            'stake_pct': 0.02, 'results': []
        },
        'underdogs': {
            'bankroll': 1000.0, 'bets': 0, 'wins': 0, 'wagered': 0, 'pnl': 0,
            'min_edge': 0.25, 'max_edge': 0.50,  # Higher edge for underdogs
            'odds_range': (2.5, 4.5),  # Medium underdogs, not longshots
            'stake_pct': 0.01,  # Smaller stakes for higher variance
            'min_td_adv': 1.0,  # Underdog MUST have strong TD advantage
            'min_reach_adv': 3.0,  # OR strong reach advantage
            'results': []
        }
    }
    
    for idx, item in enumerate(fight_odds_pairs):
        fight_row = item['fight_row']
        odds_row = item['odds_row']
        fid1, fid2 = item['fid1'], item['fid2']
        flipped = item['flipped']
        event_date = item['event_date']
        
        if flipped:
            real_fid1, real_fid2 = fid2, fid1
            odds_1, odds_2 = odds_row['odds_2'], odds_row['odds_1']
        else:
            real_fid1, real_fid2 = fid1, fid2
            odds_1, odds_2 = odds_row['odds_1'], odds_row['odds_2']
        
        # Get pre-fight stats
        st1 = stats_tracker.get(real_fid1, FighterStats()).get_stats(event_date)
        st2 = stats_tracker.get(real_fid2, FighterStats()).get_stats(event_date)
        
        if st1['total_fights'] < 2 and st2['total_fights'] < 2:
            # Update stats and skip
            try:
                time_sec = parse_time(fight_row['Fight_Time'], fight_row['Round'])
                actual_fid1, actual_fid2 = fight_row['Fighter_Id_1'], fight_row['Fighter_Id_2']
                if actual_fid1 not in stats_tracker: stats_tracker[actual_fid1] = FighterStats()
                if actual_fid2 not in stats_tracker: stats_tracker[actual_fid2] = FighterStats()
                stats_tracker[actual_fid1].update(fight_row['Result_1'], event_date, time_sec, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45)
                stats_tracker[actual_fid2].update(fight_row['Result_2'], event_date, time_sec, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.45)
            except: pass
            continue
        
        # Get Glicko
        g1_row = glicko_df[(glicko_df['Fighter_Id'] == real_fid1) & (glicko_df['Date'] < event_date)]
        g2_row = glicko_df[(glicko_df['Fighter_Id'] == real_fid2) & (glicko_df['Date'] < event_date)]
        g1 = g1_row['Rating'].iloc[-1] if len(g1_row) > 0 else 1500
        g2 = g2_row['Rating'].iloc[-1] if len(g2_row) > 0 else 1500
        g1_rd = g1_row['RD'].iloc[-1] if len(g1_row) > 0 else 350
        g2_rd = g2_row['RD'].iloc[-1] if len(g2_row) > 0 else 350
        
        b1 = fighter_bio.get(real_fid1, {'height': 175, 'reach': 175, 'stance': 'Orthodox'})
        b2 = fighter_bio.get(real_fid2, {'height': 175, 'reach': 175, 'stance': 'Orthodox'})
        
        # Build features
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
            'activity_diff': 0, 'head_pct_diff': 0, 'body_pct_diff': 0, 'leg_pct_diff': 0,
            'dist_pct_diff': 0, 'clinch_pct_diff': 0, 'ground_pct_diff': 0,
            'is_apex': 0, 'is_altitude': 0,
            'stance_1': b1['stance'], 'stance_2': b2['stance'], 'weight_class': '155 lbs'
        }
        
        try:
            df_feat = pd.DataFrame([features])
            model_prob_f1 = model.predict_proba(df_feat)[0][1]
        except: continue
        
        # Market probabilities
        mp1 = decimal_to_implied_prob(odds_1)
        mp2 = decimal_to_implied_prob(odds_2)
        total_mp = mp1 + mp2
        mp1, mp2 = mp1 / total_mp, mp2 / total_mp
        
        # Actual result
        if flipped:
            actual_winner = 2 if fight_row['Result_1'] == 'W' else 1
        else:
            actual_winner = 1 if fight_row['Result_1'] == 'W' else 2
        
        # === FAVORITES STRATEGY ===
        s = strategies['favorites']
        for side, odds, model_p, market_p in [(1, odds_1, model_prob_f1, mp1), 
                                               (2, odds_2, 1-model_prob_f1, mp2)]:
            edge = model_p - market_p
            if (s['min_edge'] <= edge <= s['max_edge'] and 
                s['odds_range'][0] <= odds <= s['odds_range'][1]):
                stake = s['bankroll'] * s['stake_pct']
                s['bets'] += 1
                s['wagered'] += stake
                
                if side == actual_winner:
                    pnl = stake * (odds - 1)
                    s['wins'] += 1
                else:
                    pnl = -stake
                
                s['bankroll'] += pnl
                s['pnl'] += pnl
                s['results'].append({'date': event_date, 'edge': edge, 'odds': odds, 'won': side == actual_winner, 'pnl': pnl})
                break  # Only one bet per fight
        
        # === UNDERDOGS STRATEGY ===
        s = strategies['underdogs']
        for side, odds, model_p, market_p, td_adv, reach_adv in [
            (1, odds_1, model_prob_f1, mp1, st1['td_avg'] - st2['td_avg'], b1['reach'] - b2['reach']),
            (2, odds_2, 1-model_prob_f1, mp2, st2['td_avg'] - st1['td_avg'], b2['reach'] - b1['reach'])
        ]:
            edge = model_p - market_p
            # Underdog conditions: high odds, edge, AND (grappling OR reach advantage)
            has_physical_adv = (td_adv >= s['min_td_adv']) or (reach_adv >= s['min_reach_adv'])
            if (s['min_edge'] <= edge <= s['max_edge'] and 
                s['odds_range'][0] <= odds <= s['odds_range'][1] and
                has_physical_adv):
                stake = s['bankroll'] * s['stake_pct']
                s['bets'] += 1
                s['wagered'] += stake
                
                if side == actual_winner:
                    pnl = stake * (odds - 1)
                    s['wins'] += 1
                else:
                    pnl = -stake
                
                s['bankroll'] += pnl
                s['pnl'] += pnl
                s['results'].append({'date': event_date, 'edge': edge, 'odds': odds, 'won': side == actual_winner, 'pnl': pnl})
                break
        
        # Update stats after prediction
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
            
            actual_fid1 = fight_row['Fighter_Id_1']
            actual_fid2 = fight_row['Fighter_Id_2']
            
            if actual_fid1 not in stats_tracker: stats_tracker[actual_fid1] = FighterStats()
            if actual_fid2 not in stats_tracker: stats_tracker[actual_fid2] = FighterStats()
            
            td1_a = td1 / 0.4 if td1 > 0 else 0
            td2_a = td2 / 0.4 if td2 > 0 else 0
            
            stats_tracker[actual_fid1].update(fight_row['Result_1'], event_date, time_sec, str1, str2,
                                              td1, td1_a, td2_a, td2, kd1, sub1, ctrl1, sig_acc1)
            stats_tracker[actual_fid2].update(fight_row['Result_2'], event_date, time_sec, str2, str1,
                                              td2, td2_a, td1_a, td1, kd2, sub2, ctrl2, sig_acc2)
        except: pass
    
    # Report
    print("\n" + "="*60)
    for name, s in strategies.items():
        print(f"\n--- {name.upper()} STRATEGY ---")
        print(f"Initial: $1000 | Final: ${s['bankroll']:.2f}")
        print(f"P/L: ${s['pnl']:.2f} | ROI: {s['pnl']/10:.2f}%")
        if s['bets'] > 0:
            print(f"Bets: {s['bets']} | Wins: {s['wins']} ({s['wins']/s['bets']*100:.1f}%)")
            print(f"Wagered: ${s['wagered']:.2f} | Yield: {s['pnl']/s['wagered']*100:.2f}%")
        
        if s['results']:
            pd.DataFrame(s['results']).to_csv(
                os.path.join(data_dir, f'backtest_{name}_strategy.csv'), index=False)
            print(f"Results saved to backtest_{name}_strategy.csv")

if __name__ == "__main__":
    run_backtest()
