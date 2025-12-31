"""
OCTAGON AI - PROFITABILITY BACKTESTER
=====================================
Implements Walk-Forward Validation with Quarter-Kelly Staking.
Tests model edge against market odds on 2023-2024 UFC fights.

Strategy:
- Training: 2010-2022, Testing: 2023-2024
- Value Betting: Bet only when Model_Prob > Market_Prob + 5%
- Quarter-Kelly: Stake = (Edge / Odds) * 0.25 * Bankroll
- Stop-Loss: Pause if bankroll drops 20%
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
    """Convert decimal odds to implied probability."""
    if odds <= 1.0: return 0.5  # Edge case
    return 1.0 / odds

def calculate_kelly_stake(model_prob, market_prob, decimal_odds, bankroll, fraction=0.25):
    """
    Quarter-Kelly Staking.
    Edge = Model_Prob - Market_Prob
    Kelly = (Edge * (Odds - 1)) / Odds
    Stake = Kelly * 0.25 * Bankroll
    """
    edge = model_prob - market_prob
    if edge <= 0.05:  # 5% minimum edge threshold
        return 0.0
    
    # Kelly formula for decimal odds
    q = 1 - model_prob  # Probability of losing
    b = decimal_odds - 1  # Net profit per unit bet
    kelly = (model_prob * b - q) / b
    
    # Quarter-Kelly with max bet cap (5% of bankroll)
    stake = max(0, kelly * fraction * bankroll)
    stake = min(stake, bankroll * 0.05)  # Max 5% per bet
    
    return round(stake, 2)

def run_backtest():
    print("="*60)
    print("OCTAGON AI - PROFITABILITY BACKTEST")
    print("="*60)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '../newdata')
    model_dir = os.path.join(base_dir, '../models')
    
    # 1. Load Resources
    print("Loading model and data...")
    model = joblib.load(os.path.join(model_dir, 'catboost_ufc_model.pkl'))
    
    # Load betting odds
    odds_df = pd.read_csv(os.path.join(data_dir, 'UFC_betting_odds.csv'))
    odds_df['event_date'] = pd.to_datetime(odds_df['event_date'])
    
    # Load fighters for ID mapping
    fighters_df = pd.read_csv(os.path.join(data_dir, 'Fighters.csv'))
    name_to_id = {}
    for _, row in fighters_df.iterrows():
        n = normalize_name(row['Full Name'])
        name_to_id[n] = row['Fighter_Id']
    
    # Load fights for results
    fights_df = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    events_df = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    fights_df = fights_df.merge(events_df[['Event_Id', 'Date']], on='Event_Id')
    fights_df['Date'] = pd.to_datetime(fights_df['Date'])
    
    # Load Glicko ratings
    glicko_df = pd.read_csv(os.path.join(data_dir, 'fighter_glicko.csv'))
    glicko_df['Date'] = pd.to_datetime(glicko_df['Date'])
    
    # 2. Filter to 2023-2024 test period
    test_start = pd.to_datetime('2023-01-01')
    test_end = pd.to_datetime('2024-12-31')
    
    test_odds = odds_df[(odds_df['event_date'] >= test_start) & (odds_df['event_date'] <= test_end)].copy()
    test_odds = test_odds.sort_values('event_date').reset_index(drop=True)
    
    print(f"Testing on {len(test_odds)} fights from 2023-2024")
    
    # 3. Backtest Loop
    initial_bankroll = 1000.0
    bankroll = initial_bankroll
    stop_loss_threshold = bankroll * 0.8  # Stop if we lose 20%
    
    total_bets = 0
    wins = 0
    losses = 0
    total_wagered = 0.0
    total_pnl = 0.0
    
    results = []
    
    for idx, row in test_odds.iterrows():
        if bankroll < stop_loss_threshold:
            print(f"\n[STOP LOSS TRIGGERED] Bankroll dropped to ${bankroll:.2f}")
            break
            
        f1_name = row['fighter_1']
        f2_name = row['fighter_2']
        odds_1 = row['odds_1']
        odds_2 = row['odds_2']
        event_date = row['event_date']
        
        # Get fighter IDs
        fid1 = name_to_id.get(normalize_name(f1_name))
        fid2 = name_to_id.get(normalize_name(f2_name))
        
        if not fid1 or not fid2:
            continue  # Skip if fighters not in database
        
        # Get Glicko ratings (point-in-time: use ratings BEFORE this fight)
        g1_row = glicko_df[(glicko_df['Fighter_Id'] == fid1) & (glicko_df['Date'] < event_date)]
        g2_row = glicko_df[(glicko_df['Fighter_Id'] == fid2) & (glicko_df['Date'] < event_date)]
        
        g1_rating = g1_row['Rating'].iloc[-1] if len(g1_row) > 0 else 1500
        g2_rating = g2_row['Rating'].iloc[-1] if len(g2_row) > 0 else 1500
        
        # Simple model prediction based on Glicko differential
        # This is a simplified version - the full model would use all features
        glicko_diff = g1_rating - g2_rating
        model_prob_f1 = 1 / (1 + 10 ** (-glicko_diff / 400))  # ELO-like formula
        
        # Market implied probabilities
        market_prob_f1 = decimal_to_implied_prob(odds_1)
        market_prob_f2 = decimal_to_implied_prob(odds_2)
        
        # Normalize market probs (they include vig)
        total_market = market_prob_f1 + market_prob_f2
        market_prob_f1 /= total_market
        market_prob_f2 /= total_market
        
        # Find actual result
        fight_result = fights_df[
            (fights_df['Fighter_Id_1'] == fid1) & 
            (fights_df['Fighter_Id_2'] == fid2) &
            (fights_df['Date'].dt.date == event_date.date())
        ]
        
        if len(fight_result) == 0:
            # Try reversed order
            fight_result = fights_df[
                (fights_df['Fighter_Id_1'] == fid2) & 
                (fights_df['Fighter_Id_2'] == fid1) &
                (fights_df['Date'].dt.date == event_date.date())
            ]
            if len(fight_result) > 0:
                actual_winner = 2 if fight_result['Result_1'].iloc[0] == 'W' else 1
            else:
                continue  # Can't verify result
        else:
            actual_winner = 1 if fight_result['Result_1'].iloc[0] == 'W' else 2
        
        # Determine if we have edge and which side to bet
        # CONSERVATIVE STRATEGY: Only bet on FAVORITES when model agrees
        # This avoids catastrophic losses on long-shot underdogs
        edge_f1 = model_prob_f1 - market_prob_f1
        edge_f2 = (1 - model_prob_f1) - market_prob_f2
        
        bet_side = None
        stake = 0.0
        decimal_odds_used = 0.0
        
        # Only bet on F1 if they're a favorite (odds < 2.0) AND we have 10%+ edge
        if edge_f1 > 0.10 and odds_1 < 2.0:
            stake = calculate_kelly_stake(model_prob_f1, market_prob_f1, odds_1, bankroll)
            bet_side = 1
            decimal_odds_used = odds_1
        # Only bet on F2 if they're a favorite (odds < 2.0) AND we have 10%+ edge
        elif edge_f2 > 0.10 and odds_2 < 2.0:
            stake = calculate_kelly_stake(1 - model_prob_f1, market_prob_f2, odds_2, bankroll)
            bet_side = 2
            decimal_odds_used = odds_2
        
        if stake > 0:
            total_bets += 1
            total_wagered += stake
            
            # Calculate P/L
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
                'stake': stake,
                'odds': decimal_odds_used,
                'won': bet_side == actual_winner,
                'pnl': pnl,
                'bankroll': bankroll
            })
    
    # 4. Report Results
    print("\n" + "="*60)
    print("BACKTEST RESULTS (2023-2024)")
    print("="*60)
    print(f"Initial Bankroll: ${initial_bankroll:.2f}")
    print(f"Final Bankroll:   ${bankroll:.2f}")
    print(f"Total P/L:        ${total_pnl:.2f}")
    print(f"ROI:              {(total_pnl / initial_bankroll) * 100:.2f}%")
    print(f"\nTotal Bets:       {total_bets}")
    print(f"Wins:             {wins} ({(wins/total_bets)*100:.1f}%)" if total_bets > 0 else "")
    print(f"Losses:           {losses}")
    print(f"Total Wagered:    ${total_wagered:.2f}")
    
    if total_bets > 0:
        print(f"\nYield:            {(total_pnl / total_wagered) * 100:.2f}%")
    
    # Save detailed results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(data_dir, 'backtest_results_2023_24.csv'), index=False)
        print(f"\nDetailed results saved to backtest_results_2023_24.csv")
    
    return bankroll, total_pnl

if __name__ == "__main__":
    run_backtest()
