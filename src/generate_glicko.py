import pandas as pd
import numpy as np
import math
import os
from datetime import datetime

"""
OCTAGON AI - GLICKO-2 IMPLEMENTATION
Tracks Fighter Rating (r), Rating Deviation (RD), and Volatility (sigma).
Features:
- Time-based RD decay (Ring Rust).
- ID-based tracking.
- Full chronological history generation.
"""

# --- Glicko-2 Constants ---
TAU = 0.5           # Constraint on volatility change (0.3 to 1.2 usually)
Min_RD = 30.0       # Minimum Rating Deviation (cannot optimize to perfection)
Max_RD = 350.0      # Unranked / Maximum Uncertainty
Default_Rating = 1500.0
Default_Vol = 0.06

class GlickoFighter:
    def __init__(self, fighter_id):
        self.id = fighter_id
        self.rating = Default_Rating
        self.rd = Max_RD
        self.vol = Default_Vol
        self.last_fight_date = None

    def get_glicko2_scale(self):
        """Convert to Glicko-2 scale for calculation"""
        mu = (self.rating - 1500.0) / 173.7178
        phi = self.rd / 173.7178
        return mu, phi

    def update_from_glicko2(self, mu, phi, sigma):
        """Convert back to standard scale"""
        self.rating = 173.7178 * mu + 1500.0
        self.rd = 173.7178 * phi
        self.vol = sigma
        
        # Clamp RD
        if self.rd < Min_RD: self.rd = Min_RD
        if self.rd > Max_RD: self.rd = Max_RD

def g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * (phi ** 2) / (math.pi ** 2))

def E(mu, mu_j, phi_j):
    return 1.0 / (1.0 + math.exp(-g(phi_j) * (mu - mu_j)))

def compute_new_stats(p1, p2, outcome):
    """
    Update p1's stats based on match against p2.
    outcome: 1.0 (Win), 0.5 (Draw), 0.0 (Loss)
    Returns: new_mu, new_phi, new_sigma for p1
    """
    mu, phi = p1.get_glicko2_scale()
    mu_j, phi_j = p2.get_glicko2_scale()
    
    # 1. Variance
    g_phi_j = g(phi_j)
    e_val = E(mu, mu_j, phi_j)
    v = 1.0 / ((g_phi_j ** 2) * e_val * (1 - e_val))
    
    # 2. Delta
    delta = v * g_phi_j * (outcome - e_val)
    
    # 3. New Volatility (sigma') - Iterative
    sigma = p1.vol
    a = math.log(sigma ** 2)
    
    def f(x):
        exp_x = math.exp(x)
        t1 = (exp_x * (delta ** 2 - phi ** 2 - v - exp_x)) / (2.0 * ((phi ** 2 + v + exp_x) ** 2))
        t2 = (x - a) / (TAU ** 2)
        return t1 - t2
        
    # Illinois Algorithm / Newton-Raphson approx
    A = a
    if (delta ** 2) > (phi ** 2 + v):
        B = math.log(delta ** 2 - phi ** 2 - v)
    else:
        k = 1
        while f(a - k * TAU) < 0:
            k += 1
        B = a - k * TAU
        
    fa = f(A)
    fb = f(B)
    
    # Convergence
    epsilon = 0.000001
    while abs(B - A) > epsilon:
        C = A + (A - B) * fa / (fb - fa)
        fc = f(C)
        if fc * fb < 0:
            A = B
            fa = fb
        else:
            fa = fa / 2.0
        B = C
        fb = fc
        
    new_sigma = math.exp(A / 2.0)
    
    # 4. New RD (phi')
    phi_star = math.sqrt(phi ** 2 + new_sigma ** 2)
    new_phi = 1.0 / math.sqrt(1.0 / (phi_star ** 2) + 1.0 / v)
    
    # 5. New Rating (mu')
    new_mu = mu + (new_phi ** 2) * g_phi_j * (outcome - e_val)
    
    return new_mu, new_phi, new_sigma

def apply_inactivity_decay(fighter, current_date):
    if fighter.last_fight_date is None:
        return
        
    days_inactive = (current_date - fighter.last_fight_date).days
    
    # Rule: RD increases if inactive > 6 months (180 days)
    if days_inactive > 180:
        periods_missed = (days_inactive - 180) / 30.0
        if periods_missed > 0:
            # 1. Uncertainty increase (Aggressive)
            # Standard Glicko is often too conservative for MMA regression
            fighter.rd = min(Max_RD, math.sqrt(fighter.rd**2 + (15.0 * periods_missed)**2))
            
            # 2. Rating Regression (The "Legacy Fix")
            # Pull high ratings back toward 1500 during layoffs
            # We use an exponential decay: 1% pull per month
            if abs(fighter.rating - 1500) > 10:
                decay_factor = math.exp(-0.01 * periods_missed) 
                fighter.rating = 1500 + (fighter.rating - 1500) * decay_factor

def generate_glicko():
    print("Generating Glicko-2 Ratings...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '../newdata')
    output_dir = os.path.join(script_dir, '../newdata') # Save where data is
    
    # Load Data
    fights = pd.read_csv(os.path.join(data_dir, 'Fights.csv'))
    events = pd.read_csv(os.path.join(data_dir, 'Events.csv'))
    
    # Merge Date
    fights = fights.merge(events[['Event_Id', 'Date']], on='Event_Id', how='left')
    fights['Date'] = pd.to_datetime(fights['Date'])
    fights = fights.sort_values('Date').reset_index(drop=True)
    
    fighters = {} # ID -> GlickoFighter
    
    history_records = []
    
    # Limit processing? No, process all.
    
    for idx, row in fights.iterrows():
        id1 = row['Fighter_Id_1']
        id2 = row['Fighter_Id_2']
        date = row['Date']
        
        if pd.isna(id1) or pd.isna(id2): continue
        if row['Result_1'] not in ['W', 'L', 'D']: continue # Skip No Contests
        
        # Init fighters if new
        if id1 not in fighters: fighters[id1] = GlickoFighter(id1)
        if id2 not in fighters: fighters[id2] = GlickoFighter(id2)
        
        p1 = fighters[id1]
        p2 = fighters[id2]
        
        # Apply Inactivity Decay BEFORE match
        apply_inactivity_decay(p1, date)
        apply_inactivity_decay(p2, date)
        
        # Store Pre-Fight Ratings (This is what we use for prediction)
        history_records.append({
            'Fight_Id': row['Fight_Id'],
            'Fighter_Id': id1,
            'Opponent_Id': id2,
            'Date': date,
            'Rating': p1.rating,
            'RD': p1.rd,
            'Vol': p1.vol,
            'Opp_Rating': p2.rating,
            'Opp_RD': p2.rd
        })
        history_records.append({
            'Fight_Id': row['Fight_Id'],
            'Fighter_Id': id2,
            'Opponent_Id': id1,
            'Date': date,
            'Rating': p2.rating,
            'RD': p2.rd,
            'Vol': p2.vol,
            'Opp_Rating': p1.rating,
            'Opp_RD': p1.rd
        })
        
        # Determine Outcome
        # Result_1: W, L, D
        s1 = 1.0 if row['Result_1'] == 'W' else (0.5 if row['Result_1'] == 'D' else 0.0)
        s2 = 1.0 - s1
        
        # Compute New Stats
        # Important: Don't update p1 in place before calculating p2!
        mu1, phi1, sig1 = compute_new_stats(p1, p2, s1)
        mu2, phi2, sig2 = compute_new_stats(p2, p1, s2)
        
        # Update Fighters
        p1.update_from_glicko2(mu1, phi1, sig1)
        p2.update_from_glicko2(mu2, phi2, sig2)
        
        p1.last_fight_date = date
        p2.last_fight_date = date
        
    # Save History
    df_hist = pd.DataFrame(history_records)
    out_path = os.path.join(output_dir, 'fighter_glicko.csv')
    df_hist.to_csv(out_path, index=False)
    print(f"Saved Glicko history to {out_path} ({len(df_hist)} rows)")
    
    # Save Current Ratings (for upcoming fights)
    current_ratings = []
    for fid, f in fighters.items():
        # Apply decay to 'today' (optional, but good for display)
        # We won't decay for the file to prevent bias, predict script will decay
        current_ratings.append({
            'Fighter_Id': fid,
            'Rating': f.rating,
            'RD': f.rd,
            'Vol': f.vol,
            'Last_Fight': f.last_fight_date
        })
    
    df_curr = pd.DataFrame(current_ratings)
    df_curr.to_csv(os.path.join(output_dir, 'current_glicko.csv'), index=False)
    print("Saved current fighter ratings.")

if __name__ == "__main__":
    generate_glicko()
