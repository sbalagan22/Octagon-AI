# 🥊 Octagon AI: The "Modern Era" UFC Prediction Engine (v16)

![Octagon AI Logo](dashboard/public/logo.png)

**Octagon AI** is a research-grade predictive intelligence framework designed specifically for Mixed Martial Arts (MMA). By synthesizing over 20 years of point-in-time historical data, Octagon AI identifies statistical edges in the UFC's "Modern Era" (2005–Present) through a multi-stage machine learning pipeline.

[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-brightgreen)](https://your-vercel-url.vercel.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-orange.svg)](https://catboost.ai)

---

## 📊 Performance Summary

| Metric | Value |
| :--- | :--- |
| **Historical Accuracy** | 60.1% (blind test 2019-2024) |
| **Backtest Period** | 2010-2024 (6,289 fights) |
| **Favorites Strategy ROI** | +14.2% (63 bets, 58.7% WR) |
| **Underdogs Strategy ROI** | +8.98% (204 bets, 31.9% WR) |

---

## 🧠 Part 1: Model Training Pipeline

### 1.1 Data Collection & Preprocessing

The model is trained on **16,000+ UFC fights** spanning 2005-2024, sourced from:

```
Data Sources:
├── UFC Stats (ufcstats.com) — Official per-fight statistics
├── ESPN MMA — Fighter bios, reach, height, stance
└── Historical odds (2010-2024) — Closing line prices
```

**Preprocessing Steps:**

1. **Name Normalization**: All fighter names are normalized using Unicode decomposition to handle accents (e.g., "José Aldo" → "jose aldo")

2. **Feature Extraction**: For each fight, we extract 30+ features as differentials between Fighter 1 and Fighter 2:
   ```python
   feature = fighter_1_stat - fighter_2_stat
   ```

3. **Temporal Integrity**: All features are computed using **only data available before fight night** (no future leakage)

### 1.2 Feature Engineering

The model uses **Point-in-Time Exponential Moving Averages (EMA)** with α=0.3 to capture "Current Form":

```python
EMA_new = α × current_fight_stat + (1-α) × EMA_previous

# α = 0.3 means:
# - Last fight: 30% weight
# - 2 fights ago: 21% weight  
# - 3 fights ago: 14.7% weight
# - 4 fights ago: 10.3% weight
```

**Complete Feature Set (30 features):**

| Category | Features | Description |
| :--- | :--- | :--- |
| **Skill Rating** | `glicko_diff`, `glicko_rd_diff` | Glicko-2 rating and uncertainty differential |
| **Physical** | `height_diff`, `reach_diff`, `age_diff` | Biological advantages |
| **Striking** | `slpm_diff`, `sapm_diff`, `sig_acc_diff` | Output, absorption, accuracy |
| **Grappling** | `td_avg_diff`, `td_acc_diff`, `td_def_diff` | Takedown volume, success, defense |
| **Control** | `ctrl_diff`, `sub_diff` | Ground control time, submission threat |
| **Finishing** | `kd_diff` | Knockdown rate differential |
| **Target/Position** | `head_%`, `body_%`, `leg_%`, `distance_%`, `clinch_%`, `ground_%` | Strike distribution differentials |
| **Form** | `win_rate_diff`, `streak_diff`, `rust_diff` | Recent performance indicators |
| **Categorical** | `stance_1`, `stance_2`, `weight_class` | Fighting style and division |
| **Context** | `is_apex`, `is_altitude` | Venue-specific factors |

### 1.3 Glicko-2 Rating System

Unlike simple ELO, Octagon AI uses **Glicko-2** which tracks three values per fighter:

| Component | Symbol | Description |
| :--- | :--- | :--- |
| Rating | r | Skill estimate (starts at 1500) |
| Rating Deviation | RD | Uncertainty (high for newcomers, ~350) |
| Volatility | σ | How consistently the fighter performs |

**Rating Update Formula:**
```
r' = r + q × g(RD_opponent) × (outcome - expected)

where:
- q = ln(10) / 400 ≈ 0.00575
- g(RD) = 1 / sqrt(1 + 3q²RD² / π²)
- expected = 1 / (1 + 10^((r_opp - r) / 400))
```

**Legacy Bias Mitigation:**
After 6 months of inactivity:
- Rating regresses toward 1500 by **1% per month**
- RD expands by **15 points per period**
- Prevents "Paper Champion" syndrome where inactive legends keep high ratings

### 1.4 Feature Importance Analysis

The trained CatBoost model reveals which factors most influence predictions:

| Rank | Feature | Importance | Interpretation |
| :--- | :--- | :--- | :--- |
| **1** | `reach_diff` | **27.88%** | Longer reach = jab dominance, distance control |
| **2** | `glicko_diff` | **10.83%** | Higher skill rating = more wins |
| **3** | `weight_class` | **6.97%** | Division-specific fighting styles matter |
| **4** | `sapm_diff` | **6.27%** | Getting hit less = better defense |
| **5** | `td_avg_diff` | **5.88%** | Takedown volume = grappling pressure |
| **6** | `rust_diff` | **5.16%** | Ring rust is a real factor |
| **7** | `age_diff` | **4.45%** | Athletic prime vs decline |
| **8** | `glicko_rd_diff` | **3.95%** | Uncertainty differential |
| **9** | `slpm_diff` | **2.96%** | Offensive output |
| **10** | `ctrl_diff` | **2.01%** | Ground control dominance |

**Key Insight**: Physical attributes (reach, height) account for ~30% of predictive power — the market systematically undervalues these.

### 1.5 Model Architecture

```python
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV

# Base model
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10.0,      # High regularization prevents Glicko over-reliance
    loss_function='Logloss',
    boosting_type='Ordered',  # Critical for temporal data
    cat_features=['stance_1', 'stance_2', 'weight_class']
)

# Probability calibration (Platt scaling)
calibrated_model = CalibratedClassifierCV(
    model, 
    method='sigmoid',  # Platt scaling
    cv=5
)
```

**Why These Choices:**

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| `boosting_type='Ordered'` | Crucial | Prevents lookahead leakage in temporal fight sequences |
| `l2_leaf_reg=10.0` | High | Prevents over-reliance on any single feature (like Glicko) |
| `depth=6` | Moderate | Captures interactions without overfitting |
| `CalibratedClassifierCV` | Sigmoid | Converts raw scores to true probabilities |

---

## 🎯 Part 2: Prediction Pipeline

### 2.1 How Predictions Are Generated

For each upcoming fight:

```python
def predict_fight(fighter_1, fighter_2, event_date):
    # 1. Get point-in-time stats (EMA up to but NOT including today)
    stats_1 = get_fighter_ema_stats(fighter_1, cutoff=event_date)
    stats_2 = get_fighter_ema_stats(fighter_2, cutoff=event_date)
    
    # 2. Get Glicko ratings (before today)
    glicko_1 = get_glicko_rating(fighter_1, before=event_date)
    glicko_2 = get_glicko_rating(fighter_2, before=event_date)
    
    # 3. Get biological data
    bio_1, bio_2 = get_fighter_bios(fighter_1, fighter_2)
    
    # 4. Compute differentials
    features = {
        'reach_diff': bio_1['reach'] - bio_2['reach'],
        'glicko_diff': min(max(glicko_1 - glicko_2, -250), 250),  # Capped
        'td_avg_diff': stats_1['td_avg'] - stats_2['td_avg'],
        # ... all 30 features
    }
    
    # 5. Predict
    prob_f1_wins = model.predict_proba([features])[0][1]
    
    return prob_f1_wins
```

### 2.2 Probability Calibration

Raw ML outputs are often overconfident. We use **Platt Scaling** to calibrate:

```
P(win) = 1 / (1 + exp(A × raw_score + B))

where A, B are fitted parameters from cross-validation
```

**Calibration Metrics:**

| Metric | Description | v16 Value |
| :--- | :--- | :--- |
| **Log Loss** | Cross-entropy between predicted and actual | 0.6442 |
| **Brier Score** | Mean squared error of probabilities | 0.2267 |
| **Calibration Slope** | Should be ~1.0 for perfect calibration | 0.98 |

### 2.3 Handling Edge Cases

| Scenario | Solution |
| :--- | :--- |
| **UFC Newcomer** | Use regional record + default Glicko (1500, RD=350) |
| **Missing reach data** | Impute with weight class average |
| **Fight cancelled** | No stats update (EMA unchanged) |
| **Draw/NC result** | Update Glicko with 0.5 expected outcome |

---

## 💰 Part 3: Betting Strategy Framework

### 3.1 The "Edge + Confirmation" Philosophy

We only bet when:
1. **Model Edge** ≥ threshold (model believes fighter is undervalued)
2. **Odds in target range** (not too extreme)
3. **Physical confirmation** (for underdogs: TD or reach advantage)

### 3.2 Strategy 1: Conservative Favorites

**Target ROI: 10-15%** | **Risk: Low-Medium**

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Edge Range** | 12-17% | "Goldilocks zone" — statistically validated optimal range |
| **Odds Range** | 1.75 - 2.00 | Slight favorites where market is uncertain |
| **Stake Size** | 2% of bankroll | Quarter-Kelly for safety |

**The Math:**
```
Expected Value = (Win_Rate × Avg_Payout) - 1
EV = (0.587 × 1.85) - 1 = +8.6%

Kelly Optimal = Edge / (Odds - 1)
Kelly = 0.14 / 0.85 = 16.5%
Quarter-Kelly = 4.1% → We use 2% for extra safety
```

**Backtest Results (2010-2024):**
- Total Bets: 63
- Wins: 37 (58.7%)
- Final Bankroll: $1,141.95 (+14.2% ROI)
- Yield: 11.09% per dollar wagered

### 3.3 Strategy 2: Smart Underdogs

**Target ROI: 5-10%** | **Risk: High**

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Edge Range** | 25-50% | Model sees significant market mispricing |
| **Odds Range** | 2.50 - 4.50 | Medium underdogs, not crazy longshots |
| **Stake Size** | 1% of bankroll | Smaller due to high variance |
| **Required** | TD adv ≥ 1.0 OR Reach adv ≥ 3cm | Physical edge confirmation |

**Why This Works:**
The market systematically undervalues:
1. **Grapplers as underdogs** — Wrestlers who can neutralize strikers
2. **Reach-advantaged underdogs** — Longer fighters who control distance

**Backtest Results (2010-2024):**
- Total Bets: 204
- Wins: 65 (31.9%)
- Final Bankroll: $1,089.76 (+8.98% ROI)
- Yield: 3.96% per dollar wagered

### 3.4 Combined Portfolio

Running both strategies simultaneously:
- **Combined P/L**: +$231.71
- **Combined ROI**: ~11.6%
- **Total Bets**: 267 over 14 years

---

## 🧪 Part 4: Backtesting Methodology

### 4.1 The "n+1" Zero-Leakage Protocol

At Time T (fight night), the model can **only** access data from Time < T:

```python
stats_tracker = {}  # Empty at start

for fight in chronologically_sorted_fights:
    # 1. EXTRACT pre-fight stats
    stats_f1 = stats_tracker.get(fighter_1, default_newcomer_stats)
    stats_f2 = stats_tracker.get(fighter_2, default_newcomer_stats)
    
    # 2. PREDICT (using only past data)
    prob = model.predict(compute_features(stats_f1, stats_f2))
    
    # 3. BET if edge exists
    if meets_betting_criteria(prob, odds):
        log_bet(...)
    
    # 4. UPDATE stats AFTER prediction
    stats_tracker[fighter_1].update(fight_stats_1)
    stats_tracker[fighter_2].update(fight_stats_2)
```

### 4.2 Validation Checks

| Check | Status | Method |
| :--- | :--- | :--- |
| Chronological order | ✅ | All fights sorted by date |
| No duplicate bets | ✅ | `seen_fights` set prevents re-betting |
| Future data filtered | ✅ | 2025+ rows excluded |
| Stats lagged properly | ✅ | EMA updated AFTER prediction |

### 4.3 Monte Carlo Validation

10,000 simulations confirmed statistical significance:

| Assumed True WR | Expected ROI | P(observing our results) |
| :--- | :--- | :--- |
| 50% (random) | 0% | 1.78% |
| 55% | +0.4% | 9.84% |
| **58%** | **+11.2%** | **21.18%** |
| 60% | +18.2% | 31.90% |

**Conclusion**: True model edge is likely 55-60%, producing sustainable 8-15% ROI.

---

## 💻 Tech Stack & Pipeline

### Backend
| Component | Technology |
| :--- | :--- |
| Data Engine | Pandas / NumPy / Joblib |
| Modeling | CatBoost / Scikit-Learn |
| Scraper | BeautifulSoup4 / Requests |

### Frontend
| Component | Technology |
| :--- | :--- |
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Charts | Recharts / Radar Charts |

### Automation
```yaml
# .github/workflows/weekly_update.yml
schedule:
  - cron: '0 5 * * 1'  # Every Monday 5AM UTC
jobs:
  - Scrape previous event results
  - Regenerate Glicko-2 ratings  
  - Generate new predictions
  - Deploy to Vercel
```

---

## 🚀 Local Development

```bash
# Install dependencies
pip install pandas numpy scikit-learn catboost joblib beautifulsoup4

# Regenerate Glicko-2 ratings
python3 src/generate_glicko.py

# Train the model
python3 src/train_model.py

# Generate predictions
python3 src/predict_model.py

# Run dashboard
cd dashboard && npm run dev
```

---

**Octagon AI** — *Predictive excellence through tactical data science.*
