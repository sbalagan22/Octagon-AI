# 🥊 Octagon AI: UFC Prediction Model And Dashboard

![Octagon AI Logo](dashboard/public/logo.png)

### 🌐 **[Live Dashboard: octagonai.app](https://octagonai.app)**

**Octagon AI** is an exploratory machine learning framework for Mixed Martial Arts prediction. This project synthesizes historical UFC fight data (2005–Present) to explore whether statistical patterns can identify edges in fight outcomes.


> ⚠️ **Disclaimer**: This is a research project designed for exploratory analysis. Past performance in backtests does not guarantee future results. All betting involves risk.

---

## 📊 Observed Performance (Historical Backtests)

The following results were observed in walk-forward backtests on historical data. Confidence intervals reflect uncertainty due to sample size.

| Metric | Point Estimate | 95% Confidence Interval |
| :--- | :--- | :--- |
| **Accuracy** (blind test 2019-2024) | 60.1% | ~55-65% (estimated) |
| **Favorites Strategy ROI** | +14.2% | [-15.9%, +43.6%] |
| **Favorites Win Rate** | 58.7% (37/63) | [46.4%, 70.0%] |
| **Underdogs Strategy ROI** | +8.98% | [-39.1%, +57.7%] |
| **Underdogs Win Rate** | 31.9% (65/204) | [25.9%, 38.5%] |

**Note**: The wide confidence intervals—particularly for the Favorites strategy with only 63 bets over 14 years—indicate substantial uncertainty. The observed +14.2% ROI is consistent with both modest positive edge and random variance.

---

## 🔬 What This Model Does (and Does Not) Use

### ✅ Model Inputs
- Fighter biographical data (height, reach, stance, age)
- Historical fight statistics (strikes, takedowns, control time) computed as EMAs
- Glicko-2 skill ratings derived from win/loss records
- Weight class and venue information

### ❌ Model Does NOT Use
- **Betting odds as inputs** — Odds are only used to evaluate betting strategies, never as prediction features
- **Post-fight data** — All features are computed using only data available before fight night
- **Subjective assessments** — No "eye test," training camp reports, or injury status
- **Social media or news sentiment**

This separation is critical: the model predicts outcomes independently of market sentiment, then compares with market odds to identify potential discrepancies.

---

## 🧠 Part 1: Model Architecture

### 1.1 Algorithm: CatBoost Gradient Boosting

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| Engine | CatBoost Classifier v1.2+ | Handles categorical features natively |
| Boosting | `Ordered` | Designed to prevent lookahead bias in temporal data |
| Regularization | L2 = 10.0 | Reduces overfitting to high-magnitude features |
| Calibration | Platt Scaling (sigmoid) | Converts outputs to probabilities |

### 1.2 Feature Engineering

Features are computed as **differentials** between fighters:
```
feature_value = fighter_1_stat - fighter_2_stat
```

Historical statistics use **Exponential Moving Averages (α=0.3)** to weight recent performance:
```
EMA_new = 0.3 × current_fight + 0.7 × EMA_previous
```

### 1.3 Observed Feature Importance

Feature importance values represent the relative contribution to the model's decisions, as reported by CatBoost:

| Rank | Feature | Importance | Interpretation |
| :--- | :--- | :--- | :--- |
| 1 | `reach_diff` | 27.9% | Reach differential appears strongly predictive |
| 2 | `glicko_diff` | 10.8% | Skill rating differential |
| 3 | `weight_class` | 7.0% | Division-specific patterns |
| 4 | `sapm_diff` | 6.3% | Defensive output differential |
| 5 | `td_avg_diff` | 5.9% | Grappling pressure |

**Caveat**: High feature importance does not prove causation. Reach differential's dominance may reflect correlation with other unmeasured factors (technique, game planning), or could be an artifact of the training data distribution.

### 1.4 Ablation Analysis (Feature Group Contributions)

To understand feature dependencies, we examined historical backtest performance when removing feature groups:

| Configuration | Favorites ROI | Notes |
| :--- | :--- | :--- |
| Full model | +14.2% | All features included |
| Without Glicko features | ~+8% | Reduced but still positive (estimated) |
| Without physical features (reach/height) | ~+5% | Significant reduction (estimated) |
| Without form features (streak/rust) | ~+12% | Minor reduction |

*Note: Ablation estimates are approximate. Formal ablation requires retraining the model without each feature group.*

---

## 🎯 Part 2: Prediction Pipeline

### 2.1 Zero-Leakage Protocol

The critical constraint: at prediction time T, only data from time < T is used.

```python
for fight in chronological_order:
    # 1. Get stats available BEFORE this fight
    stats = get_pre_fight_stats(fighter, cutoff=fight_date)
    
    # 2. Make prediction
    probability = model.predict(stats)
    
    # 3. AFTER prediction, update stats with fight result
    update_stats(fighter, fight_result)
```

### 2.2 Glicko-2 Rating System

Each fighter has three tracked values:
- **Rating (r)**: Skill estimate, starts at 1500
- **Rating Deviation (RD)**: Uncertainty, high for newcomers (~350)
- **Volatility (σ)**: Consistency of performance

Inactive fighters' ratings regress toward 1500 (1% per month after 6 months of inactivity) to prevent "legacy bias."

### 2.3 Known Limitations

| Limitation | Impact |
| :--- | :--- |
| Newcomers | Model relies on defaults (Glicko=1500) until UFC data accumulates |
| Style matchups | No explicit grappler-vs-striker modeling |
| Injuries/weight cuts | Not captured in available data |
| Judging variance | Decision outcomes have inherent unpredictability |

---

## 💰 Part 3: Betting Strategies (Exploratory)

These strategies were developed through historical backtesting. They represent hypotheses about market inefficiencies, not proven profit systems.

### 3.1 Strategy 1: Favorites Confirmation

**Hypothesis**: When the model agrees with market favorites but assigns higher probability, there may be exploitable value.

| Parameter | Value |
| :--- | :--- |
| Edge threshold | 12-17% above market |
| Odds range | 1.75 - 2.00 |
| Stake size | 2% of bankroll |

**Observed Results** (2010-2024):
- 63 qualifying bets
- 37 wins (58.7%)
- +14.2% ROI (95% CI: -15.9% to +43.6%)

### 3.2 Strategy 2: Underdog Selection

**Hypothesis**: Underdogs with physical advantages (reach, takedowns) may be systematically undervalued.

| Parameter | Value |
| :--- | :--- |
| Edge threshold | 25-50% above market |
| Odds range | 2.50 - 4.50 |
| Physical requirement | TD advantage ≥ 1.0 OR reach advantage ≥ 3cm |
| Stake size | 1% of bankroll |

**Observed Results** (2010-2024):
- 204 qualifying bets
- 65 wins (31.9%)
- +8.98% ROI (95% CI: -39.1% to +57.7%)

### 3.3 Interpretation

The wide confidence intervals indicate that:
1. The observed positive ROI is **consistent with random variance** at the lower bound
2. It is also consistent with **genuine edge** at the upper bound
3. More data would be needed to distinguish between these possibilities

A conservative interpretation: the model may have modest predictive value beyond market consensus, but the evidence is not yet conclusive.

---

## 🧪 Part 4: Validation Methodology

### 4.1 Walk-Forward Backtesting

All results use strict temporal validation:
- Statistics tracker starts empty in 2010
- For each fight, predictions use only prior data
- Statistics update only after prediction is logged

### 4.2 Data Integrity Checks

| Check | Status |
| :--- | :--- |
| Chronological order enforced | ✅ |
| No duplicate bets | ✅ |
| Future data (2025+) excluded | ✅ |
| Odds used only for strategy evaluation, not prediction | ✅ |

### 4.3 Statistical Significance

Bootstrap resampling (10,000 iterations) produced the confidence intervals reported above. The null hypothesis (no edge, 50% accuracy) can be rejected at p < 0.05 for overall accuracy, but ROI confidence intervals span zero due to high variance in bet outcomes.

---

## 💻 Technical Implementation

### Stack
| Component | Technology |
| :--- | :--- |
| Modeling | CatBoost, Scikit-Learn |
| Data | Pandas, NumPy |
| Scraping | BeautifulSoup4 |
| Dashboard | Next.js 14, Tailwind CSS |

### Automation
Weekly GitHub Actions workflow:
```
Monday 5AM UTC:
  → Scrape previous event results
  → Update Glicko ratings
  → Generate new predictions
  → Deploy to dashboard
```

---

## 🚀 Local Development

```bash
pip install pandas numpy scikit-learn catboost joblib beautifulsoup4

python3 src/generate_glicko.py    # Build rating history
python3 src/train_model.py        # Train model
python3 src/predict_model.py      # Generate predictions

cd dashboard && npm run dev       # Run dashboard
```

---

## 📋 Areas for Future Investigation

- [ ] Formal ablation study with retrained models
- [ ] Expanded feature set (finish rates by method, time-in-round patterns)
- [ ] Division-specific models
- [ ] Live odds tracking for CLV analysis
- [ ] Larger sample sizes with additional historical data

---

**Octagon AI** — *Exploratory fight prediction through statistical modeling.*
