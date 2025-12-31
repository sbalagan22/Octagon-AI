# 🥊 Octagon AI: The "Modern Era" UFC Prediction Engine (v16)

![Octagon AI Logo](dashboard/public/logo.png)

Octagon AI is a research-grade predictive intelligence framework designed specifically for Mixed Martial Arts (MMA). By synthesizing over 20 years of point-in-time historical data, Octagon AI identifies statistical edges in the UFC's "Modern Era" (2005–Present) through a multi-stage machine learning pipeline.

---

## 🔬 Scientific Methodology & Model Architecture (v16)

The v16 engine represents a fundamental departure from standard ELO-based models, utilizing a **gradient-boosted decision tree ensemble** specifically calibrated to neutralize "Legacy Bias" (where veterans' past glory overpowers current technical decay).

### 1. Core Algorithm: CatBoost (Ordered Boosting)
- **Engine**: CatBoost Classifier (v1.2+)
- **Boosting Type**: `Ordered` (Specifically chosen to handle the temporal nature of fight data and prevent lookahead leakage).
- **Loss Function**: `Logloss` (Optimized for minimizing entropy in probability estimation).
- **Regularization**: `L2 Leaf Regularization (10.0)` — High regularization is enforced to prevent the model from over-relying on high-magnitude indices like Glicko ratings, forcing it to value tactical form (Accuracy, Control Time).

### 2. Bayesian Skill Estimation: Glicko-2 with Rating Regression
Octagon AI implements a custom Glicko-2 system that tracks **Rating (r)**, **Rating Deviation (RD)**, and **Volatility (σ)**.
- **The "Legacy Fix" (Rating Regression)**: Unlike standard ELO, v16 implements a biological decay function. After 6 months of inactivity, a fighter's rating is regressed toward the 1500 mean by **1% per month**, while their RD (uncertainty) expands by **15 points/period**.
- **Capped Differentials**: Glicko differences are capped at **±250 pts** to ensure that elite "Gatekeepers" who face top competition aren't unfairly penalized when matched against rising momentum-heavy prospects.

### 3. Feature Intelligence: The "Tactical-Differential" Set
The model analyzes **Point-in-Time Exponential Moving Averages (EMA, α=0.3)** to capture a fighter's "Current Form" rather than their lifetime career average.

#### 🎯 Tactical Features:
- **Positional Distribution**: Distance % vs. Clinch % vs. Ground %.
- **Target Variety**: Head % vs. Body % vs. Leg %.
- **Differential Pressure**: `slpm_diff` (Strike Landed per Min) and `sapm_diff` (Strike Absorbed).
- **Submission Threat & Control**: `sub_rate` (EMA) and `ctrl_pct` (Ground Control dominance).

#### 🧬 Biological & External Factors:
- **Athletic Age**: Years since UFC debut (a superior proxy for "Fight Years" compared to chronological age).
- **Environmental Context**: `is_altitude` (Locations > 4,000ft) and `is_apex` (Small 25ft cage vs Large 30ft cage).
- **Ring Rust**: Total days since last competition, normalized as a nonlinear decay indicator.

---

## 📈 Probability Calibration

Standard ML outputs are often poorly calibrated for betting markets. Octagon AI utilizes **Platt Scaling** via `CalibratedClassifierCV` (Sigmoid method) to transform raw model scores into true frequentist probabilities.

| Metric | v15 (Baseline) | v16 (Modern Era) | Improvement |
| :--- | :--- | :--- | :--- |
| **LogLoss** | 0.6477 | **0.6442** | +0.5% |
| **Brier Score** | 0.2281 | **0.2267** | +0.6% |
| **Market Correlation**| Moderate | **High** | Neutralized Legacy Bias |

---

## 💻 Tech Stack & Pipeline

### Backend Intelligence
- **Data Engine**: Pandas / NumPy / Joblib
- **Modeling**: CatBoost / Scikit-Learn (Calibration / Metrics)
- **Scraper**: BeautifulSoup4 / Requests (Asynchronous polling of ESPN / UFC Stats)

### Frontend Dashboard
- **Framework**: Next.js 14 (Tailwind CSS)
- **Visualization**: Recharts / Radar Charts (D3-powered tactical overlays)

### Weekly Automation Workflow
The engine is 100% autonomous through its `weekly_update.yml` workflow:
1. **Monday 00:00**: Scrapes previous event results and builds new Glicko-2 histories.
2. **Monday 00:05**: Retrains the v16 CatBoost model on the updated 16,000+ fight dataset.
3. **Monday 00:10**: Generates predictions for the next upcoming events.
4. **Monday 00:15**: Fetches market odds and deploys the new `upcoming_predictions.json` to the production site.

---

## 🚀 Research & Development

To setup the "Modern Era" engine locally:

1. **Environment**:
    ```bash
    pip install pandas numpy scikit-learn catboost joblib beautifulsoup4
    ```
2. **Regenerate Glicko History**:
    ```bash
    python3 src/generate_glicko.py
    ```
3. **Execute Prediction Pipeline**:
    ```bash
    python3 src/predict_model.py
    ```

---
**Octagon AI** — *Predictive excellence through tactical data science.*
