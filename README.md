# 🥊 Octagon AI // UFC Prediction Engine v10

![Octagon AI Logo](dashboard/public/logo.png)

Octagon AI is a state-of-the-art predictive intelligence engine for MMA, specifically tuned for the UFC. Using a multi-stage machine learning pipeline, it analyzes over 30 years of historical fight data to generate win probabilities and "Method of Victory" (MOV) forecasts for upcoming bouts.

---

## 🧠 The Prediction Model (v10)

The core engine utilizes a **Random Forest Classifier** ensemble, optimized for non-linear relationships in fighter stylistic matchups.

### Model Architecture
- **Algorithm**: Balanced Random Forest Ensemble
- **Hyperparameters**: 300 Estimators, Max Depth 12, Min Samples Leaf 5.
- **Data Balancing**: Class-weight balancing at training time to handle the historical skew toward decision-based outcomes.
- **Symmetry Control**: Positional bias is eliminated by training on "flipped" data (Fighter 1 vs Fighter 2 and Fighter 2 vs Fighter 1).

### Feature Engineering (The "v10" Improvements)
Version 10 introduced sophisticated features that significantly improved accuracy over baseline models:
1.  **Defense & Efficiency**: Strike Differential (SLpM - SApM) and Absorption rates.
2.  **Activity & Layoff**: "Ring Rust" (days since last fight) and activity frequency in the last 12/24 months.
3.  **Cardio Proxies**: Finish rates and "Late Round Percentage" (propensity to go into the championship rounds).
4.  **Style Encoding**: Vectorized encoding of Distance vs. Clinch vs. Ground fighting styles.
5.  **Reach Modifiers**: Reach is no longer treated as a raw number but as a modifier (Reach × Distance Style), amplifying the advantage for long-range strikers.

---

## 📊 Historical Data & Training
The model is trained on a comprehensive dataset curated from nearly the entire history of the UFC.

- **Total Fights Analyzed**: 8,461
- **Date Range**: March 11, 1994 to Present (Updated Weekly)
- **Data Points**: Includes individual strike metrics, takedown success, control time, and physical measurements.

---

## � Performance Metrics

### Global Performance
- **Accuracy**: **60.6%** (validated via 5-fold cross-validation)
- **Consistency**: High reliability across mixed stylistic matchups (Striker vs. Grappler).

### Weight Class Specializations
Fight dynamics vary significantly by weight. Octagon AI utilizes specialized models for different divisions:

| Weight Class | Accuracy | Key Success Factor |
| :--- | :--- | :--- |
| **Heavyweight** | 60.6% | Reach × Distance Style |
| **Middleweight** | 61.1% | Distance Fighting Efficiency |
| **Welterweight** | 62.5% | Takedown & Control Rate |
| **Lightweight** | 61.8% | Takedown Efficiency |
| **Featherweight** | 58.3% | Strike Differential |
| **Bantamweight** | 56.5% | Strike Differential |
| **Flyweight** | 59.8% | Defense (Low SApM) |

---

## 💻 Web Application Architecture

The project is split into a data-science backend and a modern web dashboard.

### Tech Stack
- **Frontend**: Next.js 14, React, Tailwind CSS, Lucide Icons, Recharts (for MOV breakdown).
- **Backend Service**: Python 3.12 (Scikit-Learn, Pandas, BeautifulSoup4).
- **Automation**: GitHub Actions (CI/CD pipeline).

### Core Scripts
- `src/scrape_events.py`: Scrapes upcoming UFC event data and historical stats.
- `src/predict_events.py`: The powerhouse script that loads the `.pkl` models and generates predictions.
- `src/fetch_odds.py`: Real-time betting odds integration from external APIs.
- `src/train_model_v10.py`: The training pipeline for the global and weight-class models.

### Automation Workflow
The website is **fully autonomous**. Every **Monday at Midnight**, a GitHub Action triggers:
1.  Updates the historical database with last weekend's results.
2.  Scrapes the new upcoming fight card.
3.  Runs the AI models to generate fresh predictions.
4.  Fetches real-time market odds.
5.  Commits and pushes the new data directly to the web app.

---

## 🚀 Local Setup

1.  **Clone the Repo**
2.  **Install Python Dependencies**:
    ```bash
    pip install -r src/requirements.txt
    ```
3.  **Run the Dashboard**:
    ```bash
    cd dashboard
    npm install
    npm run dev
    ```
4.  **Run Manual Update**:
    ```bash
    cd src
    python predict_events.py
    ```

---
*Created with focus on predictive excellence.*
