# Octagon AI: Technical Model Specifications (Win/Loss Focus)

This document provides a detailed breakdown of the Octagon AI (v10) prediction engine, its architecture, feature engineering, and the results of a rigorous "Blind" backtest audit.

---

## 1. Core Architecture
The system uses a **Random Forest Classifier** (Scikit-learn) ensemble approach focused strictly on binary Win/Loss prediction.

- **Global Win/Loss Model**: An ensemble of 300 decision trees predicting the probability of Fighter 1 winning.
- **Division-Specific Specialization**: The engine includes 8 specialized models for individual weight classes (Heavyweight to Flyweight) to account for differing volume and pace profiles across divisions.
- **Data Period**: Trained on professional UFC bouts from 2010 to present.

---

## 2. Feature Engineering (19 Dimensions)
Inputs are derived from "Point-in-Time" historical statistics. Every feature is a **Differential** (Fighter 1 - Fighter 2).

### Career & Momentum
1.  **Win Rate Diff**: Career winning percentage.
2.  **Experience Diff**: Total number of professional UFC fights.
3.  **Streak Diff**: Current win/loss streak.

### Offensive Metrics
4.  **SLpM Diff**: Significant Strikes Landed per Minute.
5.  **KD Rate Diff**: Knockdowns per 15 minutes.
6.  **TD Rate Diff**: Takedowns landed per 15 minutes.
7.  **Sub Rate Diff**: Submission attempts per 15 minutes.
8.  **Ctrl Rate Diff**: Control time seconds per 15 minutes.

### Defensive & Efficiency
9.  **SApM Diff**: Significant Strikes Absorbed per Minute (Lower is better).
10. **Strike Differential**: (SLpM - SApM) comparison.

### Activity & Cardio
11. **Layoff Diff**: Days since the last professional bout.
12. **Activity 12m**: Number of fights in the trailing 12 months.
13. **Finish Rate**: Percentage of previous wins resulting in a finish (KO/Sub).
14. **Late Round %**: Percentage of total career rounds occurring in Round 3+.

### Style & Physicality
15. **Distance Style**: Differential in strikes landed from a distance.
16. **Ground Style**: Differential in ground-control dominance.
17. **Height Diff**: raw height difference in centimeters.
18. **Reach × Distance**: Interaction between reach advantage and distance strike volume.
19. **Reach × Volume**: Interaction between reach advantage and total striking activity.

---

## 3. Backtest Performance (Strict Blind Audit)
To verify real-world utility, a **Strict Blind Backtest** was conducted for the period **Jan 2023 – Dec 2024**.

### Audit Methodology:
- **Zero Leakage**: Model was retrained (`ufc_v10_blind_2023.pkl`) using *only* data prior to Jan 1st, 2023.
- **Walk-Forward**: Features were calculated point-in-time, ensuring no future knowledge of fight outcomes or post-fight stat corrections were accessible.
- **Betting Engine**: 0.25 Fractional Kelly Criterion with a 5% bankroll cap.

### Performance Results:
- **Starting Bankroll**: $1,000.00
- **Final Bankroll**: **$304.41**
- **Total ROI**: **-69.6%**
- **Win Rate**: 43.3%
- **Average Model Edge**: 18.1% (Heavy Overconfidence)
- **Max Drawdown**: 81.2%

---

## 4. Identified Technical Limitations

### 1. Probability Calibration (The "Edge" Problem)
The model consistently estimates an 18%+ edge over the market, but its 43% win rate suggests the probabilities are significantly over-optimistic. The model's raw output needs **Platt Scaling** or **Isotonic Regression** to better represent true win frequencies.

### 2. High-Variance Feature Sensitivity
- **Control Time Dominance**: `ctrl_rate_diff` currently has the highest feature importance (11.6%). This leads to anomalous probability spikes for "control-heavy" wrestlers who may not actually have a high win probability in new matchups.
- **Underdog Bias**: The model frequently identifies deep underdogs ($+400$) as "valuable" based on slight statistical leads in activity, failing to account for "Skill Floor" or "Weight of Competition" differences.

### 3. Missing Qualitative Variables
The model lacks features for:
- **Chin Durability (KO Resistance)**
- **Level of Competition (ELO/Glicko of past opponents)**
- **Fight Weight Variance (Short notice fights, missing weight)**

---

## 5. Professional Review Roadmap
Suggested focus areas for model improvement:
1.  **Calibration**: Force the model probabilities to align with historical win rates.
2.  **Feature Pruning**: Damping the influence of high-variance features like control time.
3.  **Rating Systems**: Incorporating opponent-strength adjusted ratings (ELO/Glicko-2) to replace raw win/loss streaks.
