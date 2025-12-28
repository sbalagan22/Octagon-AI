# 🥊 Octagon AI: Predictive Intelligence for the UFC

Octagon AI is a high-performance predictive modeling engine designed to analyze and forecast UFC fight outcomes with surgical precision. By leveraging a **Random Forest Classifier** trained on decades of historical fight data, the system identifies subtle patterns in fighter dynamics that traditional odds often overlook.

---

## 🧠 The Engine: Model v10 Architecture

The current iteration (v10) represents a significant leap in predictive accuracy by moving beyond raw stats into **contextual combat metrics**.

### 1. Balanced Training (Bias Elimination)
Historical data often shows a "Fighter 1" bias (favorites listed first). Octagon AI eliminates this by **positional flipping**: every fight in the training set is duplicated with F1 and F2 swapped. This forces the model to learn purely from mathematical differentials rather than positional order.

### 2. Weight-Class Specific Intelligence
Combat dynamics change drastically across weight classes. A "knockout" in the Heavyweight division has different statistical weight than one in the Flyweight division.
- **Global Model**: Captures universal fight truths across all 7,000+ fights.
- **Specialized Models**: Separate Random Forest models optimized for each weight class, allowing the engine to prioritize "Takedown Rate" in lower weights and "Reach Advantage" in higher weights.

---

## 🧪 Intricate Feature Engineering

The secret to Octagon AI's accuracy lies in how it processes raw data into **Pre-Fight Differentials**.

### 🥊 The "Reach Modifier" Strategy
Raw reach is a lie. Octagon AI uses **interaction terms** to calculate "Effective Reach":
- **Reach × Distance Style**: Reach is amplified if the fighter is a "Distance Striker."
- **Reach × Volume**: A reach advantage is significantly more dangerous when paired with high strike volume (SLpM).

### 🛡️ Defensive Superiority
The model prioritizes **SApM (Strikes Absorbed per Minute)** and **Strike Differential**. It doesn't just care who hits more; it calculates the "Clean Multiplier"—how much damage a fighter can inflict while remaining unhittable.

### 🔋 Cardio & Durability Proxies
Since "gas tanks" aren't a raw stat, the model derives them from:
- **Late Round %**: The percentage of career fights that reached rounds 3, 4, or 5.
- **Finishing Ability**: The ratio of wins by KO/Sub vs. total wins.
- **Average Fight Time**: Used to normalize rates and detect "decision machines."

### 🏛️ Career & Momentum Dynamics
- **Ring Rust (Layoff)**: Calculated as days since the last fight to penalize inactivity.
- **Recent Activity**: Number of fights in the last 12/24 months.
- **The "Experience Gap"**: Calculating the difference in total professional cage time.

---

## 💻 The Web Application

A premium, mobile-optimized dashboard built with **Next.js** and **Tailwind CSS**.

- **Interactive Radar Charts**: Real-time statistical comparison of fighters across Striking, Takedowns, Control, KO Power, and Submissions.
- **Responsive Schedule**: A proprietary "Smart Swipe" horizontal schedule for mobile users and a deep-dive vertical sidebar for desktop users.
- **Live Factor Breakdown**: Each prediction displays the exact SLpM, SapM, and Reach differentials that influenced the AI's decision.
- **Deep Data Deduplication**: The frontend engine cross-references `Fighters.csv` (career records) with `Fights.csv` (UFC stats) to ensure legends like Justin Gaethje show their full 26-5 career record rather than just their recent UFC tenure.

---

## 🛠️ Performance & Tech Stack

- **Model**: Scikit-learn RandomForestClassifier (v10.4)
- **Frontend**: Next.js 14, Lucide Icons, Recharts (Radar Data-Viz)
- **Data Science**: Python, Pandas, NumPy, Joblib
- **Storage**: JSON-driven prediction architecture for ultra-fast load times.

---

### 🚀 Getting Started

1. **Train the Model**:
   ```bash
   cd src && python3 train_model_v10.py
   ```
2. **Generate Predictions**:
   ```bash
   python3 predict_events.py
   ```
3. **Launch the Dashboard**:
   ```bash
   cd web-app && npm run dev
   ```

*Octagon AI is an analytical tool and does not guarantee fight outcomes. Fight at your own risk.*
