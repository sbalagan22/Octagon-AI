# 🌐 Octagon AI Dashboard

This is the frontend dashboard for **Octagon AI**, a predictive intelligence engine for the UFC.

## 🚀 Features

- **Real-time Predictions**: Interactive cards showing AI-calculated win probabilities.
- **Dynamic Stats**: Visualized fighter profiles with career records and physical metrics.
- **Radar Analysis**: Statistical spider charts comparing fighter skills (Striking, Grappling, Cardio).
- **Mobile Optimized**: Custom horizontal schedule and full-screen modal experience for touch devices.

## 🛠️ Tech Stack

- **Framework**: [Next.js 14](https://nextjs.org/) (App Router)
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Charts**: Recharts

## 🏃 Driving the App

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Run Dev Server**:
   ```bash
   npm run dev
   ```

3. **Data Source**:
   The app reads from `public/upcoming_events.json`. This file is automatically updated by the `predict_events.py` script in the root directory.

---

### 📂 Architecture

- `/app`: Main page and layout.
- `/components`: Reusable UI elements (FightCard, FighterModal).
- `/public`: Static assets, logo (`mmaverse.png`), and prediction data.
- `/types`: TypeScript interfaces for the model output.

*For full model documentation, see the root [README.md](../README.md).*
