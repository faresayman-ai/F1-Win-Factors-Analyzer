# 🏎️ F1 Win Factors Analyzer

> A machine learning pipeline that uncovers **the dominant patterns behind Formula 1 race wins** using historical race data (2021–2024) — achieving a **ROC-AUC of 0.939** and powered by SHAP explainability.

---

## 📌 Project Overview

What actually determines who wins an F1 race? Is it the car, the driver, or where you start on the grid?

This project collects real race data from the [Ergast F1 API](https://api.jolpi.ca/ergast/), trains an XGBoost binary classifier to distinguish race winners from non-winners, and then uses **SHAP values to explain which factors matter most** — and by how much.

The goal is not just prediction, but **insight**: understanding the patterns and conditions that consistently produce race winners across 4 seasons of Formula 1.

The pipeline covers:
- Live data collection from the Ergast F1 API (2021–2024)
- Feature engineering & leak-free label encoding
- XGBoost classification with class imbalance handling
- Model evaluation with classification report & ROC-AUC
- Factor analysis using SHAP values

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 92% |
| ROC-AUC | **0.939** |
| Weighted F1-Score | 0.93 |
| Macro F1-Score | 0.70 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| 0 — Did NOT win | 0.98 | 0.94 | 0.96 | 342 |
| 1 — **Race Winner** | 0.34 | 0.61 | 0.44 | 18 |

> **Note on imbalance:** Only 1 driver wins per race (~5% of all entries), making this a naturally imbalanced classification problem. The high ROC-AUC (0.939) confirms the model learned real separating patterns — not just the majority class. `scale_pos_weight` was applied to handle this imbalance.

---

## 🔬 Features Used

| Feature | Description | Importance |
|---|---|---|
| `grid` | Starting grid position (pole = 1) | **~40%** — strongest signal |
| `con_enc` | Encoded constructor (team) ID | **~33%** |
| `dri_enc` | Encoded driver ID | **~15%** |
| `year` | Race season year | **~10%** |

---

## 🧠 Key Findings (SHAP Analysis)

SHAP (SHapley Additive exPlanations) reveals **why** the model makes each decision — turning a black-box model into an analytical tool.

### What the SHAP summary plot tells us:

- **Grid position is the single most dominant factor** — drivers starting at the front (low grid value) are far more likely to win. This quantifies what F1 fans already suspect: qualifying is everything.
- **Constructor (team) is almost as important** — certain teams consistently produce winners regardless of driver. This reflects the engineering and resource gap between top and midfield teams.
- **Driver identity matters, but less than the car** — a strong driver helps, but can't fully overcome a midfield car.
- **Year has a small but real effect** — capturing the shifting dominance across eras (e.g., Red Bull's near-total dominance in 2023 vs more competitive 2021).

> **Bottom line:** In modern F1, *where you start* and *which team you drive for* explain the vast majority of race wins.

---

## 🛠️ Tech Stack

- **Python**
- **Requests** — live data collection from Ergast F1 API
- **Pandas / NumPy** — data wrangling
- **Scikit-learn** — preprocessing, train/test split, evaluation
- **XGBoost** — gradient boosted classifier
- **SHAP** — factor analysis & explainability
- **Matplotlib** — visualizations

---

## 📁 Project Structure

```
f1-win-factors-analyzer/
│
├── f1_analyzer.ipynb          # Main notebook (data → model → SHAP)
├── feature_importance.png     # Feature importance bar chart
├── shap_summary.png           # SHAP beeswarm plot
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/your-username/f1-win-factors-analyzer.git
cd f1-win-factors-analyzer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the notebook
jupyter notebook f1_analyzer.ipynb
```

**requirements.txt**
```
requests
pandas
scikit-learn
xgboost
shap
matplotlib
jupyter
```

---

## ⚙️ Key Design Decisions

### ✅ No Data Leakage
The train/test split is performed **before** label encoding. The `LabelEncoder` is fit only on training data and applied to the test set — this mirrors real-world deployment where unseen data must be encoded using past knowledge only.

### ✅ Imbalanced Class Handling
Since only 1 out of ~20 drivers wins each race, `scale_pos_weight = neg/pos` was passed to XGBoost to prevent the model from ignoring the minority class entirely.

### ✅ Explainability Over Black-Box Accuracy
The SHAP analysis is the core output of this project. Raw accuracy numbers are easy to game on imbalanced data — SHAP values show whether the model learned *real* patterns, not just shortcuts.

---

## 📡 Data Source

Race results collected from the **[Jolpica Ergast F1 API](https://api.jolpi.ca/ergast/)** — a free, open-source Formula 1 historical data API covering seasons from 1950 to present.

