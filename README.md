# Concrete Compressive Strength Prediction

A machine learning project predicting concrete compressive strength (MPa) from mix ingredients. Models explored: Linear Regression, Linear Regression with Feature Engineering, and XGBoost.

Part of my personal ML learning journey — working through real datasets, building proper pipelines, and understanding what actually moves the needle.

---

## Dataset

**Source**: UCI Machine Learning Repository — donated by Prof. I-Cheng Yeh (Chung-Hua University, Taiwan).

1030 samples, 8 input features, 1 target. The readme claimed no missing values — EDA found **25 duplicate rows**, which get dropped before training (clean set: 1005 rows).

| Feature | Unit |
|---|---|
| Cement | kg/m³ |
| Blast Furnace Slag | kg/m³ |
| Fly Ash | kg/m³ |
| Water | kg/m³ |
| Superplasticizer | kg/m³ |
| Coarse Aggregate | kg/m³ |
| Fine Aggregate | kg/m³ |
| Age | days |
| **Compressive Strength** *(target)* | MPa |

### Target Distribution

![Target Distribution](assets/target_distribution.png)

### Feature Correlation

![Correlation Heatmap](assets/correlation.png)

---

## Scripts

### `eda.py`
Runs before anything else. Checks missing values, duplicate rows, plots feature distributions, correlation heatmap, and feature-vs-target scatter plots. Saves everything to `plots/`.

### `linear_regression.py`
Baseline model — no feature engineering. Just the raw 8 features scaled and fed into `LinearRegression`.

**Pipeline**: EDA checks → 80/20 split → StandardScaler → LinearRegression → evaluate

| Metric | Value |
|---|---|
| MAE | 8.896 MPa |
| RMSE | 11.192 MPa |
| R² | 0.5801 |

### `linear_regression_feat_eng.py`
Same pipeline but with two extra features derived from the raw data:

- **`wc_ratio`** = water / cement — a known physical indicator of strength in concrete science. Lower ratio means stronger mix.
- **`log_age`** = log(1 + age) — age has diminishing returns on strength (gains fast early, plateaus later). The log transform captures that curve inside a linear model.

| Metric | Baseline | Feature Eng. | Change |
|---|---|---|---|
| MAE | 8.896 MPa | **5.749 MPa** | -2.15 |
| RMSE | 11.192 MPa | **7.442 MPa** | -3.75 |
| R² | 0.5801 | **0.8143** | +0.234 |

### Actual vs Predicted

![Actual vs Predicted](assets/actual_vs_predicted.png)

### Feature Coefficients

![Feature Coefficients](assets/feature_coefficients.png)

`log_age` and `cement` came out as the top two predictors. `wc_ratio` has a negative coefficient — higher water-to-cement ratio means weaker concrete, which matches the physics exactly.

### `xgboost_model.py`
XGBoost regressor with feature engineering and hyperparameter tuning via `GridSearchCV` (5-fold CV, 108 candidate combinations).

**Pipeline**: deduplication → `wc_ratio` + `log_age` features → 80/20 split → XGBRegressor → GridSearchCV → evaluate

**Best hyperparameters**

| Parameter | Value |
|---|---|
| `n_estimators` | 400 |
| `max_depth` | 5 |
| `learning_rate` | 0.1 |
| `subsample` | 0.8 |
| `colsample_bytree` | 1.0 |

**Evaluation**

| Metric | Value |
|---|---|
| MAE | **2.539 MPa** |
| RMSE | **4.124 MPa** |
| R² | **0.9430** |
| CV RMSE (5-fold) | 4.321 MPa |

**Feature importances (gain)**

| Feature | Importance |
|---|---|
| `wc_ratio` | 47.8% |
| `age` | 16.9% |
| `blast_furnace_slag` | 11.2% |
| `superplasticizer` | 6.9% |
| `fine_aggregate` | 5.3% |
| others | ~8% |

`wc_ratio` is the single most important predictor — consistent with concrete physics (lower water-to-cement ratio = stronger mix).

---

## Overall Model Comparison

| Model | MAE (MPa) | RMSE (MPa) | R² |
|---|---|---|---|
| Linear Regression | 8.896 | 11.192 | 0.5801 |
| LR + Feature Engineering | 5.749 | 7.442 | 0.8143 |
| **XGBoost (tuned)** | **2.539** | **4.124** | **0.9430** |

XGBoost achieves a **63% reduction in RMSE** over the plain linear regression baseline.

---


## Setup

```bash
pip install numpy pandas scikit-learn matplotlib seaborn openpyxl xgboost
```

Run in order:
```bash
python eda.py
python linear_regression.py
python linear_regression_feat_eng.py
python xgboost_model.py
```

Plots from EDA and model runs save to `plots/`. README assets are in `assets/`.

---

## What is next

- [x] Linear Regression baseline
- [x] Feature engineering (`wc_ratio`, `log_age`)
- [x] XGBoost with hyperparameter tuning
- [x] Cross-validation (5-fold GridSearchCV)
- [x] Side-by-side model comparison table
- [ ] Random Forest for comparison
- [ ] SHAP values for deeper XGBoost explainability
- [ ] Polynomial features on linear model