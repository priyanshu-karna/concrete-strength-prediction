# Concrete Compressive Strength Prediction

A machine learning project where I try to predict concrete compressive strength (MPa) from its mix ingredients using linear regression.

This is part of my personal ML learning journey — working through real datasets, building models from scratch, and figuring out what actually matters in the pipeline.

---

## Dataset

**Source**: UCI Machine Learning Repository — donated by Prof. I-Cheng Yeh (Chung-Hua University, Taiwan).

1030 samples. No missing values. 8 input features, 1 target.

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

---

## What I built

### `linear_regression.py`
Baseline linear regression model. Straightforward pipeline:
- 80/20 train-test split
- StandardScaler for feature scaling
- Evaluated on MAE, RMSE, and R²
- Three plots saved: Actual vs Predicted, Residual plot, Feature Coefficients

**Test set results:**

| Metric | Value |
|---|---|
| MAE | 7.745 MPa |
| RMSE | 9.797 MPa |
| R² | 0.6275 |

Cement and blast furnace slag came out as the strongest predictors. Water had the only negative coefficient — makes sense, too much water weakens the mix.

R² of 0.63 isn't great, but it's expected. The dataset readme literally says strength is a *highly nonlinear* function of age and ingredients. Linear regression is a reasonable starting point, not the final answer.

---

## Setup

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl
python linear_regression.py
```

Plots are saved to the `plots/` folder.

---

## What's next

- Feature engineering: w/c ratio, log(age)
- Try Random Forest and Gradient Boosting
- Cross-validation instead of a single train-test split
- Compare all models side by side
