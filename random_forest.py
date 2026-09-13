"""
Random Forest model to predict concrete compressive strength (MPa).
Dataset: 1030 samples (1005 after deduplication), 8 input features + 2 engineered.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 1. Load & clean ──────────────────────────────────────────────────────────
df = pd.read_excel("Concrete_Data.xls")

df.columns = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate",
    "age", "compressive_strength",
]

dupes = df.duplicated().sum()
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {dupes} duplicate rows. Clean shape: {df.shape}")

# ── 2. Feature engineering ───────────────────────────────────────────────────
df["wc_ratio"] = df["water"] / df["cement"]   # lower w/c ratio -> stronger concrete
df["log_age"]  = np.log1p(df["age"])          # compresses skewed age distribution

# ── 3. Split ─────────────────────────────────────────────────────────────────
X = df.drop(columns="compressive_strength")
y = df["compressive_strength"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Tune ──────────────────────────────────────────────────────────────────
# ponytail: small grid to keep fit time reasonable; expand n_estimators/max_features if RMSE matters more than speed
param_grid = {
    "n_estimators": [200, 400],
    "max_depth":    [None, 10, 20],
    "max_features": ["sqrt", 0.5],
    "min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid,
    scoring="neg_root_mean_squared_error",
    cv=5,
    n_jobs=-1,
    verbose=1,
)
print("Running GridSearchCV...")
grid.fit(X_train, y_train)
print(f"Best params : {grid.best_params_}")
print(f"Best CV RMSE: {-grid.best_score_:.3f} MPa")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
model  = grid.best_estimator_
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)
cv_rmse = -cross_val_score(model, X_train, y_train,
                            scoring="neg_root_mean_squared_error", cv=5).mean()

print("\n-- Model Evaluation (test set) --")
print(f"  MAE     : {mae:.3f} MPa")
print(f"  RMSE    : {rmse:.3f} MPa")
print(f"  R2      : {r2:.4f}")
print(f"  CV RMSE : {cv_rmse:.3f} MPa  (5-fold)")

print("\n-- Comparison --")
print(f"  {'Model':<35} {'RMSE':>8}  {'R2':>7}")
print(f"  {'-'*52}")
print(f"  {'Linear Regression':<35} {'11.192':>8}  {'0.5801':>7}")
print(f"  {'LR + Feature Engineering':<35} {'7.442':>8}  {'0.8143':>7}")
print(f"  {'XGBoost (tuned)':<35} {'4.124':>8}  {'0.9430':>7}")
print(f"  {'Random Forest (tuned)':<35} {rmse:>8.3f}  {r2:>7.4f}")

# ── 6. Feature importance ─────────────────────────────────────────────────────
imp_df = pd.DataFrame({
    "feature":    X.columns,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\n-- Feature Importances --")
print(imp_df.to_string(index=False))

# ── 7. Plots ──────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Random Forest – Concrete Compressive Strength Prediction", fontsize=14)

# Actual vs Predicted
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3, color="steelblue")
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_xlabel("Actual Strength (MPa)")
ax.set_ylabel("Predicted Strength (MPa)")
ax.set_title(f"Actual vs Predicted  (R² = {r2:.3f})")
ax.legend()

# Residuals
ax = axes[0, 1]
residuals = y_test.values - y_pred
ax.scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3, color="darkorange")
ax.axhline(0, color="r", linestyle="--")
ax.set_xlabel("Predicted Strength (MPa)")
ax.set_ylabel("Residual (Actual − Predicted)")
ax.set_title("Residual Plot")

# Feature importance
ax = axes[1, 0]
ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1], color="mediumseagreen")
ax.set_xlabel("Feature Importance (mean decrease in impurity)")
ax.set_title("Random Forest Feature Importances")

# Model comparison
ax = axes[1, 1]
models    = ["Linear\nRegression", "LR +\nFeat Eng", "XGBoost\n(tuned)", "Random Forest\n(tuned)"]
rmse_vals = [11.192, 7.442, 4.124, rmse]
colors    = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
bars = ax.bar(models, rmse_vals, color=colors, width=0.5)
ax.set_ylabel("RMSE (MPa)  ← lower is better")
ax.set_title("Model RMSE Comparison")
for bar, val in zip(bars, rmse_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig("plots/random_forest_results.png", dpi=150)
print("\nPlot saved to plots/random_forest_results.png")
