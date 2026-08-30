"""
Linear Regression with feature engineering on concrete compressive strength dataset.
New features: w/c ratio (water/cement) and log(age).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_excel("Concrete_Data.xls")

df.columns = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate",
    "age", "compressive_strength",
]

missing = df.isnull().sum()  # check for missing values
print("Missing values:", missing[missing > 0].to_dict() if missing.any() else "None")

dupes = df.duplicated().sum()  # check for duplicate rows
print(f"Duplicate rows: {dupes}")
if dupes > 0:
    df = df.drop_duplicates().reset_index(drop=True)  # drop and reindex
    print(f"Shape after dropping duplicates: {df.shape}")

# feature engineering
df["wc_ratio"] = df["water"] / df["cement"]   # lower w/c ratio -> stronger concrete
df["log_age"]  = np.log1p(df["age"])          # log(1+age) compresses the skewed age distribution

print("Sample of engineered features:")
print(df[["water", "cement", "wc_ratio", "age", "log_age"]].head())

X = df.drop(columns="compressive_strength")
y = df["compressive_strength"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit on train only
X_test_scaled  = scaler.transform(X_test)       # apply same scale to test

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n-- Baseline (no feature engineering) --")
print("  MAE  : 7.745 MPa")
print("  RMSE : 9.797 MPa")
print("  R2   : 0.6275")

print("\n-- With Feature Engineering --")
print(f"  MAE  : {mae:.3f} MPa")
print(f"  MSE  : {mse:.3f}")
print(f"  RMSE : {rmse:.3f} MPa")
print(f"  R2   : {r2:.4f}")

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_,
}).sort_values("coefficient", key=abs, ascending=False)  # sort by importance

print("\n-- Feature Coefficients (sorted by |magnitude|) --")
print(coef_df.to_string(index=False))

os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Linear Regression with Feature Engineering - Concrete Compressive Strength", fontsize=13)

axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors="k", linewidths=0.3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", label="Perfect prediction")  # diagonal = perfect fit
axes[0].set_xlabel("Actual Strength (MPa)")
axes[0].set_ylabel("Predicted Strength (MPa)")
axes[0].set_title(f"Actual vs Predicted  (R2 = {r2:.3f})")
axes[0].legend()

residuals = y_test.values - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, edgecolors="k", linewidths=0.3)
axes[1].axhline(0, color="r", linestyle="--")  # zero-error reference line
axes[1].set_xlabel("Predicted Strength (MPa)")
axes[1].set_ylabel("Residual (Actual - Predicted)")
axes[1].set_title("Residual Plot")

axes[2].barh(coef_df["feature"], coef_df["coefficient"])
axes[2].axvline(0, color="k", linewidth=0.8)
axes[2].set_xlabel("Coefficient (scaled)")
axes[2].set_title("Feature Coefficients")

plt.tight_layout()
plt.savefig("plots/linear_regression_feat_eng_results.png", dpi=150)
print("\nPlot saved to plots/linear_regression_feat_eng_results.png")
