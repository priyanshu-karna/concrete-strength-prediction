# -*- coding: utf-8 -*-
"""Generate small, focused plots for the README (saved to assets/)."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

df = pd.read_excel("Concrete_Data.xls")
df.columns = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate",
    "age", "compressive_strength",
]
df = df.drop_duplicates().reset_index(drop=True)  # drop 25 dupes found in EDA

os.makedirs("assets", exist_ok=True)

# 1. Correlation heatmap
fig, ax = plt.subplots(figsize=(7, 5.5))
corr = df.corr().round(2)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, linewidths=0.5, annot_kws={"size": 7})
ax.set_title("Feature Correlation", fontsize=11)
ax.tick_params(axis="x", rotation=45, labelsize=7)
ax.tick_params(axis="y", rotation=0, labelsize=7)
plt.tight_layout()
plt.savefig("assets/correlation.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: assets/correlation.png")

# 2. Target distribution
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(df["compressive_strength"], bins=30, color="steelblue",
        edgecolor="k", linewidth=0.4, alpha=0.85)
ax.set_xlabel("Compressive Strength (MPa)", fontsize=9)
ax.set_ylabel("Count", fontsize=9)
ax.set_title("Target Distribution", fontsize=11)
plt.tight_layout()
plt.savefig("assets/target_distribution.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: assets/target_distribution.png")

# 3. Actual vs Predicted: baseline vs feature engineering
X_base = df.drop(columns="compressive_strength")
y = df["compressive_strength"]
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
sc = StandardScaler()
m = LinearRegression()
m.fit(sc.fit_transform(X_train), y_train)
y_pred_base = m.predict(sc.transform(X_test))
r2_base = r2_score(y_test, y_pred_base)

df["wc_ratio"] = df["water"] / df["cement"]  # lower ratio = stronger
df["log_age"]  = np.log1p(df["age"])         # log compresses skewed age
X_eng = df.drop(columns="compressive_strength")
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_eng, y, test_size=0.2, random_state=42)
sc2 = StandardScaler()
m2 = LinearRegression()
m2.fit(sc2.fit_transform(X_train2), y_train2)
y_pred_eng = m2.predict(sc2.transform(X_test2))
r2_eng = r2_score(y_test2, y_pred_eng)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, yt, yp, r2, title in [
    (axes[0], y_test,  y_pred_base, r2_base, "Baseline"),
    (axes[1], y_test2, y_pred_eng,  r2_eng,  "With Feature Engineering"),
]:
    ax.scatter(yt, yp, alpha=0.5, s=18, edgecolors="k", linewidths=0.2, color="steelblue")
    lims = [min(yt.min(), yp.min()) - 2, max(yt.max(), yp.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Actual (MPa)", fontsize=9)
    ax.set_ylabel("Predicted (MPa)", fontsize=9)
    ax.set_title(f"{title}  |  R2 = {r2:.3f}", fontsize=10)
plt.tight_layout()
plt.savefig("assets/actual_vs_predicted.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: assets/actual_vs_predicted.png")

# 4. Feature coefficients bar chart (feature eng model)
coef_df = pd.DataFrame({
    "feature": X_eng.columns,
    "coefficient": m2.coef_,
}).sort_values("coefficient", key=abs, ascending=True)

colors = ["tomato" if c < 0 else "steelblue" for c in coef_df["coefficient"]]
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(coef_df["feature"], coef_df["coefficient"], color=colors, edgecolor="k", linewidth=0.4)
ax.axvline(0, color="k", linewidth=0.8)
ax.set_xlabel("Coefficient (scaled)", fontsize=9)
ax.set_title("Feature Coefficients (Feature Eng. Model)", fontsize=10)
ax.tick_params(labelsize=8)
plt.tight_layout()
plt.savefig("assets/feature_coefficients.png", dpi=130, bbox_inches="tight")
plt.close()
print("Saved: assets/feature_coefficients.png")