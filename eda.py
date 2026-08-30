"""
Exploratory Data Analysis for the concrete compressive strength dataset.
Checks: shape, missing values, duplicates, distributions, and correlations.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_excel("Concrete_Data.xls")

df.columns = [
    "cement", "blast_furnace_slag", "fly_ash", "water",
    "superplasticizer", "coarse_aggregate", "fine_aggregate",
    "age", "compressive_strength",
]

os.makedirs("plots", exist_ok=True)

# --- basic info ---
print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nDescriptive stats:\n", df.describe().round(2))

# --- missing values ---
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0] if missing.any() else "  None found")

# --- duplicates ---
dupes = df.duplicated().sum()
print(f"\nDuplicate rows: {dupes}")
if dupes > 0:
    print("Dropping duplicates...")
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Shape after dropping duplicates: {df.shape}")

# --- feature distributions ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(df.columns):
    axes[i].hist(df[col], bins=30, edgecolor="k", color="steelblue", alpha=0.8)
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("plots/eda_distributions.png", dpi=150, bbox_inches="tight")
print("\nPlot saved: plots/eda_distributions.png")

# --- correlation heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr().round(2)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/eda_correlation.png", dpi=150)
print("Plot saved: plots/eda_correlation.png")

# --- target vs each feature (scatter) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
features = [c for c in df.columns if c != "compressive_strength"]
for i, col in enumerate(features):
    axes[i].scatter(df[col], df["compressive_strength"], alpha=0.4, edgecolors="k", linewidths=0.2)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Compressive Strength (MPa)")
plt.suptitle("Feature vs Target", fontsize=14)
plt.tight_layout()
plt.savefig("plots/eda_feature_vs_target.png", dpi=150)
print("Plot saved: plots/eda_feature_vs_target.png")
