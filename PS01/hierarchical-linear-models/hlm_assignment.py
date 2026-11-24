import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Generate simple hierarchical data

np.random.seed(0)

data = []
for j in range(5):
    u = np.random.normal(0, 2)  # Random effect for each group
    for i in range(20):
        x = np.random.uniform(0, 5)
        y = 10 + 3*x + u + np.random.normal(0, 4)
        data.append([j, x, y])

df = pd.DataFrame(data, columns=["group", "x", "y"])

# Visualize hierarchical structure (Task 2)

plt.figure(figsize=(8,5))
for g in df["group"].unique():
    sub = df[df["group"] == g]
    plt.scatter(sub["x"], sub["y"], label=f"Group {g}", alpha=0.7)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Hierarchical Data")
plt.tight_layout()
plt.savefig("Q5_plot.png", dpi=300)  # Save the figure with high resolution
plt.show()

print("Figure saved as Q5_plot.png")

# Fit HLM model (Task 2)

model = smf.mixedlm("y ~ x", df, groups=df["group"])  # HLM with random intercept
result = model.fit()

print("\nHLM Results:")
print(result.summary())

# Compare predictions: HLM vs OLS (Task 3)

# Predictions from HLM
df["pred_hlm"] = result.predict(df)   # HLM predictions (with group effect)

# OLS Model (without group effect)
ols = smf.ols("y ~ x", df).fit()       # OLS Model
df["pred_ols"] = ols.predict(df)      # OLS predictions (without group effect)

# Plot both predictions to compare
plt.figure(figsize=(8,5))
for g in df["group"].unique():
    sub = df[df["group"] == g]
    plt.scatter(sub["x"], sub["y"], alpha=0.35, label=f"Group {g} (actual)")

plt.plot(df["x"], df["pred_hlm"], label="HLM Prediction (with group effect)", color='r', linewidth=2)
plt.plot(df["x"], df["pred_ols"], label="OLS Prediction (no group effect)", color='g', linestyle='--', linewidth=2)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Predictions: HLM vs OLS")
plt.legend()
plt.tight_layout()

# Save figure for comparison
plt.savefig("Q5_predictions_comparison.png", dpi=300)
plt.show()

print("Figure saved as Q5_predictions_comparison.png")