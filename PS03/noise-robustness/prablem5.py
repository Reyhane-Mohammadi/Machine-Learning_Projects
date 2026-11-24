import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt


df = pd.read_csv("breast_cancer_wisconsin_diagnostic.csv")

df = df.drop(columns=["ID"])
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})


X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)




rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)


xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_train, y_train)


baseline_rf = accuracy_score(y_test, rf.predict(X_test))
baseline_xgb = accuracy_score(y_test, xgb.predict(X_test))

print("Baseline accuracies:")
print(f"  RandomForest: {baseline_rf:.4f}")
print(f"  XGBoost     : {baseline_xgb:.4f}")


noise_ratio = 0.2
num_features = X_test.shape[1]
num_noisy = int(noise_ratio * num_features)

noise_features = np.random.choice(num_features, num_noisy, replace=False)

X_noisy = X_test.copy()
for f in noise_features:
    noise = np.random.normal(0, 0.15, size=X_noisy.shape[0])
    X_noisy[:, f] += noise

acc_rf_noise = accuracy_score(y_test, rf.predict(X_noisy))
acc_xgb_noise = accuracy_score(y_test, xgb.predict(X_noisy))

print("\nAccuracies with Gaussian noise on 20% features:")
print(f"  RandomForest: {acc_rf_noise:.4f}")
print(f"  XGBoost     : {acc_xgb_noise:.4f}")


feature_id = np.random.randint(0, num_features)

X_corrupt = X_test.copy()
np.random.shuffle(X_corrupt[:, feature_id])

acc_rf_corrupt = accuracy_score(y_test, rf.predict(X_corrupt))
acc_xgb_corrupt = accuracy_score(y_test, xgb.predict(X_corrupt))

print(f"\nCorrupted feature index: {feature_id}")
print("Accuracies with ONE feature shuffled:")
print(f"  RandomForest: {acc_rf_corrupt:.4f}")
print(f"  XGBoost     : {acc_xgb_corrupt:.4f}")


rf_probs = rf.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)

auc_rf = auc(fpr_rf, tpr_rf)
auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC = {auc_rf:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {auc_xgb:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Question 5")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('tight')
ax.axis('off')

table_data = [
    ["RandomForest (Q2)", f"{baseline_rf:.3f}", f"{acc_rf_noise:.3f}", f"{acc_rf_corrupt:.3f}"],
    ["XGBoost (Q4)", f"{baseline_xgb:.3f}", f"{acc_xgb_noise:.3f}", f"{acc_xgb_corrupt:.3f}"]
]

columns = ["Model", "Baseline Accuracy", "Gaussian Noise", "Feature Corruption"]

table = ax.table(
    cellText=table_data,
    colLabels=columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.6, 1.4)

plt.title("Robustness Comparison Table (Question 5)", pad=20)
plt.tight_layout()
plt.show()
