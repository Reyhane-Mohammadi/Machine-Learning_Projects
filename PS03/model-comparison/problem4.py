import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


df = pd.read_csv("breast_cancer_wisconsin_diagnostic.csv")

df = df.drop(columns=["ID"])
df["Diagnosis"] = df["Diagnosis"].map({"B": 0, "M": 1})

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)


svm_model = SVC(kernel="rbf", probability=True)
rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_split=5)
xgb_model = XGBClassifier(
    learning_rate=0.1,
    max_depth=4,
    n_estimators=200,
    eval_metric="logloss"
)


svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)


models = {
    "SVM_RBF": svm_model,
    "RandomForest": rf_model,
    "XGBoost": xgb_model,
}

results = []

for name, m in models.items():
    y_pred = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba)
    })


results_df = pd.DataFrame(results)
print(results_df)


plt.figure(figsize=(7,5))
for name, m in models.items():
    y_proba = m.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.grid(True)
plt.show()
