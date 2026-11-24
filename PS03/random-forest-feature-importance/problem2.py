import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("breast_cancer_wisconsin_diagnostic.csv")


df = df.drop("ID", axis=1)


df["Diagnosis"] = df["Diagnosis"].map({"B": 0, "M": 1})


X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)


def plot_feature_importances(model, feature_names, title, top_n=5):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]  

    plt.figure(figsize=(7, 4))
    sns.barplot(
        x=importances[indices],
        y=[feature_names[i] for i in indices],
        orient="h",
        palette="viridis"
    )
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)

y_train_pred_default = rf_default.predict(X_train)
y_test_pred_default = rf_default.predict(X_test)

acc_train_default = accuracy_score(y_train, y_train_pred_default)
acc_test_default  = accuracy_score(y_test, y_test_pred_default)

print("Random Forest (default params)")
print("Train accuracy:", acc_train_default)
print("Test  accuracy:", acc_test_default)

plot_feature_importances(
    rf_default,
    feature_names=X.columns,
    title="Random Forest (Default) - Top 5 Feature Importances",
    top_n=5
)


rf_tuned = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf_tuned.fit(X_train, y_train)

y_train_pred_tuned = rf_tuned.predict(X_train)
y_test_pred_tuned = rf_tuned.predict(X_test)

acc_train_tuned = accuracy_score(y_train, y_train_pred_tuned)
acc_test_tuned  = accuracy_score(y_test, y_test_pred_tuned)

print("\nRandom Forest (tuned params)")
print("Train accuracy:", acc_train_tuned)
print("Test  accuracy:", acc_test_tuned)

plot_feature_importances(
    rf_tuned,
    feature_names=X.columns,
    title="Random Forest (Tuned) - Top 5 Feature Importances",
    top_n=5
)
