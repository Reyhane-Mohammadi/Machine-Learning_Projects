import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("breast_cancer_wisconsin_diagnostic.csv") 


# Preprocessing
df = df.drop("ID", axis=1)

df["Diagnosis"] = df["Diagnosis"].map({"B": 0, "M": 1})

X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

# Grid Search for each kernel

# Linear
params_linear = {"C": [0.01, 0.1, 1, 10], "gamma": ["scale", "auto"]}
run_svm("linear", params_linear)

# RBF
params_rbf = {"C": [1, 10, 50], "gamma": ["scale", "auto"]}
run_svm("rbf", params_rbf)

# Polynomial (degree 3)
params_poly = {"C": [1, 10, 50], "gamma": ["scale", "auto"], "degree": [3]}
run_svm("poly", params_poly)


# Helper to run grid search for any kernel
def run_svm(kernel_name, params):
    print(f"\n---- Running SVM with kernel = {kernel_name} ----")
    
    svm = SVC(kernel=kernel_name)
    grid = GridSearchCV(svm, params, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Best parameters:", grid.best_params_)
    print("Accuracy on test data:", acc)
    print("Confusion matrix:\n", cm)

    plot_confusion(cm, f"SVM ({kernel_name}) - Confusion Matrix")

    return acc, cm


# Function to plot confusion matrix
def plot_confusion(cm, title):
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()
