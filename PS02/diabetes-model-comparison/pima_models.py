import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("pima_indians_diabetes.csv")


# Fix zero values as missing (as required in project)
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zero:
    data[col].replace(0, np.nan, inplace=True)
    data[col].fillna(data[col].median(), inplace=True)

# 2. Split data
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 3. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(probability=True),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        verbosity=0
    )
}

# 4. Metric functions (manual)
def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    return tp / (tp + fp + 1e-12)

def recall(tp, fn):
    return tp / (tp + fn + 1e-12)

def f1(p, r):
    return 2*p*r / (p + r + 1e-12)

def mcc(tp, tn, fp, fn):
    num = (tp * tn) - (fp * fn)
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) + 1e-12
    return num / den

# 5. Train models and collect results
results = {}

for name, model in models.items():
    print(f"Training {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    acc = accuracy(tp, tn, fp, fn)
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    f1_score = f1(prec, rec)
    mcc_score = mcc(tp, tn, fp, fn)

    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1_score,
        "MCC": mcc_score
    }

    # Plot confusion matrix
    plt.figure(figsize=(4,4))
    sns.heatmap([[tn, fp], [fn, tp]],
                annot=True, fmt="d",
                cmap="Blues", cbar=False,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"])
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

# 6. Convert results to DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df[["Accuracy", "Precision", "Recall", "F1", "MCC"]]

print("\nFinal Results Table:\n")
print(results_df)

# 7. Bar plot comparison
plt.figure(figsize=(14, 7))
sns.set_theme(style="whitegrid")
results_df.plot(kind="bar", figsize=(14,7))
plt.title("Model Performance Comparison (Pima Diabetes Dataset)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
