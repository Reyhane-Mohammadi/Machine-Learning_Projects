# Import Libraries
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,4)

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

# Load Dataset
df = pd.read_csv("pima_indians_diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Kernel Comparison
kernels = ['linear', 'rbf', 'poly']
results = []

for k in kernels:
    model = SVC(kernel=k, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append([
        k,
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ])

results_df = pd.DataFrame(results, columns=['kernel','accuracy','precision','recall','f1'])

plt.figure()
for metric in ['accuracy','precision','recall','f1']:
    plt.plot(results_df['kernel'], results_df[metric])
plt.title("Kernel Comparison")
plt.xlabel("Kernel")
plt.ylabel("Score")
plt.legend(['Accuracy','Precision','Recall','F1'])
plt.show()

# Effect of C
C_values = [0.01, 0.1, 1, 10, 100]
acc_scores = []

for c in C_values:
    model = SVC(kernel='rbf', C=c)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(C_values, acc_scores)
plt.xscale('log')
plt.title("Effect of C")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.show()

# Effect of Gamma
gamma_values = [0.001, 0.01, 0.1, 1]
acc_scores = []

for g in gamma_values:
    model = SVC(kernel='rbf', gamma=g)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_scores.append(accuracy_score(y_test, y_pred))

plt.figure()
plt.plot(gamma_values, acc_scores)
plt.xscale('log')
plt.title("Effect of Gamma")
plt.xlabel("Gamma")
plt.ylabel("Accuracy")
plt.show()

# Grid Search (IMPORTANT FIX: faster + stable)
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale']
}

grid = GridSearchCV(
    SVC(probability=True),
    param_grid,
    cv=3,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="AUC = %.2f" % roc_auc)
plt.plot([0,1], [0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()