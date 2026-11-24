import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import xgboost as xgb

# Load dataset
df = pd.read_csv("breast_cancer_wisconsin_diagnostic.csv")

df = df.drop(columns=["ID"])
df["Diagnosis"] = df["Diagnosis"].map({"M": 1, "B": 0})

X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

#  Scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.30, random_state=42, stratify=y
)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# XGBoost basic model (old-version compatible)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,
    "eta": 0.1
}

num_round = 150  

model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_round
)


# Predict & AUC

y_pred_proba = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred_proba)

print("Final Test AUC:", auc)

# Simple Learning Curve (SAFE for old versions)
rounds_list = [10, 30, 60, 100, 150]
train_losses = []
test_losses = []

from sklearn.metrics import log_loss

for r in rounds_list:
    mdl = xgb.train(params=params, dtrain=dtrain, num_boost_round=r)
    pred_train = mdl.predict(dtrain)
    pred_test = mdl.predict(dtest)
    
    train_losses.append(log_loss(y_train, pred_train))
    test_losses.append(log_loss(y_test, pred_test))

plt.figure(figsize=(8,5))
plt.plot(rounds_list, train_losses, label="Train Logloss")
plt.plot(rounds_list, test_losses, label="Test Logloss")
plt.xlabel("Boosting Rounds")
plt.ylabel("Logloss")
plt.title("Simple XGBoost Learning Curve (Compatible Version)")
plt.legend()
plt.grid(True)
plt.show()
