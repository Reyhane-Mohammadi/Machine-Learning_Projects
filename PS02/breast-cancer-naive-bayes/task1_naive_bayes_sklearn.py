import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# data 
csv_path = "breast_cancer_wisconsin_diagnostic.csv"
df = pd.read_csv(csv_path)


if "ID" in df.columns:
    df = df.drop(columns=["ID"])

if "Diagnosis" not in df.columns:

    raise ValueError("ستون 'Diagnosis' در دیتافریم پیدا نشد. نام ستون‌ها را بررسی کنید.")

y_raw = df["Diagnosis"]
X = df.drop(columns=["Diagnosis"])

le = LabelEncoder()
y = le.fit_transform(y_raw)  

#  train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  
    stratify=y        
)

# model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)


plt.figure(figsize=(6, 5))
plt.matshow(cm, cmap='Blues', fignum=1) 


for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i][j]), ha='center', va='center', color='black')

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Naive Bayes Confusion Matrix\nAccuracy = {acc:.3f}")

if hasattr(le, 'classes_'):
    plt.xticks(ticks=[0, 1], labels=le.classes_)
    plt.yticks(ticks=[0, 1], labels=le.classes_)

plt.tight_layout() 
plt.show() 
