import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt 

# Load dataset
path = 'breast_cancer_wisconsin_diagnostic.csv'
try:
    df = pd.read_csv(path)
except FileNotFoundError:
    print(f"Error: The file '{path}' was not found. Please ensure the file is in the correct directory.")
    exit()

# Drop ID column (note: it's 'ID' not 'id')
if 'ID' in df.columns:
    df = df.drop(columns=['ID'])

# Encode Diagnosis: M -> 1, B -> 0
le = LabelEncoder()
if 'Diagnosis' not in df.columns:
    print("Error: 'Diagnosis' column not found in the CSV file.")
    exit()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

# Features and target
X = df.drop(columns=['Diagnosis'])
y = df['Diagnosis']

# Check missing values
missing_per_column = df.isna().sum()
print("Missing values per column (first 5 columns):\n", missing_per_column.head())

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Naive Bayes with scikit-learn
nb_sklearn = GaussianNB()
nb_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = nb_sklearn.predict(X_test_scaled)

metrics_sklearn = {
    'accuracy': accuracy_score(y_test, y_pred_sklearn),
    'precision': precision_score(y_test, y_pred_sklearn),
    'recall': recall_score(y_test, y_pred_sklearn),
    'f1': f1_score(y_test, y_pred_sklearn),
}

print("\n--- scikit-learn Gaussian Naive Bayes Metrics ---")
print(f"Accuracy: {metrics_sklearn['accuracy']:.4f}")
print(f"Precision: {metrics_sklearn['precision']:.4f}")
print(f"Recall: {metrics_sklearn['recall']:.4f}")
print(f"F1-Score: {metrics_sklearn['f1']:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_sklearn, target_names=[str(k) for k in le.classes_]))


# Naive Bayes from scratch (Gaussian)
class ScratchGaussianNB:
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.classes_ = np.unique(y)
        n_features = X.shape[1]

        self.priors_ = {}
        self.mean_ = {}
        self.var_ = {}

        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = X_c.shape[0] / X.shape[0]
            self.mean_[c] = X_c.mean(axis=0)
            self.var_[c] = X_c.var(axis=0) 
            self.var_[c] = np.maximum(self.var_[c], 1e-6)
        return self

    def _log_gaussian_pdf(self, class_idx, X):
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        log_prob = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * ((X - mean) ** 2) / var
        return np.sum(log_prob, axis=1)

    def predict(self, X):
        X = np.array(X)
        log_posteriors = []
        for c in self.classes_:
            prior = np.log(self.priors_[c] + 1e-9) 
            class_conditional = self._log_gaussian_pdf(c, X)
            log_posteriors.append(prior + class_conditional)
        log_posteriors = np.array(log_posteriors).T
        return self.classes_[np.argmax(log_posteriors, axis=1)]

# Train and evaluate scratch implementation
nb_scratch = ScratchGaussianNB().fit(X_train_scaled, y_train)
y_pred_scratch = nb_scratch.predict(X_test_scaled)

metrics_scratch = {
    'accuracy': accuracy_score(y_test, y_pred_scratch),
    'precision': precision_score(y_test, y_pred_scratch),
    'recall': recall_score(y_test, y_pred_scratch),
    'f1': f1_score(y_test, y_pred_scratch),
}

print("\n--- Scratch Gaussian Naive Bayes Metrics ---")
print(f"Accuracy: {metrics_scratch['accuracy']:.4f}")
print(f"Precision: {metrics_scratch['precision']:.4f}")
print(f"Recall: {metrics_scratch['recall']:.4f}")
print(f"F1-Score: {metrics_scratch['f1']:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_scratch, target_names=[str(k) for k in le.classes_]))

print("\nClass Mapping:", class_mapping)

# Confusion Matrix Visualization

# Plotting confusion matrix for scikit-learn
plt.figure(figsize=(8, 6))
cm_sklearn = confusion_matrix(y_test, y_pred_sklearn)
plt.matshow(cm_sklearn, cmap='Blues', fignum=1) 
for i in range(cm_sklearn.shape[0]):
    for j in range(cm_sklearn.shape[1]):
        plt.text(j, i, str(cm_sklearn[i][j]), ha='center', va='center', color='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Scikit-learn Naive Bayes Confusion Matrix\nAccuracy = {metrics_sklearn['accuracy']:.3f}")
# Use the actual class labels for ticks
plt.xticks(ticks=[0, 1], labels=[str(k) for k in le.classes_])
plt.yticks(ticks=[0, 1], labels=[str(k) for k in le.classes_])
plt.tight_layout()
plt.show() 

# Plotting confusion matrix for scratch implementation
plt.figure(figsize=(8, 6))
cm_scratch = confusion_matrix(y_test, y_pred_scratch)
plt.matshow(cm_scratch, cmap='Greens', fignum=1) 
for i in range(cm_scratch.shape[0]):
    for j in range(cm_scratch.shape[1]):
        plt.text(j, i, str(cm_scratch[i][j]), ha='center', va='center', color='black')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Scratch Naive Bayes Confusion Matrix\nAccuracy = {metrics_scratch['accuracy']:.3f}")
# Use the actual class labels for ticks
plt.xticks(ticks=[0, 1], labels=[str(k) for k in le.classes_])
plt.yticks(ticks=[0, 1], labels=[str(k) for k in le.classes_])
plt.tight_layout()
plt.show() 
