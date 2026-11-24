import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load Dataset (Portable Path)

base_path = os.path.dirname(__file__)
file_path = os.path.join(base_path, 'data.csv')

df = pd.read_csv(file_path)

# 2. Preprocessing

y = df['Price'].values
df = df.drop(columns=['Price', 'Model', 'Trim'])

df = pd.get_dummies(df, columns=['Make', 'Type'], drop_first=True)

X = df.values

scaler = StandardScaler()
X = scaler.fit_transform(X)

# 3. Basis Functions

def polynomial_basis(X, degree):
    Phi = np.ones((X.shape[0], 1))
    for d in range(1, degree + 1):
        Phi = np.hstack((Phi, X ** d))
    return Phi

# 4. Gradient Descent + Early Stopping

def gradient_descent(Phi_train, y_train,
                     Phi_val, y_val,
                     lr=0.001,
                     max_iter=10000,
                     patience=100):

    N, d = Phi_train.shape
    w = np.zeros(d)

    best_val_loss = float('inf')
    best_w = None
    patience_counter = 0

    train_losses = []
    val_losses = []

    for i in range(max_iter):

        y_pred = Phi_train @ w
        error = y_pred - y_train
        grad = (2/N) * Phi_train.T @ error
        w -= lr * grad

        train_loss = np.mean(error**2)
        val_loss = np.mean((Phi_val @ w - y_val)**2)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w = w.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at iteration {i}")
            break

    return best_w, train_losses, val_losses

# 5. Experiment Function

def run_experiment(train_ratio, val_ratio, test_ratio, degree):

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42)

    val_relative = val_ratio / (train_ratio + val_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=42)

    Phi_train = polynomial_basis(X_train, degree)
    Phi_val = polynomial_basis(X_val, degree)
    Phi_test = polynomial_basis(X_test, degree)

    w, train_losses, val_losses = gradient_descent(
        Phi_train, y_train, Phi_val, y_val)

    test_loss = np.mean((Phi_test @ w - y_test)**2)

    return train_losses, val_losses, test_loss

# 6. Run All Required Experiments

splits = [
    (0.8, 0.1, 0.1),
    (0.7, 0.1, 0.2),
    (0.6, 0.1, 0.3)
]

degrees = [1, 2, 3]  # simple → more complex

print("\n================ FINAL RESULTS ================\n")

for split in splits:

    print(f"Split (Train/Val/Test): {split}")
    print("------------------------------------------------")

    for deg in degrees:

        train_losses, val_losses, test_loss = run_experiment(
            split[0], split[1], split[2], deg)

        print(f"Basis Degree: {deg}")
        print(f"Final Train Loss: {train_losses[-1]:.2f}")
        print(f"Final Validation Loss: {val_losses[-1]:.2f}")
        print(f"Test Loss: {test_loss:.2f}")
        print()

        # Plot Loss Curve (for full requirement coverage)
        plt.figure()
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f"Split {split}, Degree {deg}")
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        plt.legend()
        plt.show()

    print("================================================\n")