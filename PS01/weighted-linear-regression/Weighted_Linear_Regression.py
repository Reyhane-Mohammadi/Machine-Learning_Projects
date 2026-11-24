import numpy as np
import matplotlib.pyplot as plt


# 1) Load dataset 

# Each file contains one number per line
x = np.loadtxt("features.txt")   
t = np.loadtxt("labels.txt")     

# Ensure they are 1D arrays
x = x.reshape(-1)
t = t.reshape(-1)

# 2) Simple Linear Regression (Ordinary Least Squares)
# Model: y = w0 + w1*x

# Design matrix with bias term
X = np.column_stack([np.ones_like(x), x])

# Solve normal equation using least squares
w_ols = np.linalg.lstsq(X, t, rcond=None)[0]

# Prediction function for OLS
def predict_ols(x_query):
    return w_ols[0] + w_ols[1] * x_query


# 3) Weighted Linear Regression (Locally Weighted)
# Weight formula:
# u_i = exp( -2 (x_i - x')^2 / tau^2 )

def lwlr_predict(x_train, t_train, x_query, tau):

    # Design matrix
    X_train = np.column_stack([np.ones_like(x_train), x_train])

    # Compute weights relative to x_query
    weights = np.exp(-2.0 * (x_train - x_query)**2 / (tau**2))

    # Create diagonal weight matrix
    U = np.diag(weights)

    # Weighted normal equation
    A = X_train.T @ U @ X_train
    b = X_train.T @ U @ t_train

    # Solve for parameters
    w = np.linalg.pinv(A) @ b

    # Return prediction at x_query
    return w[0] + w[1] * x_query


# 4) Plot results for different tau values

taus = [0.1, 0.3, 2, 10]

# Create smooth grid for plotting lines
x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)

plt.figure(figsize=(9, 5))

# Plot original data
plt.scatter(x, t, s=20, alpha=0.8, label="Data")

# Plot OLS line
plt.plot(x_grid, predict_ols(x_grid),
         linewidth=2, label="Simple Linear Regression (OLS)")

# Plot weighted regression curves
for tau in taus:
    y_values = np.array([lwlr_predict(x, t, xg, tau) for xg in x_grid])
    plt.plot(x_grid, y_values, linewidth=2, label=f"Weighted LR (tau={tau})")

plt.xlabel("Feature (x)")
plt.ylabel("Target (t)")
plt.title("Simple vs Weighted Linear Regression")
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig("q3_tau_comparison.png", dpi=300)
plt.show()

print("OLS parameters:")
print("w0 =", w_ols[0])
print("w1 =", w_ols[1])
print("\nPlot saved as: q3_tau_comparison.png")