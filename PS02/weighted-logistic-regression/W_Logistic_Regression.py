import numpy as np
import matplotlib.pyplot as plt

# Load Data
X = np.loadtxt('features.txt')
y = np.loadtxt('labels.txt')

m = X.shape[0]

# Add bias
X = np.hstack([np.ones((m,1)), X])

# Sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic Regression (Newton)
def logistic_regression(X, y, lambda_=1e-4, iterations=15):
    m, n = X.shape
    theta = np.zeros(n)

    for _ in range(iterations):
        h = sigmoid(X @ theta)

        grad = -lambda_ * theta + (1/m) * X.T @ (y - h)

        S = np.diag(h * (1 - h))
        H = -lambda_ * np.eye(n) - (1/m) * X.T @ S @ X

        theta = theta - np.linalg.pinv(H) @ grad

    return theta

# Plot Linear Boundary
def plot_linear(theta, X, y):
    plt.figure()

    plt.scatter(X[y==0][:,1], X[y==0][:,2], label='Class 0')
    plt.scatter(X[y==1][:,1], X[y==1][:,2], label='Class 1')

    x_vals = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    y_vals = -(theta[0] + theta[1]*x_vals) / theta[2]

    plt.plot(x_vals, y_vals, label='Decision Boundary')
    plt.title("Logistic Regression (Linear)")
    plt.legend()
    plt.show()

# Compute Weights
def compute_weights(x_query, X, tau):
    diff = X - x_query
    norms = np.sum(diff**2, axis=1)
    w = np.exp(-norms / (2 * tau**2))
    return np.diag(w)

# Weighted Logistic Regression
def weighted_logistic(X, y, x_query, tau, iterations=10):
    m, n = X.shape
    theta = np.zeros(n)

    W = compute_weights(x_query, X, tau)

    for _ in range(iterations):
        h = sigmoid(X @ theta)

        grad = X.T @ W @ (y - h)

        S = np.diag(h * (1 - h))
        H = -X.T @ W @ S @ X

        H = H - 1e-5 * np.eye(n)

        theta = theta - np.linalg.pinv(H) @ grad

    return theta

# Plot Weighted Boundary
def plot_weighted(X, y, tau):
    plt.figure()

    plt.scatter(X[y==0][:,1], X[y==0][:,2], label='Class 0')
    plt.scatter(X[y==1][:,1], X[y==1][:,2], label='Class 1')

    grid_x = np.linspace(min(X[:,1]), max(X[:,1]), 60)
    grid_y = np.linspace(min(X[:,2]), max(X[:,2]), 60)

    Z = np.zeros((len(grid_y), len(grid_x)))

    for i in range(len(grid_x)):
        for j in range(len(grid_y)):
            x_query = np.array([1, grid_x[i], grid_y[j]])
            theta = weighted_logistic(X, y, x_query, tau)
            Z[j, i] = sigmoid(x_query @ theta)

    plt.contour(grid_x, grid_y, Z, levels=[0.5])
    plt.title(f"Weighted Logistic Regression (tau = {tau})")
    plt.legend()
    plt.show()

# Run
# Part A: Logistic Regression
theta = logistic_regression(X, y)
plot_linear(theta, X, y)

# Part B: Weighted (tau = 0.8)
plot_weighted(X, y, tau=0.8)

# Part C: Different taus
taus = [0.01, 0.05, 0.1, 0.5, 1, 5]

for tau in taus:
    plot_weighted(X, y, tau)