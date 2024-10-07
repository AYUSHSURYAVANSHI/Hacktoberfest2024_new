import numpy as np
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # 100 samples, single feature
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 4 + 3x + noise

# Add the bias term (intercept) to the feature matrix
X_b = np.c_[np.ones((100, 1)), X]  # add x0 = 1 for the bias term

# Linear Regression formula: theta_best = (X_b.T * X_b)^(-1) * X_b.T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("Intercept (bias):", theta_best[0])
print("Slope:", theta_best[1])

# Prediction
X_new = np.array([[0], [2]])  # values for which we want to predict
X_new_b = np.c_[np.ones((2, 1)), X_new]  # add bias term
y_predict = X_new_b.dot(theta_best)

# Plotting the results
plt.plot(X_new, y_predict, "r-", label="Prediction")
plt.plot(X, y, "b.", label="Data Points")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
