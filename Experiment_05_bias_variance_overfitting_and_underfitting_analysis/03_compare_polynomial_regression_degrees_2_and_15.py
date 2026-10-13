# -*- coding: utf-8 -*-
"""
Experiment 5: Bias–Variance, Overfitting and Underfitting Analysis
Topic / Question 3: Compare Polynomial Regression Degrees 2 and 15
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.linspace(-3, 3, 50).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.2, 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

plt.scatter(X_train, y_train, label="Training Data")

for degree in [2, 15]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    print(f"Degree {degree} Training MSE:",
          mean_squared_error(y_train, model.predict(X_train)))
    print(f"Degree {degree} Testing MSE:",
          mean_squared_error(y_test, model.predict(X_test)))

    plt.plot(X_plot, model.predict(X_plot), label=f"Degree {degree}")

plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Underfitting vs Overfitting")
plt.legend()
plt.ylim(-3, 3)
plt.show()
