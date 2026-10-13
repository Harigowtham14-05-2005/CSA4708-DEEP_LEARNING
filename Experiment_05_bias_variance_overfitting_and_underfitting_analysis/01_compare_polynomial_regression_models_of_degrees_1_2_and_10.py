# -*- coding: utf-8 -*-
"""
Experiment 5: Bias–Variance, Overfitting and Underfitting Analysis
Topic / Question 1: Compare Polynomial Regression Models of Degrees 1, 2 and 10
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.linspace(0, 10, 30).reshape(-1, 1)
y = 2 * X.ravel()**2 + 3 * X.ravel() + np.random.normal(0, 10, 30)

X_plot = np.linspace(0, 10, 300).reshape(-1, 1)

plt.scatter(X, y, label="Actual Data")

for degree in [1, 2, 10]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    plt.plot(X_plot, model.predict(X_plot), label=f"Degree {degree}")

plt.xlabel("House Area")
plt.ylabel("House Price")
plt.title("Polynomial Regression Comparison")
plt.legend()
plt.show()
