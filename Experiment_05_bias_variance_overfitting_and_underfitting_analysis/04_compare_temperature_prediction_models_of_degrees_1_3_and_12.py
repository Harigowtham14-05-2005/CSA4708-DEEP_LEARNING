# -*- coding: utf-8 -*-
"""
Experiment 5: Bias–Variance, Overfitting and Underfitting Analysis
Topic / Question 4: Compare Temperature Prediction Models of Degrees 1, 3 and 12
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

np.random.seed(42)

X = np.linspace(0, 24, 100).reshape(-1, 1)
y = 25 + 5 * np.sin(X.ravel() * np.pi / 12) + np.random.normal(0, 1, 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_plot = np.linspace(0, 24, 300).reshape(-1, 1)

for degree in [1, 3, 12]:
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X_train, y_train)

    print(f"Degree {degree} Training MSE:",
          mean_squared_error(y_train, model.predict(X_train)))
    print(f"Degree {degree} Testing MSE:",
          mean_squared_error(y_test, model.predict(X_test)))

    plt.plot(X_plot, model.predict(X_plot), label=f"Degree {degree}")

plt.scatter(X, y, label="Actual Data")
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.title("Temperature Model Comparison")
plt.legend()
plt.show()
