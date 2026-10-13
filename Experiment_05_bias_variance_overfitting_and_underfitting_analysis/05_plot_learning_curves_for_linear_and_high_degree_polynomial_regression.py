# -*- coding: utf-8 -*-
"""
Experiment 5: Bias–Variance, Overfitting and Underfitting Analysis
Topic / Question 5: Plot Learning Curves for Linear and High-Degree Polynomial Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = np.sin(X.ravel()) + np.random.normal(0, 0.2, 100)

models = {
    "Linear Regression": make_pipeline(
        PolynomialFeatures(1), LinearRegression()
    ),
    "Polynomial Degree 15": make_pipeline(
        PolynomialFeatures(15), LinearRegression()
    )
}

for name, model in models.items():
    sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=5,
        scoring="neg_mean_squared_error"
    )

    train_error = -train_scores.mean(axis=1)
    val_error = -val_scores.mean(axis=1)

    plt.plot(sizes, train_error, marker="o", label=f"{name} Training")
    plt.plot(sizes, val_error, marker="s", label=f"{name} Validation")

plt.xlabel("Number of Training Samples")
plt.ylabel("Mean Squared Error")
plt.title("Learning Curves")
plt.legend()
plt.grid()
plt.show()
