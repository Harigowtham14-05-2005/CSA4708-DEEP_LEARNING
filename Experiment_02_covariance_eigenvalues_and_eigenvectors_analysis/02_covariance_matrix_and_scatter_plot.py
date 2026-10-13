# -*- coding: utf-8 -*-
"""
Experiment 2: Covariance, Eigenvalues and Eigenvectors Analysis
Topic / Question 2: Covariance Matrix and Scatter Plot
"""

import numpy as np
import matplotlib.pyplot as plt

temperature = np.array([22, 25, 28, 30, 32])
humidity = np.array([80, 75, 68, 60, 55])

data = np.column_stack((temperature, humidity))
print("Covariance Matrix:\n", np.cov(data, rowvar=False))

plt.scatter(temperature, humidity)
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.title("Temperature vs Humidity")
plt.grid()
plt.show()
