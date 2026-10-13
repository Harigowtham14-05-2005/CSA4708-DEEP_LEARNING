# -*- coding: utf-8 -*-
"""
Experiment 2: Covariance, Eigenvalues and Eigenvectors Analysis
Topic / Question 1: Sample Covariance and Scatter Plot
"""

import numpy as np
import matplotlib.pyplot as plt

hours = np.array([2, 4, 6, 8, 10])
marks = np.array([35, 45, 60, 75, 85])

cov = np.cov(hours, marks)[0, 1]
print("Sample Covariance:", cov)

plt.scatter(hours, marks)
plt.xlabel("Study Hours")
plt.ylabel("Examination Marks")
plt.title("Study Hours vs Marks")
plt.grid()
plt.show()
