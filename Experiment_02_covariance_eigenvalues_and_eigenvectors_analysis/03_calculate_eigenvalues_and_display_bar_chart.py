# -*- coding: utf-8 -*-
"""
Experiment 2: Covariance, Eigenvalues and Eigenvectors Analysis
Topic / Question 3: Calculate Eigenvalues and Display Bar Chart
"""

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[4, 2],
              [2, 4]])

values = np.linalg.eigvals(A)
print("Eigenvalues:", values)

plt.bar(range(len(values)), values)
plt.xlabel("Eigenvalue Index")
plt.ylabel("Value")
plt.title("Eigenvalues")
plt.show()
