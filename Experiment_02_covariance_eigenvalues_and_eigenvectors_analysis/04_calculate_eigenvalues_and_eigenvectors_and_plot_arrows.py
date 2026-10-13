# -*- coding: utf-8 -*-
"""
Experiment 2: Covariance, Eigenvalues and Eigenvectors Analysis
Topic / Question 4: Calculate Eigenvalues and Eigenvectors and Plot Arrows
"""

import numpy as np
import matplotlib.pyplot as plt

C = np.array([[3, 1],
              [1, 3]])

values, vectors = np.linalg.eig(C)

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

for i in range(len(values)):
    plt.quiver(0, 0, vectors[0, i], vectors[1, i],
               angles="xy", scale_units="xy", scale=1,
               label=f"Eigenvalue {values[i]}")

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0)
plt.axvline(0)
plt.grid()
plt.legend()
plt.title("Eigenvectors")
plt.show()
