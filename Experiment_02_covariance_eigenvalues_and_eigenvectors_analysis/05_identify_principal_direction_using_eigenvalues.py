# -*- coding: utf-8 -*-
"""
Experiment 2: Covariance, Eigenvalues and Eigenvectors Analysis
Topic / Question 5: Identify Principal Direction Using Eigenvalues
"""

import numpy as np
import matplotlib.pyplot as plt

C = np.array([[5, 2],
              [2, 2]])

values, vectors = np.linalg.eig(C)
idx = np.argmax(values)
principal = vectors[:, idx]

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)
print("Largest Eigenvalue:", values[idx])
print("Principal Eigenvector:", principal)

for i in range(len(values)):
    plt.quiver(0, 0, vectors[0, i], vectors[1, i],
               angles="xy", scale_units="xy", scale=1,
               label=f"Eigenvalue {values[i]:.2f}")

plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0)
plt.axvline(0)
plt.grid()
plt.legend()
plt.title("Principal Direction of Variation")
plt.show()
