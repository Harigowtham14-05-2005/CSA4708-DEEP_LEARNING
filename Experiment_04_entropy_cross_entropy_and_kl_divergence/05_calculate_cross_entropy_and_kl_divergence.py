# -*- coding: utf-8 -*-
"""
Experiment 4: Entropy, Cross-Entropy and KL Divergence
Topic / Question 5: Calculate Cross-Entropy and KL Divergence
"""

import numpy as np
import matplotlib.pyplot as plt

Y = np.array([0, 1, 0, 0])
P = np.array([0.1, 0.6, 0.2, 0.1])

cross_entropy = -np.sum(Y * np.log(P))
kl = np.sum(Y[Y > 0] * np.log(Y[Y > 0] / P[Y > 0]))

print("Cross-Entropy Loss:", cross_entropy)
print("KL Divergence:", kl)

x = np.arange(4)
plt.bar(x - 0.2, Y, width=0.4, label="Actual")
plt.bar(x + 0.2, P, width=0.4, label="Predicted")
plt.xticks(x, ["Class 1", "Class 2", "Class 3", "Class 4"])
plt.ylabel("Probability")
plt.legend()
plt.title("Actual vs Predicted Distribution")
plt.show()
