# -*- coding: utf-8 -*-
"""
Experiment 4: Entropy, Cross-Entropy and KL Divergence
Topic / Question 2: Calculate Cross-Entropy Loss
"""

import numpy as np
import matplotlib.pyplot as plt

Y = np.array([1, 0, 0])
P = np.array([0.7, 0.2, 0.1])

loss = -np.sum(Y * np.log(P))
print("Cross-Entropy Loss:", loss)

x = np.arange(3)
plt.bar(x - 0.2, Y, width=0.4, label="Actual")
plt.bar(x + 0.2, P, width=0.4, label="Predicted")
plt.xticks(x, ["Cat", "Dog", "Bird"])
plt.ylabel("Probability")
plt.legend()
plt.title("Actual vs Predicted Probabilities")
plt.show()
