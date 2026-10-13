# -*- coding: utf-8 -*-
"""
Experiment 4: Entropy, Cross-Entropy and KL Divergence
Topic / Question 1: Calculate Entropy of Weather Probabilities
"""

import numpy as np
import matplotlib.pyplot as plt

P = np.array([0.6, 0.3, 0.1])

entropy = -np.sum(P * np.log2(P))
print("Entropy:", entropy, "bits")

plt.bar(["Sunny", "Rainy", "Cloudy"], P)
plt.ylabel("Probability")
plt.title("Weather Probabilities")
plt.show()
