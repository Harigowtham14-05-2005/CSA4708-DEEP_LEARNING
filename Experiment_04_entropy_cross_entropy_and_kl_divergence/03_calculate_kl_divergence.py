# -*- coding: utf-8 -*-
"""
Experiment 4: Entropy, Cross-Entropy and KL Divergence
Topic / Question 3: Calculate KL Divergence
"""

import numpy as np
import matplotlib.pyplot as plt

P = np.array([0.5, 0.3, 0.2])
Q = np.array([0.4, 0.4, 0.2])

kl = np.sum(P * np.log(P / Q))
print("KL Divergence:", kl)

x = np.arange(3)
plt.bar(x - 0.2, P, width=0.4, label="P")
plt.bar(x + 0.2, Q, width=0.4, label="Q")
plt.xticks(x, ["Word 1", "Word 2", "Word 3"])
plt.ylabel("Probability")
plt.legend()
plt.title("Probability Distribution Comparison")
plt.show()
