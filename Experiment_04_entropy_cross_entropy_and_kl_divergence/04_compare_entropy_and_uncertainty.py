# -*- coding: utf-8 -*-
"""
Experiment 4: Entropy, Cross-Entropy and KL Divergence
Topic / Question 4: Compare Entropy and Uncertainty
"""

import numpy as np
import matplotlib.pyplot as plt

P = np.array([0.9, 0.1])
Q = np.array([0.5, 0.5])

H_P = -np.sum(P * np.log2(P))
H_Q = -np.sum(Q * np.log2(Q))

print("Entropy of P:", H_P)
print("Entropy of Q:", H_Q)
print("Greater Uncertainty:", "Q" if H_Q > H_P else "P")

plt.bar(["P", "Q"], [H_P, H_Q])
plt.ylabel("Entropy (bits)")
plt.title("Entropy Comparison")
plt.show()
