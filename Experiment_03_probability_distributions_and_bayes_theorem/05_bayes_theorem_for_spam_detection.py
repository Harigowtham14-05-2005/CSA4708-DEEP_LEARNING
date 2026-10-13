# -*- coding: utf-8 -*-
"""
Experiment 3: Probability Distributions and Bayes’ Theorem
Topic / Question 5: Bayes’ Theorem for Spam Detection
"""

import matplotlib.pyplot as plt

P_S = 0.30
P_W_S = 0.70
P_W_notS = 0.10

P_W = P_W_S * P_S + P_W_notS * (1 - P_S)
posterior = P_W_S * P_S / P_W

print("Prior Spam Probability:", P_S)
print("Posterior Spam Probability:", posterior)

plt.bar(["Prior", "Posterior"], [P_S, posterior])
plt.ylabel("Probability")
plt.title("Spam Probability After Observing 'prize'")
plt.show()
