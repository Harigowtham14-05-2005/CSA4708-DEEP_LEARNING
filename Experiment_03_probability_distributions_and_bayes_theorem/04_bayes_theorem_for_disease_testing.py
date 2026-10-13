# -*- coding: utf-8 -*-
"""
Experiment 3: Probability Distributions and Bayes’ Theorem
Topic / Question 4: Bayes’ Theorem for Disease Testing
"""

import matplotlib.pyplot as plt

P_D = 0.02
P_Pos_D = 0.95
P_Pos_notD = 0.05

P_Pos = P_Pos_D * P_D + P_Pos_notD * (1 - P_D)
posterior = P_Pos_D * P_D / P_Pos

print("Prior Probability:", P_D)
print("Posterior Probability:", posterior)

plt.bar(["Prior", "Posterior"], [P_D, posterior])
plt.ylabel("Probability")
plt.title("Disease Probability Before and After Positive Test")
plt.show()
