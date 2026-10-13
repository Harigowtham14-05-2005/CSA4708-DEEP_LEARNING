# -*- coding: utf-8 -*-
"""
Experiment 3: Probability Distributions and Bayes’ Theorem
Topic / Question 3: Poisson Distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

lam = 4
x = np.arange(13)
prob = poisson.pmf(x, lam)

print("P(X = 6):", poisson.pmf(6, lam))

plt.bar(x, prob)
plt.xlabel("Customer Arrivals")
plt.ylabel("Probability")
plt.title("Poisson Distribution")
plt.show()
