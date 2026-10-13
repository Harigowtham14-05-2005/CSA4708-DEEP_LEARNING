# -*- coding: utf-8 -*-
"""
Experiment 3: Probability Distributions and Bayes’ Theorem
Topic / Question 1: Binomial Distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

n, p = 20, 0.1
x = np.arange(21)
prob = binom.pmf(x, n, p)

print("P(X = 3):", binom.pmf(3, n, p))

plt.bar(x, prob)
plt.xlabel("Number of Defective Components")
plt.ylabel("Probability")
plt.title("Binomial Distribution")
plt.show()
