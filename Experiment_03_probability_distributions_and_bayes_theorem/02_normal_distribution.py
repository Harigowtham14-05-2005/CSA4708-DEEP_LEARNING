# -*- coding: utf-8 -*-
"""
Experiment 3: Probability Distributions and Bayes’ Theorem
Topic / Question 2: Normal Distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mean, sd = 70, 10

probability = norm.cdf(80, mean, sd) - norm.cdf(60, mean, sd)
print("P(60 <= X <= 80):", probability)

x = np.linspace(40, 100, 500)
y = norm.pdf(x, mean, sd)

plt.plot(x, y)
plt.fill_between(x, y, where=(x >= 60) & (x <= 80), alpha=0.4)
plt.xlabel("Marks")
plt.ylabel("Probability Density")
plt.title("Normal Distribution")
plt.show()
