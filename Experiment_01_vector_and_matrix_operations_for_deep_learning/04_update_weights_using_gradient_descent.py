# -*- coding: utf-8 -*-
"""
Experiment 1: Vector and Matrix Operations for Deep Learning
Topic / Question 4: Update Weights Using Gradient Descent
"""

import numpy as np

W = np.array([0.7, 0.5, 0.9])
G = np.array([0.2, 0.1, 0.3])
learning_rate = 0.1

updated_W = W - learning_rate * G

print("Updated Weights:", updated_W)
