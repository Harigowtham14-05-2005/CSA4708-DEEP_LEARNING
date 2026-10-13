# -*- coding: utf-8 -*-
"""
Experiment 1: Vector and Matrix Operations for Deep Learning
Topic / Question 3: Matrix Multiplication
"""

import numpy as np

X = np.array([[30, 60],
              [35, 70]])

W = np.array([[0.2, 0.5],
              [0.4, 0.3]])

output = np.dot(X, W)

print("Predicted Output Matrix:\n", output)
