# -*- coding: utf-8 -*-
"""
Experiment 1: Vector and Matrix Operations for Deep Learning
Topic / Question 1: Flatten Image Matrix and Calculate Sum
"""

import numpy as np

A = np.array([[10, 20, 30],
              [40, 50, 60],
              [70, 80, 90]])

vector = A.flatten()

print("Flattened Vector:", vector)
print("Total Pixel Intensity:", np.sum(A))
