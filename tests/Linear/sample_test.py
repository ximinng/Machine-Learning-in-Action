# -*- coding: utf-8 -*-
"""
   Description :   Test SimpleLinearRegression
   Author :        xxm
"""
import numpy as np
import matplotlib.pyplot as plt
from mlic.linear_model import SimpleLinearRegression

"""make fake data"""
m = 100000
big_x = np.random.random(size=m)
big_y = big_x * 2 + 3 + np.random.normal(size=m)

"""show data distribution"""
plt.scatter(big_x, big_y)
plt.show()

"""train model"""
reg = SimpleLinearRegression()
reg.fit(big_x, big_y)

"""what does model learn"""
print(reg.a_)
print(reg.b_)
