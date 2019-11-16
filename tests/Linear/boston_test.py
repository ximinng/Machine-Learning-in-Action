# -*- coding: utf-8 -*-
"""
   Description :   Test SimpleLinearRegression with boston room price.
   Author :        xxm
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from mlic.utils import train_test_split
from mlic.linear_model import SimpleLinearRegression
from mlic.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

"""load data"""
boston = datasets.load_boston()
x = boston.data[:, 5]  # rooms
y = boston.target

plt.scatter(x, y)
plt.show()

"""clean data"""
x = x[y < 50.0]
y = y[y < 50.0]

plt.scatter(x, y)
plt.show()

"""data split"""
x_train, x_test, y_train, y_test = train_test_split(x, y, seed=666)

"""train model"""
slr = SimpleLinearRegression()
slr.fit(x_train, y_train)

"""what a,b does model learn"""
plt.scatter(x_train, y_train)
plt.scatter(x_test, y_test, color="c")
plt.plot(x_train, slr.predict(x_train), color='r')
plt.show()

"""MSE"""
y_predict = slr.predict(x_test)
mse_test = mean_squared_error(y_test, y_predict)

"""RMSE"""
rmse_test = root_mean_squared_error(y_test, y_predict)

"""MAE"""
mae_test = mean_absolute_error(y_test, y_predict)
