# -*- coding: utf-8 -*-
"""
   Description :   Test SimpleLinearRegression with boston room price.
   Author :        xxm
"""

from sklearn import datasets
from mlic.linear_model import LinearRegression
from mlic.utils import train_test_split

"""load dataset"""
boston = datasets.load_boston()
X = boston.data
y = boston.target

"""process"""
X = X[y < 50.0]
y = y[y < 50.0]
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

"""train model"""
lr = LinearRegression()
lr.fit_normal(X_train, y_train)

"""view params"""
print(lr.coef_)
print(lr.intercept_)

print(lr.score(X_test, y_test))
