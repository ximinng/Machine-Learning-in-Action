# -*- coding: utf-8 -*-
"""
   Description :   Simple Linear Regression learning unary function
   Author :        xxm
"""

import numpy as np
from mlic.metrics import R_square


class SimpleLinearRegression:

    def __init__(self):
        """init Simple Linear Regression"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """
        Train Simple Linear Regression model
        :param x_train: x_train_vector ndim == 1
        :param y_train: y_train_vector ndim == 1
        :return: self
        """
        assert x_train.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        """vector calculate"""
        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        """do not use for loop"""
        # num = 0.0
        # d = 0.0
        # for x, y in zip(x_train, y_train):
        #     num += (x - x_mean) * (y - y_mean)
        #     d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """
        Predict based on x_predict
        :param x_predict: vector ndim == 1
        :return: result vector of prediction ndim == 1
        """
        assert x_predict.ndim == 1, "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """
        Predict based on single x
        :param x_single: scale
        :return: scale
        """
        return self.a_ * x_single + self.b_

    def score(self, x_test, y_test):
        """
        Evaluate the trained model.
        :return: scale
        """
        y_predict = self.predict(x_test)
        return R_square(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression()"
