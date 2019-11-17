# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import numpy as np


def accuracy_score(y_true, y_predict):
    """
    Calculate the accuracy between y_true and y_predict
    :return: acc
    """
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)


def mean_squared_error(y_true, y_predict):
    """
    Mean squares error between y_true and y_predict
    :return:
    """
    assert len(y_true) == len(y_predict), "the size of y_true must be equal to the size of y_predict"

    return np.sum((y_true - y_predict) ** 2) / len(y_true)


def root_mean_squared_error(y_true, y_predict):
    """
    Root Mean squares error between y_true and y_predict
    :return:
    """

    return np.sqrt(mean_squared_error(y_true, y_predict))


def mean_absolute_error(y_true, y_predict):
    """
    Meaning absolute error between y_true and y_predict
    :return:
    """

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)


def R_square(y_true, y_predict):
    """
    R Square : 1 - MSE(y,y_predict) / Var(y)
    :return:
    """
    return 1 - (mean_squared_error(y_true, y_predict)) / np.var(y_true)
