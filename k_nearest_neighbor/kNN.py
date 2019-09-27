# -*- coding: utf-8 -*-
"""
   Description :   knn
   Author :        xxm
"""

import numpy as np
from math import sqrt
from collections import Counter
from k_nearest_neighbor.metric import accuracy_score


class KNNClassifier:

    def __init__(self, k):
        """
        init KNNClassifier
        :param k: k
        """
        assert k >= 1, "k must be valid."

        self.k = k
        self._X_train = None
        self._y_train = None
        self.flag = False

    def fit(self, X_train, y_train):
        """
        fit KNNClassifier with X_train and y_train
        :param X_train: data
        :param y_train: label
        :return: KNNClassifier
        """
        assert X_train.shape[0] == y_train.shape[0], "the size of X_train must be equal to the size of y_train"
        assert self.k <= X_train.shape[0], "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        self.flag = True
        return self

    def predict(self, X_predict):
        """
        predict X_predict with KNNClassifier
        :param X_predict:
        :return: all result
        """
        assert self.flag, "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], "the feature number of X_predict must be equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """
        predict x with KNNClassifier
        :param x:
        :return: signal result
        """
        assert x.shape[0] == self._X_train.shape[1], "the feature number of x must be equal to X_train"

        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """
        accuracy
        :param X_test:
        :param y_test:
        :return:
        """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN (k=%d)" % self.k
