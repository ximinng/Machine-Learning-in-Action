# -*- coding: utf-8 -*-
"""
   Description :   Test KNN algorithm with iris
   Author :        xxm
"""

import numpy as np
import matplotlib as plt
from sklearn import datasets
from mlic.utils import train_test_split, accuracy_score
from mlic.neighbors import KNNClassifier

"""load and split dataset"""
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, seed=666)

"""fit KNN Classifier"""
knn_clf = KNNClassifier(k=3)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)

"""calculate accuracy score"""
acc = accuracy_score(y_test, y_predict)
