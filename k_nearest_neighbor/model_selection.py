# -*- coding: utf-8 -*-
"""
   Description :   model_selection
   Author :        xxm
"""
import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, "test_ration must be valid"

    if seed:
        np.random.seed(seed)

    shuffled_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffled_indexes[:test_size]
    train_indexes = shuffled_indexes[test_size:]

    X_train = X[train_indexes]
    y_train = y[train_indexes]

    X_test = X[test_indexes]
    y_test = y[test_indexes]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    from sklearn import datasets
    from k_nearest_neighbor.kNN import KNNClassifier

    iris = datasets.load_iris()

    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, 0.2, seed=666)

    knn_clf = KNNClassifier(k=3)
    knn_clf.fit(X_train, y_train)
    y_predict = knn_clf.predict(X_test)

    print("预测结果: ", sum(y_test == y_predict))
    print("准确率: ", sum(y_predict == y_test) / len(y_test))
