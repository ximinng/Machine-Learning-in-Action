# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
import matplotlib
import matplotlib.pyplot as plt
from sklearn import datasets


def accuracy_score(y_true, y_predict):
    """
    计算y_true和y_predict之间的准确率
    :param y_true:
    :param y_predict:
    :return:
    """
    assert y_true.shape[0] == y_predict.shape[0], "the size of y_true must be equal to the size of y_predict"

    return sum(y_true == y_predict) / len(y_true)


if __name__ == '__main__':
    """
    Task: KNN based on digits
    """
    digits = datasets.load_digits()
    digits.keys()

    X = digits.data
    y = digits.target

    some_digit = X[666]
    some_digit_image = some_digit.reshape(8, 8)

    plt.imshow(some_digit_image, cmap=matplotlib.cm.binary)
    plt.show()

    from KNN.function.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)

    from KNN.function.kNN import KNNClassifier

    knn_clf = KNNClassifier(k=3)
    knn_clf.fit(X_train, y_train)
    y_predict = knn_clf.predict(X_test)

    print(accuracy_score(y_test, y_predict))
