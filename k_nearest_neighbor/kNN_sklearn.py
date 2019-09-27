# -*- coding: utf-8 -*-
"""
   Description :  knn with sklearn
   Author :        xxm
"""
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""
Task one: iris
"""
iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=666)

knn_clf = KNeighborsClassifier(n_neighbors=6)
knn_clf.fit(X_train, y_train)
y_predict = knn_clf.predict(X_test)

print(accuracy_score(y_test, y_predict))

"""
Task two: digits
"""
digits = datasets.load_digits()

digits_X_train, digits_X_test, digits_y_train, digits_y_test = train_test_split(digits.data, digits.target)
knn_clf.fit(digits_X_train, digits_y_train)
digits_y_predict = knn_clf.predict(digits_X_test)

print(accuracy_score(digits_y_test, digits_y_predict))
