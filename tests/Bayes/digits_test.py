# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
from sklearn import datasets
from mlic.utils import train_test_split, accuracy_score
from mlic.naive_bayes import BernoulliNB

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

bernoulliNB = BernoulliNB()
bernoulliNB.fit(X_train, y_train)

y_predict = bernoulliNB.predict(X_test)

accuracy_score = accuracy_score(y_test, y_predict)

print(accuracy_score)
