# -*- coding: utf-8 -*-
"""
   Description : 
   Author :        xxm
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

X_train, X_test, y_train, y_test = train_test_split(load_digits().data, load_digits().target)

best_p = -1
best_method = ""
best_k = -1
best_score = 0.0

start = time.time()

for p in range(1, 6):
    for method in ["uniform", "distance"]:
        for k in range(1, 11):
            knn_clf = KNeighborsClassifier(n_neighbors=k, weights=method, p=p)
            knn_clf.fit(X_train, y_train)
            score = knn_clf.score(X_test, y_test)

            if score > best_score:
                best_score = score
                best_p = p
                best_method = method
                best_k = k

end = time.time()

print("best_method =", best_method)
print("best_p =", best_p)
print("best_k =", best_k)
print("best_score =", best_score)

print('totally cost', end - start)
