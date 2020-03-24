#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:12:44 2019

@author: dchhitarka
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()


n_samples = len(digits.images)
features = digits.images.reshape(n_samples,-1)
labels = digits.target

from sklearn.model_selection import train_test_split as tts
f_train, f_test, l_train, l_test = tts(features, labels, test_size=0.2, random_state=10)

# KNN on MNIST
from sklearn.neighbors import KNeighborsClassifier
knn_best_score = 0
best_n = 0
for n in range(3,12):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(features[:n_samples // 2], labels[:n_samples // 2])

    knn_score = knn.score(features[:n_samples // 2], labels[:n_samples // 2])
    if(knn_score > knn_best_score):
        best_n = n

knn = KNeighborsClassifier(n_neighbors=best_n)
knn.fit(features[:n_samples // 2], labels[:n_samples // 2])

expected = labels[n_samples // 2:]
predicted = knn.predict(features[n_samples // 2:])

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(expected, predicted)
ac = accuracy_score(expected, predicted) #0.948832
knn_test_score_on_half = knn.score(features[n_samples // 2:], labels[n_samples // 2:]) #0.948832
knn_train_score_on_half = knn.score(features[:n_samples // 2], labels[:n_samples // 2]) #0.984409

#KNN on split data
knn.fit(f_train, l_train)
l_pred = knn.predict(f_test)
knn.score(f_test, l_test) #0.99166
knn.score(f_train, l_train) #0.98260
accuracy_score(l_test, l_pred)  #0.99166

