#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:49:35 2019

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
f_train, f_test, l_train, l_test = tts(features, labels, test_size=0.2, random_state=1)

from sklearn.linear_model import LogisticRegression
logRes = LogisticRegression()
logRes.fit(features[:n_samples // 2], labels[:n_samples // 2])

pred = logRes.predict(features[n_samples//2:])
actual = labels[n_samples//2:]

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
f1score = f1_score(actual, pred, average='weighted') #0.916998
ac = accuracy_score(actual, pred) #0.91657397
lr_train_score = logRes.score(features[:n_samples // 2], labels[:n_samples // 2]) #1.0

#With split data
logRes.fit(f_train, l_train)
lr_pred = logRes.predict(f_test)
accuracy_score(l_test, lr_pred) #0.96944
logRes.score(f_train,l_train) #0.99652052

