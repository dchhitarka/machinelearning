#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 23:29:03 2019

@author: dchhitarka
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('Social_Network_Ads .csv')
df.head()
df.info()
x = df.iloc[:,1:-1]
y = df.iloc[:,-1]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
scale = StandardScaler()
x[['Age', 'EstimatedSalary']] = scale.fit_transform(x[['Age', 'EstimatedSalary']])
le = LabelEncoder()
x.Gender = le.fit_transform(x.Gender)

ohe = OneHotEncoder(categorical_features = [0])
x = ohe.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split
f_train, f_test, l_train, l_test= train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.svm import SVC
classifier=SVC(kernel="sigmoid", random_state=0)
classifier.fit(f_train, l_train)

labels_pred=classifier.predict(f_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(l_test, labels_pred)

score_train = classifier.score(f_train, l_train)

score_test = classifier.score(f_test, l_test)

