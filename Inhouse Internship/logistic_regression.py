# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 09:19:52 2019

@author: sharm
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv("Social_Network_Ads .csv")
df.isnull().any(axis=0)


features=df.iloc[:,2:4].values
labels=df.iloc[:,4].values

from sklearn.cross_validation import train_test_split
f_train, f_test, l_train, l_test=train_test_split(features, labels, test_size=0.25, random_state=0 )

# feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
f_train=sc.fit_transform(f_train)
f_test=sc.fit_transform(f_test)

#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(f_train, l_train)

# predict
labels_pred= lr.predict(f_test)

# making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(l_test,labels_pred)

score=lr.score(f_test,l_test)