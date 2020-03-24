#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 15:53:12 2019

@author: dchhitarka

Are a person's brain size and body size (Height and weight) predictive of his or her 
intelligence?

Import the iq_size.csv file

It Contains the details of 38 students, where

Column 1: The intelligence (PIQ) of students

Column 2:  The brain size (MRI) of students (given as count/10,000).

Column 3: The height (Height) of students (inches)

Column 4: The weight (Weight) of student (pounds)

What is the IQ of an individual with a given brain size of 90, height of 70 inches, 
and weight 150 pounds ?
Build an optimal model and conclude which is more useful in predicting intelligence 
Height, Weight or brain size.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('iq_size.csv')
df.isnull().any(axis=0)
df.dtypes

X = df.iloc[:,1:]
y = df.iloc[:,0]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

score_train = regressor.score(x_train, y_train)
score_test = regressor.score(x_test, y_test)

#building optimal model using back elimination
import statsmodels.api as sm
X=np.append(arr=np.ones((38,1)).astype(int), values=X, axis=1)

regressor_OLS=sm.OLS(y, X).fit()
regressor_OLS.summary()

X_opt = X[:,[0,1,2]]
regressor_OLS=sm.OLS(y, X_opt).fit()
regressor_OLS.summary()

# Probability for height is 0.009 while for brain is 0.001. So, brain is more optimal
X_opt = X[:,[0,1]]
regressor_OLS=sm.OLS(y, X_opt).fit()
regressor_OLS.summary()














