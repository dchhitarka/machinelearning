# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 17:13:34 2019

@author: sharm
"""

import numpy as np
import pandas as pd

#import the dataset
dataset=pd.read_csv("Salary_Classification.csv")
dataset.isnull().any(axis=0)

features=dataset.iloc[:,:-1].values
labels=dataset.iloc[:,-1].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder=LabelEncoder()

features[:,0]=labelencoder.fit_transform(features[:,0])

onehotencoder=OneHotEncoder(categorical_features=[0])
features= onehotencoder.fit_transform(features).toarray()  

#avoiding the dummy variable trap
features=features[:,1:]

# splitting into train and test dataset
from sklearn.cross_validation import train_test_split
f_train, f_test, l_train, l_test=train_test_split(features, labels, test_size=0.2, random_state=0)

#fitting multiple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(f_train, l_train)
    

#predicting the test results
Pred=regressor.predict(f_test)

#getting score
Score=regressor.score(f_train, l_train)

#building optimal model using back elimination
import statsmodels.formula.api as sm
features=np.append(arr=np.ones((30,1)).astype(int), values=features, axis=1)

features_opt=features[:,[0,1,2,3,4,5]]

regressor_OLS=sm.OLS(endog=labels, exog=features_opt).fit()
regressor_OLS.summary()
     


features_opt=features[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=labels, exog=features_opt).fit()
regressor_OLS.summary()

features_opt=features[:,[0,1,3,5]]
regressor_OLS=sm.OLS(endog=labels, exog=features_opt).fit()
regressor_OLS.summary()

features_opt=features[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=labels, exog=features_opt).fit()
regressor_OLS.summary()

features_opt=features[:,[0,5]]
regressor_OLS=sm.OLS(endog=labels, exog=features_opt).fit()
regressor_OLS.summary()

print(regressor_OLS.summary())