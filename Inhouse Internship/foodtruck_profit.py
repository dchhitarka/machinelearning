#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:55:25 2019

@author: dchhitarka

You will implement linear regression to predict the profits for a food chain company.

Case: Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet. The chain already has food-trucks in various cities and you have data for profits and populations from the cities. You would like to use this data to help you select which city to expand to next.

Foodtruck.csv contains the dataset for our linear regression problem. The first column is the population of a city and the second column is the profit of a food truck in that city. A negative value for profit indicates a loss.

Perform Simple Linear regression to predict the profit based on the population observed and visualize the result.
Based on the above trained results, what will be your estimated profit, if you set up your outlet in Jaipur? (Current population in Jaipur is 3.073 million)
Source Code Submission
"""
# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np

# Read the data from csv and make pandas DataFrame
df = pd.read_csv('Foodtruck.csv')

# Check for any null values in any column
df.isnull().any(axis=0)

df.columns
plt.plot(df['Population'], df['Profit'])

# Here X and y are DataFrames because I used [[]] if I used [] only, then it will be series
X = df[['Population']] 
y = df[['Profit']]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

score_test = lr.score(X_test, y_test)
score_train = lr.score(X_train, y_train)
#score_pred = lr.score(X_test, y_pred)

# Visualize training Data
plt.scatter(X_train, y_train, color='red')
plt.title('Population vs Profit')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

# Visualize fit line
plt.scatter(X_train, lr.predict(X_train), color='blue')
plt.title('Population vs Profit')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

# Visualize test data
plt.scatter(X_test, y_pred, color='red')
plt.scatter(X_test, y_test, color='blue')
plt.title('Population vs Profit')
plt.xlabel('Population')
plt.ylabel('Profit')
plt.show()

from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)

mean_squared_error(y_train, lr.predict(X_train))
