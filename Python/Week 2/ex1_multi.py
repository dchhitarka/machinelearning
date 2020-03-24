#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:53:08 2020

@author: dchhitarka
"""

"""
%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m
%     plotData.m
%     gradientDescent.m
%     computeCost.m
%     gradientDescentMulti.m
%     computeCostMulti.m
%     featureNormalize.m
%     normalEqn.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization
"""
## ================ Part 1: Feature Normalization ================
print("Loading Data ...")
#Load Data
import pandas as pd
import numpy as np
data = pd.read_csv("ex1data2.txt", header=None)
X = data.iloc[:,[0,1]]
y = data.iloc[:,2]
m = len(y)

# Print out some data points
print('First 10 examples from the dataset: ');
print(f"x = \n{X.head(10)}, \ny = \n{y.head(10)}");

input('Program paused. Press enter to continue.\n');

# Scale features and set them to zero mean
print('Normalizing Features ...');
from featureNormalize import featureNormalize
X, mu, sigma = featureNormalize(X);
# Add intercept term to X
X = np.hstack((np.ones((m,1), dtype="float64"),X))

# Part 2: Gradient Descent
from gradientDescentMulti import gradientDescentMulti
print("Running Gradient Descent ...")
alpha = 0.01
num_iters = 400
theta = np.zeros((3,1))
theta, J_history = gradientDescentMulti(X,y, theta, alpha, num_iters)
print(theta)
print(J_history.shape)