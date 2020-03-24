"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the second part
%  of the exercise which covers regularization with logistic regression.
%
%  You will need to complete the following functions in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column
%  contains the label (y).
"""
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex2data2.txt', delimiter=",")
X = data[:, [0, 1]]
y = data[:, 2].reshape(-1,1)

x1 = X[:,0]
x2 = X[:,1]
plt.scatter(x1[y[:,0] == 1], x2[y[:,0] == 1], c="k", marker="+", label="y=1")
plt.scatter(x1[y[:,0] == 0], x2[y[:,0] == 0], c="yellow", marker="o", label="y=0")
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend()
plt.show()

"""
%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
"""

#% Add Polynomial Features

#% Note that mapFeature also adds a column of ones for us, so the intercept
#% term is handled
from mapFeature import mapFeature
X = mapFeature(x1, x2)

#% Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

#% Set regularization parameter lambda to 1
lamb = 1;

#% Compute and display initial cost and gradient for regularized logistic
#% regression

from costFunctionReg import costFunctionReg, gradInitial 

cost = costFunctionReg(initial_theta, X, y, lamb)
grad = gradInitial(initial_theta, X, y, lamb)

print('Cost at initial theta (zeros): \n', cost[0][0])
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros) - first five values only:\n')
print(f"{grad[0][0]}\n{grad[1][0]}\n{grad[2][0]}\n{grad[3][0]}\n{grad[4][0]}")
print('Expected gradients (approx) - first five values only:\n')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n')

input('\nProgram paused. Press enter to continue.\n')

#% Compute and display cost and gradient
#% with all-ones theta and lambda = 10

test_theta = np.ones((X.shape[1],1))
cost = costFunctionReg(test_theta, X, y, 10)
grad = gradInitial(test_theta, X, y, 10)

print('\nCost at test theta (with lambda = 10): \n', cost)
print('Expected cost (approx): 3.16\n')
print('Gradient at test theta - first five values only:\n')
print(f"{grad[0][0]}\n{grad[1][0]}\n{grad[2][0]}\n{grad[3][0]}\n{grad[4][0]}")
print('Expected gradients (approx) - first five values only:\n')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n')

input('\nProgram paused. Press enter to continue.\n')

"""
%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%
"""

#% Initialize fitting parameters
initial_theta = np.zeros((X.shape[1], 1))

#% Set regularization parameter lambda to 1 (you should vary this)
lamb = 1;

import scipy.optimize as opt

output = opt.fmin_tnc(func = costFunctionReg, x0 = initial_theta.flatten(), fprime = gradInitial, args = (X, y.flatten(), lamb))

theta = output[0]
theta = theta.reshape(-1,1)
cost = costFunctionReg(theta, X, y, lamb)

# Plot
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta+0.5)

#mask = y.flatten() == 1
#passed = plt.scatter(X[mask][0], X[mask][1])
#failed = plt.scatter(X[~mask][0], X[~mask][1])
plt.scatter(x1[y[:,0] == 1], x2[y[:,0] == 1], c="k", marker="+", label="y=1")
plt.scatter(x1[y[:,0] == 0], x2[y[:,0] == 0], c="yellow", marker="o", label="y=0")
plt.contour(u,v,z,0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend()
plt.show()


#% Compute accuracy on our training set
from sigmoid import sigmoid
p = [sigmoid(X @ theta) >= 0.5]

print('Train Accuracy: \n', np.mean(p == y.flatten()) * 100);
print('Expected accuracy (with lambda = 1): 83.1 (approx)\n');


