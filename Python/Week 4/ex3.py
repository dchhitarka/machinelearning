"""
%% Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

#% =========== Part 1: Loading and Visualizing Data =============
#%  We start the exercise by first loading and visualizing the dataset.
#%  You will be working with a dataset that contains handwritten digits.

#% Load Training Data
print('Loading and Visualizing Data ...\n')
data = loadmat('ex3data1.mat') #% training data stored in arrays X, y
X = data['X']
y = data['y']
y[y==10] = 0
m = X.shape[0]

#% Randomly select 100 data points to display
rand_indices = np.random.choice(m, 100)
sel = X[rand_indices, :]
yel = y[rand_indices, :]

fig = plt.figure(figsize=(50, 50))
for i in range(100):
    sub = fig.add_subplot(10, 10, i + 1)
    sub.imshow(sel[i].reshape(20,20).T)
    plt.gray()
#    sub.set_title(yel[i])
    sub.axis("off")
plt.show()

input('Program paused. Press enter to continue.\n')
"""
%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%
"""
#% Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization');

X_t = np.hstack((np.ones((len(y),1)),X))
y_t = np.eye(10)[y].reshape(-1,10)


from lrCostFunction import lrCostFunction, gradFind

lambda_t = 3;
X_s = np.hstack((np.ones((5,1)), (np.arange(1,16).reshape(3,5)).T /10))
y_s = np.array([1,0,1,0,1]).reshape(-1,1)
theta_s = np.array([-2, -1, 1, 2])[:,np.newaxis]

J    = lrCostFunction(theta_s, X_s, y_s, lambda_t)
grad = gradFind(theta_s, X_s, y_s, lambda_t)

print('\nCost:', J);
print('Expected cost: 2.534819\n');
print('Gradients:\n');
print(grad);
print('Expected gradients:');
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

input('Program paused. Press enter to continue.\n');

## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

# Not WORKING
from oneVsAll import oneVsAll

lambda_n = 0.1;
theta_opt = oneVsAll(X_t, y_t, 10, lambda_n)

input('Program paused. Press enter to continue.\n')

## ================ Part 3: Predict for One-Vs-All ================
from predictOneVsAll import predictOneVsAll

pred = predictOneVsAll(theta_opt, X);
print('\nTraining Set Accuracy:\n', np.mean(pred == y) * 100);

