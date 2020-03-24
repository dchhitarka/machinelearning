import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Part 1: Basic Function
print('Running warmUpExercise ...')
print('5x5 Identity Matrix:')

from warmUpExercise import warmUpExercise
print(warmUpExercise())
input('Program paused. Press enter to continue.')

# Part 2: Plotting 
print("Plotting Data ...")
data = pd.read_csv('ex1data1.txt', header=None)
# data.columns = ["A","B"]
# print(data.head())
X = data.iloc[:, 0] # data[0]
y = data.iloc[:,1] # data[1]
m = len(y)

# Plot Data
plt.scatter(X,y, c="orange",marker="+", linewidths=1)
plt.show()
input('Program paused. Press enter to continue.')

ones = np.ones((m), dtype="float64")
X = np.vstack((ones, X))
X = X.T # Add a column of ones for x0 features
theta = np.zeros((2,1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

print('Testing the cost function ...')
# compute and display initial cost
from computeCost import computeCost
J = computeCost(X, y, theta);

print('With theta = [0 ; 0]\nCost computed =', J);
print('Expected cost value (approx) 32.07\n');

# further testing of the cost function
J = computeCost(X, y, [[-1] ,[ 2]]);
print('With theta = [-1 ; 2]\nCost computed =', J);
print('Expected cost value (approx) 54.24');

input('Program paused. Press enter to continue.');

print('\nRunning Gradient Descent ...')
# run gradient descent
from gradientDescent import gradientDescent
theta, J = gradientDescent(X, y, theta, alpha, iterations);

# print theta to screen
print('Theta found by gradient descent:\n');
print('', theta);
print('Expected theta values (approx)\n');
print(' -3.6303\n  1.1664\n');

# Plot the linear fit
plt.plot(X[:,1],y, "+")
plt.plot(X[:,1], X@theta)
plt.legend(["Training Data","Linear Regression"])
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5])@theta;
print('For population = 35,000, we predict a profit of',predict1[0]*10000);
predict2 = np.array([1, 7]) @ theta;
print('For population = 70,000, we predict a profit of',predict2[0]*10000);

input('Program paused. Press enter to continue.\n');

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100);
theta1_vals = np.linspace(-1, 4, 100);

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)));

# Fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
         t = np.array([theta0_vals[i], theta1_vals[j]])
         J_vals[i,j] = computeCost(X, y, t.reshape(2,1));

J_vals = J_vals.T

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(theta0_vals, theta1_vals, J_vals)
ax.set_title("Cost Function Plot")
plt.show()

#Contour Plot
