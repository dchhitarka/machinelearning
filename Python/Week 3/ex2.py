"""
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoid.py
%     costFunction.py
%     predict.py
%     costFunctionReg.py
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Data
# The first two columns contains the exam scores and the third column
#  contains the label.

data = pd.read_csv('ex2data1.txt', header=None);
data.columns = ["Exam1", "Exam2", "Label"]
X = data.iloc[:, [0,1]]
y = data.iloc[:,2]
y = y.to_numpy()
# Observe the data
#data.head()
#X.head()
#y.head()

"""
%% ==================== Part 1: Plotting ====================
%  We start the exercise by first plotting the data to understand the 
%  the problem we are working with.
"""
print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n')
# In octvae file, there is a plotData() function used to plot.
y1 = data[data["Label"]==1] # Contain rows where y = 1
y0 = data[data["Label"]==0] # Contain rows where y = 0
plt.scatter(y1["Exam1"], y1["Exam2"], c="k", marker="+")
plt.scatter(y0["Exam1"], y0["Exam2"], c="orange", marker="o")
# Alternate one line approach but have same marker
#plt.scatter(data["Exam1"], data["Exam2"], c=data["Label"])
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
plt.legend(["Admitted", "Not Admitted"])
plt.show()

input("Program paused. Press enter to continue.")

"""
%% ============ Part 2: Compute Cost and Gradient ============
%  In this part of the exercise, you will implement the cost and gradient
%  for logistic regression. You neeed to complete the code in 
%  costFunction.m
"""
#%  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

#% Add intercept term to x and X_test
X = np.hstack((np.ones((m, 1)), X))
y = y.reshape(-1,1)
#% Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

#% Compute and display initial cost and gradient
from costFunction import costFunction, gradInitial

cost = costFunction(initial_theta, X, y)
grad = gradInitial(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693\n')
print('Gradient at initial theta (zeros): ')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

#% Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost = costFunction(test_theta, X, y)
grad = gradInitial(test_theta, X, y)

print('\nCost at test theta:', cost)
print('Expected cost (approx): 0.218\n')
print('Gradient at test theta:')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('Program paused. Press enter to continue.')

## ---- Done TILL HERE. Do from below next time
"""
%% ============= Part 3: Optimizing using built-in function =============
%  In this exercise, you will use a built-in function to find the
%  optimal parameters theta.
"""
# Here we have multiple options. We have to use optimize class from scipy which has diff built-in funcxtions to find optimised values. Or we can write a gradient function similar to the one in last week with nbo of iterations to find theta.
import scipy.optimize as opt
temp = opt.fmin_tnc(func = costFunction, 
                    x0 = initial_theta.flatten(),fprime = gradInitial, 
                    args = (X, y.flatten()))
#the output of above function is a tuple whose first element #contains the optimized values of theta
theta = temp[0]
theta = theta.reshape(-1,1)
cost = costFunction(theta, X, y)
print('Cost at theta found by ?', cost)
print('Expected cost (approx): 0.203\n')
print('theta:')
print(theta)
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201\n')

#% Plot Boundary
plt.scatter(y1["Exam1"], y1["Exam2"], c="k", marker="+")
plt.scatter(y0["Exam1"], y0["Exam2"], c="orange", marker="o")
#On;y 2 endpoints req to plot line. So, find min and max points
plot_x = np.array([np.min(X[:,1]-2), np.max(X[:,2]+2)]).reshape(-1,1)
# Calculate decision boundary
plot_y = (-1/theta[2])*(theta[0] + plot_x @ theta[1])  
plt.plot(plot_x, plot_y)
plt.xlabel("Exam 1")
plt.ylabel("Exam 2")
plt.legend(["Admitted", "Not Admitted"])
plt.show()

input('\nProgram paused. Press enter to continue.\n')

"""
%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 
"""
from sigmoid import sigmoid 
prob = sigmoid(np.array([1, 45, 85]) @ theta);
print('For a student with scores 45 and 85, we predict an admission probability of', prob)
print('Expected value: 0.775 +/- 0.002\n')

#% Compute accuracy on our training set
p = [sigmoid(X@theta) >= 0.5]
acc = np.mean(p == y)
print('Train Accuracy:\n', acc * 100)
print('Expected accuracy (approx): 89.0\n')
