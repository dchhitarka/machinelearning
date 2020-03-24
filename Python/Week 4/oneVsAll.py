import numpy as np
from lrCostFunction import lrCostFunction, gradFind
from scipy import optimize as opt

def oneVsAll(X_t, y, num_labels, lambda_t):
    """
    %ONEVSALL trains multiple logistic regression classifiers and returns all
    %the classifiers in a matrix all_theta, where the i-th row of all_theta 
    %corresponds to the classifier for label i
    %   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
    %   logistic regression classifiers and returns each of these classifiers
    %   in a matrix all_theta, where the i-th row of all_theta corresponds 
    %   to the classifier for label i
    """
    #% Some useful variables
    #m = np.size(X, 0)
    n = np.size(X_t, 1)
    
    #% You need to return the following variables correctly 
    theta_t = np.zeros((n,num_labels))
    
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the following code to train num_labels
    %               logistic regression classifiers with regularization
    %               parameter lambda. 
    %
    % Hint: theta(:) will return a column vector.
    %
    % Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
    %       whether the ground truth is true/false for this class.
    %
    % Note: For this assignment, we recommend using fmincg to optimize the cost
    %       function. It is okay to use a for-loop (for c = 1:num_labels) to
    %       loop over the different classes.
    %
    %       fmincg works similarly to fminunc, but is more efficient when we
    %       are dealing with large number of parameters.
    %
    % Example Code for fmincg:
    """
    #   % Set Initial theta
#    initial_theta = np.zeros((n + 1, 1))
         
    def findOptParam(p_num):
        outcome = np.array(y == p_num).astype(int)
        initial_theta = theta_t[:,p_num][:, np.newaxis]
        temp = opt.fmin_tnc(func = lrCostFunction, 
                    x0 = initial_theta.flatten(),fprime = gradFind, 
                    args = (X_t, outcome.flatten(), lambda_t)
                    )
        theta_t[:,p_num] = temp[0]
    
    for digit in range(10):
        findOptParam(digit)     
        
    return theta_t
