import numpy as np
from sigmoid import sigmoid

def lrCostFunction(theta, X, y, lambda_t):
    """    
    %LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
    %regularization
    %   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 
    """
    # Initialize some useful values
    m = len(y) #% number of training examples
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Hint: The computation of the cost function and gradients can be
    %       efficiently vectorized. For example, consider the computation
    %
    %           sigmoid(X * theta)
    %
    %       Each row of the resulting matrix will contain the value of the
    %       prediction for that example. You can make use of this to vectorize
    %       the cost function and gradient computations. 
    %
    % Hint: When computing the gradient of the regularized cost function, 
    %       there're many possible vectorized solutions, but one solution
    %       looks like:
    %           grad = (unregularized gradient for logistic regression)
    %           temp = theta; 
    %           temp(1) = 0;   % because we don't add anything for j = 0  
    %           grad = grad + YOUR_CODE_HERE (using the temp variable)
    %
    """
    h = sigmoid(X @ theta)
    theta_n = np.vstack((np.zeros((1,1)), theta[1:]))
    err =  (-y.T @ np.log(h)) - (1 - y).T @ np.log(1 - h)
    J = (1/m) * err + (lambda_t/(2*m))*(theta_n.T @ theta_n)
    return J
    
def gradFind(theta, X, y, lambda_t):
    m = X.shape[0]
    theta = theta.reshape(-1,1)
    h = sigmoid(X @ theta)
    print("b")
    grad = (1/m) * (X.T @ (h - y))
    print("c")
    theta_n = np.vstack((np.zeros((1,1)), theta[1:])) #For regularization only
    print("d")
    grad = grad + (lambda_t/m)*theta_n
    print("e")
    return grad
    