import numpy as np
from sigmoid import sigmoid

def costFunction(theta, X, y):
    """
    %COSTFUNCTION Compute cost and gradient for logistic regression
    %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
    %   parameter for logistic regression and the gradient of the cost
    %   w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y) # number of training examples
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    %
    % Note: grad should have the same dimensions as theta
    """
#    theta = theta.reshape(-1,1)
    h = sigmoid(X@theta)
    err =  (np.log(h) * -y) - np.log(1 - h)*(1 - y) 
    J = (1/m) * sum(err)
    return J

# Partial Derivatives
def gradInitial(theta, X, y):
    m = len(y) # number of training examples
#    grad = np.zeros((len(theta), 1))
#    x1 = X[:,[1]]
#    x2 = X[:,[2]]
    h = sigmoid(X@theta)
#    t0  = (1/m) * sum(h-y)
#    t1  = (1/m) * sum((h - y)*x1)
#    t2  = (1/m) * sum((h - y)*x2)
#    grad = np.array([t0, t1, t2])
    return (1/m) * X.T @ (h-y)


