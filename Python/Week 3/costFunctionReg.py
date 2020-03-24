from sigmoid import sigmoid
import numpy as np
def costFunctionReg(theta, X, y, lamb):
    """
    %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
    %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
    %   theta as the parameter for regularized logistic regression and the
    %   gradient of the cost w.r.t. to the parameters. 
    """
    m = len(y)
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the cost of a particular choice of theta.
    %               You should set J to the cost.
    %               Compute the partial derivatives and set grad to the partial
    %               derivatives of the cost w.r.t. each parameter in theta
    """
    h = sigmoid(X @ theta)
    err =  (-y * np.log(h)) - (1 - y)* np.log(1 - h)
    J = (1/m) * sum(err) + (lamb/(2*m)) * (theta[1:].T @ theta[1:])
    return J

def gradInitial(theta, X,y, lamb):
    m = len(y)
    grad = np.zeros((m,1))
    h = sigmoid(X @ theta)
    grad = (1/m)*(X.T @ (h-y))
    grad[1:] = grad[1:] + lamb*theta[1:]
    return  grad