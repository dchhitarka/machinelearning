import numpy as np
from scipy.special import expit
def sigmoid(z):
    g = np.zeros((len(z)))
    """
    %SIGMOID Compute sigmoid function
    %   g = SIGMOID(z) computes the sigmoid of z.
    % You need to return the following variables correctly
    """
    g = expit(z)
#    g = 1/(1 + np.exp(-z))
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    %               vector or scalar).
    """
    return g
