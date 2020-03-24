import numpy as np
def mapFeature(X1, X2):
    """
    % MAPFEATURE Feature mapping function to polynomial features
    %
    %   MAPFEATURE(X1, X2) maps the two input features
    %   to quadratic features used in the regularization exercise.
    %
    %   Returns a new feature array with more features, comprising of 
    %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    %
    %   Inputs X1, X2 must be the same size
    %
    """
    degree = 6
    out = np.ones((len(X1)))[:, np.newaxis] # [:, np.newaxis] reshape the array in (m,1) 
    for i in range(1,degree+1):
        for j in range(i+1):
            out = np.hstack((out, (X1** (i-j) * (X2**j))[:,np.newaxis]))
    return out

