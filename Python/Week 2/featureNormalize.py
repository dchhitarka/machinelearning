#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:18:51 2020
@author: dchhitarka
"""

import numpy as np
def featureNormalize(X):
    """
    %FEATURENORMALIZE Normalizes the features in X 
    %   FEATURENORMALIZE(X) returns a normalized version of X where
    %   the mean value of each feature is 0 and the standard deviation
    %   is 1. This is often a good preprocessing step to do when
    %   working with learning algorithms.
    
    % You need to set these values correctly
    """
    X_norm = X;
    mu = np.zeros((1, X.shape[1]));
    sigma = np.zeros((1, X.shape[1]));
    """
    % ====================== YOUR CODE HERE ======================
    % Instructions: First, for each feature dimension, compute the mean
    %               of the feature and subtract it from the dataset,
    %               storing the mean value in mu. Next, compute the 
    %               standard deviation of each feature and divide
    %               each feature by it's standard deviation, storing
    %               the standard deviation in sigma. 
    %
    %               Note that X is a matrix where each column is a 
    %               feature and each row is an example. You need 
    %               to perform the normalization separately for 
    %               each feature. 
    %
    # Hint: You might find the 'mean' and 'std' functions useful.
    """    
    mu = X.mean()
    X_norm = X_norm - mu
    sigma = X.std()
    X_norm = X_norm/sigma
    return (X_norm, mu, sigma)