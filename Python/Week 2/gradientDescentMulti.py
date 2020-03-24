#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 17:55:15 2020

@author: dchhitarka
"""
import numpy as np
from computeCost import computeCost
def gradientDescentMulti(X,y,theta,alpha,num_iters):
    print(1)
    m = len(y) # no. of training examples
    yh = y.to_numpy().reshape(-1,1)
    J_history = np.zeros((num_iters, 1))
    print(2)
    for iter in range(num_iters):
        # Your code
        h = X@theta
        temp = [0 for _ in range(len(theta))]
        temp[0] = theta[0] - alpha*(1/m)*sum(h-yh) #*x0 as it is one so we dont write it
        print(3)
        for i in range(1,len(theta)):
            temp[i] = theta[i][0] - alpha*(1/m)*sum((h-yh)*X[:,i]) #element wise product 
#        temp2 = theta[2] - alpha*(1/m)*sum((h-yh)*X[:,2]) #element wise product 
        print(4)
        theta = np.array(temp)
        theta.resize(3,1)
        # ==================
        # Save the cost J in every iterations
#        J_history[iter] = computeCost(X,y,theta)
        print(5)
    return (theta, J_history)
