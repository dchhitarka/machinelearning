import numpy as np
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, iterations):
    m = len(y) #no. of training examples
    J_history = np.zeros((iterations, 1))
    for iter in range(iterations):
        x1 = X[:,1]
        h = theta[0] + theta[1]*x1
        temp0 = theta[0] - alpha*(1/m)*sum(h-y) #*x0 as it is one so we dont write it
        temp1 = theta[1] - alpha*(1/m)*sum((h-y)*x1) #element wise product 
        theta = np.array([temp0, temp1])
        theta.resize(2,1)
        # ==================
        # Save the cost J in every iterations
        # J_history[iterations] = computeCost(X,y,theta)
        
    return (theta, J_history)
