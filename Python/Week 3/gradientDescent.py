import numpy as np
from sigmoid import sigmoid
from costFunction import costFunction

def gradientDescent(X, y, theta, alpha, num_iters, lamb=0):
    m = len(y)
    x1 = X[:,[1]]
    x2 = X[:,[2]]  
    print("Initial Theta -\n",theta)
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = sigmoid(X@theta)
        theta[0]  = theta[0] - alpha*(1/m) * sum(h-y)
        theta[1]  = theta[1] - alpha*(1/m) * sum((h - y)*x1)
        theta[2]  = theta[0] - alpha*(1/m) * sum((h - y)*x2)
        J_history[i] = costFunction(theta, X,y)
        print(i," -",J_history[i])
        print(theta)
        
    return (J_history, theta)