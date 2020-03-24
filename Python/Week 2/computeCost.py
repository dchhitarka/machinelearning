import numpy as np

def computeCost(X,y,theta):
    m = len(y)
#    x1 = X[:,1]

    h = X@theta #Hypothesis = theta0 + theta1 * X
    err = (h - y.to_numpy().reshape(-1,1))**2 #MSE between predicted and ctual
    J = 1/(2*m) * np.sum(err) #Cost Function = (1/m) sum(err)  
    return J