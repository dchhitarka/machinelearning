import numpy as np

def sigmoid(z):
    #SIGMOID Compute sigmoid functoon
    g = 1/ (1 + np.exp(-z))
    return g
