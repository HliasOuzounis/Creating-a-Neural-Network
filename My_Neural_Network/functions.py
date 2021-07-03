import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x)) 

def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_der(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(x, 0)

def relu_der(x):
    return (x > 0) * 1

def linear(x):
    return x

def linear_der(x):
    return 1
