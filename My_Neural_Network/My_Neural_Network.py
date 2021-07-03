import numpy as np

from My_Neural_Network.functions import *
from My_Neural_Network.Neural_Network_Layers import Layer

class Neural_Network():

    def __init__(self, hidden_layers = (15, ), function = "relu", output_function = "linear", learning_rate = 0.1, random_state = None, nof_iterations = 500):
        #hidden_layers -> Number of nodes for each hidden layer, len of the tuple indicates the number of hidden layers
        #function -> The activation function used in the hidden layers. Can be "linear", "sigmoid", "tanh", "relu"
        #output_function -> The activation function used in the output layer. The most common is linear
        #learning_rate -> How fast the network learns
        #random_state -> A seed to get the same results for every test. Used for the np.random() function
        #nof_iterations -> The number of iterations the network will complete until it's considered trained

        if random_state:
            np.random.seed(random_state)

        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.func, self.func_der = self.decide_function(function)
        self.output_func, self.output_func_der = self.decide_function(output_function)
        self.nof_iterations = nof_iterations
    
    def fit(self, X, y):
        if y.ndim == 1: 
            y = y.reshape(-1, 1) #converting the array to 2d if it's not

        if X.shape[0] != y.shape[0]:
            raise Exception("Arrays of different lenght were passed") #tests for the input data

        self.layers = []
        input_size = X.shape[1]
        for i in self.hidden_layers: #the number of hidden layers the user wants
            self.layers.append(Layer(input_size, i, self.func, self.func_der, self.learning_rate)) 
            #creates a layer object and appends it in the self.layers list
            input_size = i
        self.layers.append(Layer(input_size, y.shape[1], self.output_func, self.output_func_der, self.learning_rate))
        #This is the output layer so the function passed is "linear" in order to be able to output any number
        #sigmoid with range(0, 1), tanh with (-1, 1) and relu with (0, inf) can't do that 
        
        self.train(X, y)

    def train(self, X, y): #Handles all the training
        if X.ndim == 2:
            X = np.expand_dims(X, 1)
        if y.ndim == 2:
            y = np.expand_dims(y, 1) #useful tranformations

        self.iterations = 0
        self.loss_curve = []
        self.predictions = []

        while self.iterations < self.nof_iterations:

            self.iterations += 1

            prediction = self.predict(X) #makes a prediction 
            self.predictions.append(prediction) #saves the prediction in a list so the user can use it to see the improvements

            output_error_der = 2*(prediction - y) #derivative of the loss function

            for i in reversed(self.layers): #for every layer begginign from the end,
                output_error_der = i.backprop(output_error_der) #it excecutes the backpropagation algorithm 
                
            output_error = self.loss(self.predict(X), y) #calculates the output error
            self.loss_curve.append(output_error.mean()) #and saves it in a list to be used for the loss curve graph
            
            # print(self.iterations, output_error.mean()) #for testing, to see the output error decrease while the network learns

        
    def predict(self, X): 
        inputs = X
        for i in self.layers: 
            output = i.forwardsprop(inputs)
            inputs = np.copy(output)
        #calculates the output of each layer and feeds it to the next as input
        return output #returns the final output
    
    def score(self, X, y): #calculates the R squared score of the neural netowrk
        if y.ndim == 1: 
            y = y.reshape(-1, 1)

        y_pred = self.predict(X)
        score = 1 - ((y - y_pred) ** 2).sum()/((y - y.mean()) ** 2).sum()
        return score

    def loss(self, pred, real): #the loss function of the neural network. 
        #The goal is to minimize it for the given set of data points
        return np.mean(((real - pred)**2), 0)

    def decide_function(self, inp):
        if inp == "sigmoid":
            return sigmoid, sigmoid_der
        if inp == "tanh":
            return tanh, tanh_der
        if inp == "relu":
            return relu, relu_der
        if inp == "linear":
            return linear, linear_der
        
        raise Exception("{0} is not a known function. Choose between \"sigmoid\", \"tanh\", \"relu\", \"linear\"".format(inp))





    


