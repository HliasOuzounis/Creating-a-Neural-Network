import numpy as np

class Layer():

    def __init__(self, input_size, nodes, func, func_der, learning_rate):
        #input_size -> number of input nodes
        #nodes -> number of nodes in this layer
        #func, func_der -> activation function for this layer and its derivative
        #learning_rate -> How much to adjust the weights and biases after each iteration
        

        self.nodes = nodes
        self.weights = np.random.rand(input_size, self.nodes) - 0.5
        self.biases = np.random.rand(1, self.nodes) - 0.5 #Initializing weights and biases in range (-0.5, 0.5 )
        self.func = func
        self.func_der = func_der
        self.learning_rate = learning_rate
    
    def forwardsprop(self, inputs): #Calculatiing the output of this layer
        self.input = inputs
        self.output = self.func(np.add(np.dot(self.input, self.weights), self.biases)) # y = X * W + b
        return self.output
    
    def backprop(self, output_error): #The learning mechanism, calculates how much to adjust the weights and biases for the fastest improvemnet
        #For the equations that make this possible I suggest this series by 3blue1brown on youtube: 
        #https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi which I used to understand the math behind neural networks

        d_weights = np.mean([np.dot(i.T, np.multiply(j, self.func_der(np.add(np.dot(i, self.weights), self.biases)))) 
                            for i, j in zip(self.input, output_error)], 0) #How much the weigths need to change 
        d_biases = np.mean([np.multiply(j, self.func_der(np.add(np.dot(i, self.weights), self.biases))) 
                            for i, j in zip(self.input, output_error)], 0) #How much the biases need to change
        prev_output_error = np.array([np.dot(np.multiply(j, self.func_der(np.add(np.dot(i, self.weights), self.biases))), self.weights.T) 
                            for i, j in zip(self.input, output_error)]) #The error from the previous layer. It's used as the output error of the previous layer

        self.weights -= d_weights * self.learning_rate
        self.biases -= d_biases * self.learning_rate #adjusting the weights and biases

        return prev_output_error
