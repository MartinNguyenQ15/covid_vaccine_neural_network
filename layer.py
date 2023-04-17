import numpy as np

class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward_propagation(self, input_data):
        raise NotImplementedError

    def backwards_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    def backwards_propagation(self, delta_bias, learning_rate):
        input_error = np.dot(delta_bias, self.weights.T)
        weights_error = np.dot(self.input.T, delta_bias)
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * delta_bias
        return input_error
    

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backwards_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
        

