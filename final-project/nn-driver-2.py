import numpy as np
import pandas as pd
import numpy as np
from threading import Thread, Lock
from itertools import islice
import pickle
import time
import os

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
        

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def mse(y, ynext):
    return np.mean(np.power(y - ynext, 2))

def mse_prime(y, ynext):
    return 2 * (ynext - y) / y.size


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.err = 0

    def add (self, layer):
        self.layers.append(layer)

    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            forward_data = input_data[i]
            for layer in self.layers:
                forward_data = layer.forward_propagation(forward_data)
            result.append(forward_data)
        
        return result

    def fit(self, x, y, steps, learning_rate):
        samples_per_step = len(x)

        for _ in range(steps):
            for j in range(samples_per_step):
                forward = x[j]
                for layer in self.layers:
                    forward = layer.forward_propagation(forward)
                self.err += self.loss(y[j], forward)
                backwards = self.loss_prime(y[j], forward)
                for layer in reversed(self.layers):
                    backwards = layer.backwards_propagation(backwards, learning_rate)
            self.err /= samples_per_step
            print("{:0.2f}% Training Complete".format((_ / steps) * 100))


print("Reading data")
df = pd.read_csv('./cali_dataset.csv')
selected_columns = ['county', 'administered_date', 'pfizer_doses', 'jj_doses', 'moderna_doses']

without = df[selected_columns]
without = without[without['county'] == 'Calaveras']
without = without.loc[without['administered_date'].str.contains(r'|'.join(['2022', '2021', '2020']), regex=True)]

without['administered_date'] = pd.to_datetime(without['administered_date'])
without['administered_date'] = without['administered_date'].apply(lambda x: x.toordinal())
 
inputs = np.array([without.iloc[:, 1].values])
outputs = np.array([without.iloc[:,2:].values])

print(outputs)

print("Creating Neural Network")
net = Network()
net.add(FullyConnectedLayer(1, 12))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(12, 6))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(6, 3))
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse, mse_prime)

if (os.path.exists('./traindata.pkl')):
    with open('./traindata.pkl', 'rb') as f:
        net = pickle.load(f)
else:
    print("Training Started")
    for i, o in zip(inputs, outputs):
        net.fit(i, o, steps=1000, learning_rate=0.1)
    print("Training complete")
    print("Saving to file")
    with open('traindata.pkl', 'wb') as file:
        pickle.dump(net, file)
    print("Saving complete")

print(net.predict([
    [738521],
]))

