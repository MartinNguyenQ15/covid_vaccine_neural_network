# Testing: Solving XOR
import numpy as np
from layer import ActivationLayer, FullyConnectedLayer
from network import Network
from layerops import tanh, tanh_prime, mse, mse_prime

training_input = np.array([
    [[0, 0]],
    [[0, 1]],
    [[1, 0]],
    [[1, 1]],
])

training_output = np.array([ [[0]], [[1]], [[1]], [[0]] ])

test_input = np.array([
    [[0, 1]],
    [[0, 0]],
    [[1, 1]],
    [[1, 0]],
])

print(training_input.shape)
print(training_input)

net = Network()
net.add(FullyConnectedLayer(2, 12))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(12, 6))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(6, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.fit(training_input, training_output, steps=1000, learning_rate=0.1)

out = net.predict(test_input)
print(out)
