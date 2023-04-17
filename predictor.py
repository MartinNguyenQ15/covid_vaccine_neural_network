import pickle
import numpy as np
from network import Network
from training_tools import phrase_scoring


net = Network()

with open('./traindata.pkl', 'rb') as file:
    net = pickle.load(file)

phrase = input("Enter phrase to predict: ")

test_input = np.array([[phrase_scoring(phrase)]])

print(test_input)
output = net.predict(test_input)
print(output)
