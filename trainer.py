import pandas as pd
import numpy as np
from layer import ActivationLayer, FullyConnectedLayer
from network import Network
from layerops import tanh_prime, mse_prime, tanh, mse
from threading import Thread, Lock
from itertools import islice
from training_tools import phrase_scoring, all_categories
import pickle
import time

def train(lock, net, i, o):
    with lock:
        net.fit(i, o, steps=1000, learning_rate=0.1)

start_time = time.time()
items = 150
items_per_cat = items // 3
thread_count = 16
lock = Lock()

print("Reading data")
df = pd.read_csv('./clean/l2.csv')
query_category_tuples = [(x, y) for x, y in zip(df['query'], df['category'])]
df = None
print("Reading data complete. +{:0.2f}s".format(time.time() - start_time))
since = time.time()

print("Creating sample space")
sample_space = []
sample_space.extend([(x, y) for x,y in query_category_tuples if y == "covid19_vaccination"][0:items_per_cat])
sample_space.extend([(x, y) for x,y in query_category_tuples if y == "safety_side_effects"][0:items_per_cat])
sample_space.extend([(x, y) for x,y in query_category_tuples if y == "vaccination_intent"][0:items_per_cat])
print("Sample space complete. Total time: +{:0.2f}s".format(time.time() - since))
since = time.time()

lens = len(sample_space)
splice_size = lens // thread_count

inputs = []
outputs = []
print("Building input/output spaces from sample space")
for x, y in sample_space:
    c, i = phrase_scoring(x)
    category_output = all_categories[y]
    inputs.append([[c, i]])
    outputs.append([category_output])
sample_space = None
print("Building complete. Total time: +{:0.2f}s".format(time.time() - since))
since = time.time()

training_data_input = np.array(inputs)
training_data_output = np.array(outputs)
inputs = None
outputs = None

print("Slicing spaces")
iter_input = iter(training_data_input)
iter_output = iter(training_data_output)
sliced_input = [list(islice(iter_input, splice_size))
                for _ in range(lens // splice_size)]
sliced_output = [list(islice(iter_output, splice_size))
                 for _ in range(lens // splice_size)]
training_data_input = None
training_data_output = None
iter_output = None
iter_input = None
print("Slicing complete. Created {0} slice(s). +{1:0.2f}s".format(thread_count, time.time() - since))
since = time.time()

print("Creating Neural Network")
net = Network()
net.add(FullyConnectedLayer(2, 12))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(12, 6))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FullyConnectedLayer(6, 2))
net.add(ActivationLayer(tanh, tanh_prime))
net.use(mse, mse_prime)
print("Network complete. +{:0.2f}s".format(time.time() - since))
since = time.time()

print("Training started.")
threads = []
for i, o in zip(sliced_input, sliced_output):
    t = Thread(target=train, args=(lock, net, i, o))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("Training complete. +{:0.2f}s".format(time.time() - since))
since = time.time()

print("Saving to file")
with open('traindata.pkl', 'wb') as file:
    pickle.dump(net, file)
print("Saving complete. +{:0.2f}s".format(time.time() - since))

print("="*50)

print("Used {} threads for {:,} item(s).".format(
    thread_count, "all" if items == None else items))
print("Network Stats:")
print("\tError Percentage: {}%".format(net.err * 100))
print("\tNumber of layers: {}".format(len(net.layers)))
print("Total time: {:0.2f}s".format(time.time() - start_time))

