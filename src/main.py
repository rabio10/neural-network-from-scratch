from network import Network
import numpy as np
import random
from keras.datasets import mnist



if __name__ == "__main__":
    # load data 
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    layers_sizes = [28*28,8,8,10]
    print(train_X.shape)

    train_data = [(x.flatten().reshape(-1, 1)/253,y) for x,y in zip(train_X, train_y)]
    test_data = [(x.flatten().reshape(-1, 1)/253,y) for x,y in zip(test_X, test_y)]
    
    net = Network(layers_sizes)

    # let's train
    print("BEGIN training")
    net.train(100,train_data,0.3, test_data=test_data)
