import time
import numpy as np
import random



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def inverse_sigmoid(y, epsilon=1e-10):
    y_clipped = np.clip(y, epsilon, 1.0 - epsilon)
    return np.log(y_clipped / (1 - y_clipped))

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x)
    # Normalize by dividing by the sum of exponentials
    return exp_x / np.sum(exp_x)

class Network():
    def __init__(self,layers_sizes, activation="sigmoid"):
        self.layers_sizes = layers_sizes
        self.num_layers = len(layers_sizes)
        self.activation_func = activation

        # list of all weight matrices 
        self.weights = [np.random.randn(i,j) for i,j in zip(layers_sizes[1:], layers_sizes[:-1])]
        # list of biases
        self.biases =  [np.random.randn(i,1) for i in layers_sizes[1:]]
        
        # list to store activations
        self.activations = [np.zeros((i,1)) for i in layers_sizes]


    def feed_forward(self, initial_activations):
        """
        initial_activations : should already be normalized and flatten (elements / 255)
        returns : 
                    last activation
        """
        activation = self.activation_func
        self.activations[0] = initial_activations
        # go through all layers EXCEPT the final one
        # choose activation of hidden layers (sigmoid, relu)
        # choose activation of last layer (sigmoid, softmax)
        for l in range(self.num_layers-2):
            temp = ( self.weights[l] @ self.activations[l] ) + self.biases[l]
            # do the activation choosen
            if activation == "sigmoid":
                act = sigmoid(temp)
            elif activation == "relu":
                act = relu(temp)
            # save the activation
            self.activations[l+1] = act
        
        # final layer
        L = self.num_layers-2
        temp = ( self.weights[L] @ self.activations[L] ) + self.biases[L]
        # do the activation choosen
        if activation == "sigmoid":
            act = sigmoid(temp)
        elif activation == "relu":
            act = softmax(temp)
        # save the activation
        self.activations[L+1] = act

        return self.activations[-1]

    def test_accuracy(self, test_data):
        """
        test_data : list of tuples (train_x, train_y), i.e. shape is [(np(28*28),1), (np(28*28),1)]
        """
        counter = 0
        test_data_len = len(test_data)
        for i in range(test_data_len):
            last_act = self.feed_forward(test_data[i][0])
            y_pred = self.decision(last_act)
            # compare with target 
            if y_pred == test_data[i][1]:
                counter += 1
        
        return counter / test_data_len

    
    def decision(self, activation):
        indice_max = np.argmax(activation)
        return indice_max
    
    def train(self, epoch, train_data, learning_rate, test_data=None, mini_batch_size=32):
        """
        train_data : list of tuples (train_x, train_y), i.e. shape is [(np(28*28),1), (np(28*28),1)]
        it should be flattened 
        """
        data_len = len(train_data)
        for ep in range(epoch):
            time1 = time.time()
            # shuffle data
            random.shuffle(train_data)
            # divide data into mini batches
            train_mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, data_len, mini_batch_size)]
            # pass every batche to learning
            for mb in train_mini_batches:
                self.update_mini_batch(mb, learning_rate)
            
            time2 = time.time()
            # lets test and print epoch info 
            print(f"Epoch {ep}/{epoch} : done in {time2 - time1:.2f}", end="")
            if test_data != None:
                print(f", accuracy={self.test_accuracy(test_data)}")
            

    def update_mini_batch(self,mini_batch, learning_rate):
        """
        update the weights and biaises of neural network after each batch of training data.\n
        mini_batch is a list of tuples (x,y)\n
        x it's shape is :  (28*28)
        y it's shape is : (1)
        """
        dc_dw = [np.zeros((i,j)) for i,j in zip(self.layers_sizes[1:], self.layers_sizes[:-1])]
        dc_db = [np.zeros((i,1)) for i in self.layers_sizes[1:]]

        for datapoint in mini_batch: # datapoint is a tuple
            # get derivatives for this datapoint
            dc_dw_datapoint, dc_db_datapoint = self.backprop(datapoint)
            
            # sum it with the others
            dc_dw = [dc_dw__ + dc_dw__dtpt for dc_dw__, dc_dw__dtpt in zip(dc_dw, dc_dw_datapoint)]
            dc_db = [dc_db__ + dc_db__dtpt for dc_db__, dc_db__dtpt in zip(dc_db, dc_db_datapoint)]
        
        # calculate the derivative 
        dc_dw = [d / len(mini_batch) for d in dc_dw]
        dc_db = [d / len(mini_batch) for d in dc_db]

        # update the wieghts and biases
        self.weights = [w - (learning_rate * dw) for w, dw in zip(self.weights, dc_dw)]
        self.biases = [b - (learning_rate * db) for b, db in zip(self.biases, dc_db)]


    def backprop(self, datapoint):
        """
        datapoint : is a tuple (x,y)
        NB: for now only works with activation_func="sigmoid", because there's no inverse of relu and softmax here
        """
        # initialize
        dc_dw_datapoint = [np.zeros((i,j)) for i,j in zip(self.layers_sizes[1:], self.layers_sizes[:-1])]
        dc_db_datapoint = [np.zeros((i,1)) for i in self.layers_sizes[1:]]

        # create a np array for y_datapoint
        y_datapoint = np.eye(10)[datapoint[1]].reshape(-1, 1)
        # feed forward
        last_act = self.feed_forward(datapoint[0])

        dc_da = 2 * last_act * (last_act - y_datapoint)
        # da_dz = sigmoid_prime(z)
        da_dz = sigmoid_prime(inverse_sigmoid(last_act))

        for l in range(self.num_layers-2,-1,-1):
            dc_dz = dc_da * da_dz

            dc_dw_datapoint[l] = np.dot(self.activations[l], dc_dz.transpose()).transpose()
            dc_db_datapoint[l] = dc_dz

            dc_da = self.weights[l].transpose() @ dc_dz
            da_dz = sigmoid_prime(inverse_sigmoid(self.activations[l]))
        
        return dc_dw_datapoint, dc_db_datapoint

        
