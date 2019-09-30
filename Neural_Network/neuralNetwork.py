# -*- coding: utf-8 -*-
"""
   Description :   neural network
   Author :        xxm
"""
import numpy as np
import scipy.special as func

import matplotlib.pyplot as plt


class neuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """initialise the neural network"""

        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.lr = learning_rate

        # link weight matrices, wih(weight input2hidden) and who(weight hidden2output)
        self.wih = np.random.normal(loc=0.0,
                                    scale=pow(self.hnodes, -0.5),
                                    size=(self.hnodes, self.inodes))
        self.who = np.random.normal(loc=0.0,
                                    scale=pow(self.hnodes, -0.5),
                                    size=(self.onodes, self.hnodes))

        # activation function is the sigmoid
        self.activation_function = lambda x: func.expit(x)

    def train(self, inputs_list, targets_list):
        """train the neural network"""

        # convert inputs list to 2d list
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate the signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # error
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hiddne_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                     np.transpose(hidden_outputs))

        # update the weights for the links between the hidden and output layers
        self.wih += self.lr * np.dot((hiddne_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                     np.transpose(inputs))
        pass

    def query(self, inputs_list):
        """query the neural network"""

        # convert inputs list to 2d list
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate the signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    # number of input,hidden and output nodes
    input_nodes = 784  # 28 x 28
    hidden_nodes = 100
    output_nodes = 10  # 0...9

    # learning rate is 0.3
    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # load the mnist training data csv file into a list
    training_data_file = open("/Users/ximingxing/MNIST/mnist_train.csv", "r")
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # train the neural network

    # go though all records in the training data set
    for record in training_data_list:
        all_values = record.split(",")
        # scale and shift the inputs
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.1
        # create the target output values
        targets = np.zeros(output_nodes) + 0.1
        targets[int(all_values[0])] = 0.99

        n.train(inputs, targets)
        pass
