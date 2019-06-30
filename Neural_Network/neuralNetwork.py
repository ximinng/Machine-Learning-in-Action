# -*- coding: utf-8 -*-
"""
   Description : neural network class definition
   Author :        xxm
"""
import numpy as np
import scipy.special as func


class neuralNetwork:

    def __init__(self,
                 input_nodes,
                 hidden_nodes,
                 output_nodes,
                 learning_rate):
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

    def train(self):
        pass

    def query(self,
              inputs_list):
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
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3

    learning_rate = 0.5

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
