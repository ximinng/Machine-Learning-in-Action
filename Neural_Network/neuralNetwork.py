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

    def train(self,
              inputs_list,
              targets_list):
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

    print(n.query([1.0, 0.5, -1.5]))
