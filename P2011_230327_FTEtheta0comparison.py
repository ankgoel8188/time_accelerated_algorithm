"""
Filename:   P2011_230327_FTEtheta0comparison.py
Author:     Turibius Rozario
Email:      s175@umbc.edu
Advisor:    Dr. Ankit Goel
Created:    March 27, 2023
Minimizes a single NN layer using Adam, SGD, FTE (finite time estimator).
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Optimizer
import math
import numpy as np
from P20_ClassesFunctions import FTEstimator, generate_Xi

# Global Variables
EPOCHS = 500
NI = 10  # Number of Input features
NO = 5  # Number of Output features
NS = 100  # Number of Samples
SCALAR = 1
ALPHA_1 = 0.5
ALPHA_2 = 2.5
C_1 = 1.5
C_2 = 1.5
DelT = 0.1
TESTS = 10
FILE_START = "P2011/"
LAYER_TYPE = "3"


# Class Definitions
class Model(nn.Module):
    """
    Base model for our setup
    """
    def __init__(self, seed=1):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(NI, NO, dtype=torch.float64)

    def forward(self, inputs):
        Y_pred = torch.sigmoid(self.linear(inputs))
        return Y_pred


class Operator:
    """
    Main operations are done via this class for simplicity
    """

    def __init__(self, inputs, outputs, weights, bias, optim, seed=1, split_train_test=None):
        super().__init__()
        self.optim = optim
        self.cost_function = None
        if split_train_test is not None:
            if not ((split_train_test > 0.0) and (split_train_test < 1.0)):
                raise ValueError("split_train_test input must be between 0 and 1 (exclusive), or None")
            else:
                train_samples = int(NS - math.floor((split_train_test * NS)))
                self.X_train = inputs[:train_samples][:]
                self.Y_train = outputs[:train_samples]
                self.X_test = inputs[train_samples::][:]
                self.Y_test = outputs[train_samples:]
        else:
            self.X_train = inputs
            self.Y_train = outputs
            self.X_test = None
            self.Y_test = None
        self.model = Model(seed)
        self.W = weights
        self.B = bias
        torch.set_default_dtype(torch.float64)

    def setup(self):
        if (self.X_train is None) or (self.Y_train is None):
            print("Run .generate() first!")
        else:
            self.cost_function = nn.MSELoss()
            if self.optim == 'SGD':
                self.optim = SGD(self.model.parameters(), lr=0.1)
            elif self.optim == 'Adam':
                self.optim = Adam(self.model.parameters(), lr=0.1)
            elif self.optim == 'FTE':
                self.optim = FTEstimator(self.model.parameters(), alpha=(ALPHA_1, ALPHA_2),
                                         c=(C_1, C_2), delT=DelT)

    def train_model(self):
        if (self.cost_function is None) or (self.optim is None):
            print("Run .setup() first!")
            save_cost_history = None
        else:
            print(self.model.linear.state_dict()['weight'])
            save_cost_history = np.zeros(EPOCHS)
            for epoch in range(EPOCHS):
                Y_pred = self.model(self.X_train)
                cost = self.cost_function(Y_pred, self.Y_train)
                cost.backward()
                self.optim.step()
                self.optim.zero_grad()
                # print(f'Epoch {epoch + 1}, Cost {cost.item()}')
                save_cost_history[epoch] = cost.item()
            print(f'Completed. Cost at epoch {EPOCHS} is {save_cost_history[-1]}')
            # W_pred = self.model.linear.state_dict()['weight']
            # B_pred = self.model.linear.state_dict()['bias']
            # print(f'Weight norm is {torch.linalg.norm(self.W - W_pred, ord=2)}')
            # print(f'Bias norm is {torch.linalg.norm(self.B - B_pred, ord=2)}')
        return save_cost_history

    def test_model(self):
        if self.X_test is None:
            print("You need to split train and test!")
            return None
        else:
            Y_test_pred = self.model(self.X_test)
            cost = self.cost_function(Y_test_pred, self.Y_test)
            print(f'Testing cost = {cost}')
            return cost.detach().numpy()

    def master(self):
        self.setup()
        cost_history = self.train_model()
        test = self.test_model()
        return cost_history, test


# Function Definitions
def generate_truth(seed=1, split_train_test=None):
    """
    Generates scaled random (normal) X, weight and bias, with which Y = sigmoid(X * W + B)
    :param seed: Integer to seed random with.
    :param split_train_test: If none, all X and Y are training. Float between 0 and 1 indicates
    the proportion of data that should be used for tarining and testing (thus returning (X_train,
    X_test), (Y_train, Y_test), W, B)
    :return: (X_train, X_test), (Y_train, Y_test), W, B. Return type is torch tensor. If no split,
    then X_train, Y_train, W, B
    """
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    inputs = torch.randn((NS, NI)) * SCALAR
    weights = torch.randn((NO, NI))
    biases = torch.randn(NO)
    outputs = F.linear(inputs, weights, bias=biases)
    outputs = F.sigmoid(outputs)
    if split_train_test is not None:
        if not ((split_train_test > 0.0) and (split_train_test < 1.0)):
            raise ValueError("generate_truth input must be between 0 and 1 (exclusive), or None")
        else:
            train_samples = int(NS - math.floor((split_train_test * NS)))
            inputs_train = inputs[:train_samples, :]
            outputs_train = outputs[:train_samples]
            inputs_test = inputs[train_samples:, :]
            outputs_test = outputs[train_samples:]
            return (inputs_train, inputs_test), (outputs_train, outputs_test), weights, biases
    else:
        inputs_train = inputs
        outputs_train = outputs
        return inputs_train, outputs_train, weights, biases


def save_cost(costs, optim, seed):
    file = FILE_START + "costs_" + optim + "_" + LAYER_TYPE + str(seed) + ".csv"
    np.savetxt(file, costs, delimiter='\n')


if __name__ == "__main__":
    X, Y, W, B = generate_truth()
    optimizers = ["SGD", "Adam", "FTE"]
    operators = list()
    test_Js = list()
    for i in range(len(optimizers)):
        operators.append(list())
        test_Js.append(list())
        for j in range(TESTS):
            operators[i].append(Operator(X, Y, W, B, optimizers[i], seed=j, split_train_test=0.2))
            Js, test_J = operators[i][j].master()
            save_cost(Js, optimizers[i], j)
            test_Js[i].append(test_J)
    np.savetxt(FILE_START + "test_costs_theta0_" + LAYER_TYPE + ".csv", np.asarray(test_Js), delimiter='\n')
