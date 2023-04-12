"""
Filename:   P2010_230324_DiscreteFT.py
Author:     Turibius Rozario
Email:      s175@umbc.edu
Advisor:    Dr. Ankit Goel
Created:    March 24, 2023
Attempts to implement a discrete finite time algorithm on a non-linear problem.
The structure is Y_hat = sigmoid(X * W_hat + B_hat). Part of this stems from
P2009...DrGargExample.py. Later, it compares FTE with SGD and Adam.
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, Optimizer
import math
import numpy as np

# Global Variables
SEED = 1
EPOCHS = 500
NI = 1  # Number of Input features
NO = 1  # Number of Output features
NS = 100  # Number of Samples
SCALAR = 2
ALPHA_1 = 0.5
ALPHA_2 = 2.5
C_1 = 1.5
C_2 = 1.5
DelT = 0.1
FILE_START = "P2010/"
FILE_END = "1.csv"


# Classes
class FTestimator(Optimizer):
    """
    Use global variables to set parameters or manually.
    :param c: The scalar coefficients for two terms in FT equation
    :param delT: Discrete time step
    :param alpha_1: Between 0 and 1
    :param alpha_2: Greater than 1
    :param params: A torch module's or the neural network's parameters, obtainable by
    my_model.parameters()
    """
    def __init__(self, params, alpha=(ALPHA_1, ALPHA_2), c=(C_1, C_2), delT=DelT):
        if not ((alpha[0] > 0) and (alpha[0] < 1)):
            raise ValueError("alpha_1 must be in between (0, 1)")
        if not alpha[1] > 1:
            raise ValueError("alpha_2 must be > 1")
        if not (c[0] > 0 or c[1] > 0):
            raise ValueError("Scalar coefficients c must be positive scalar")
        if not (delT > 0):
            raise ValueError("Discrete time step must be positive scalar")
        defaults = dict(alpha=alpha, c=c, delT=delT)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        # grad = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            alpha_1, alpha_2 = group['alpha']
            c_1, c_2 = group['c']
            delT = group['delT']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                state['step'] += 1

                big_equation = torch.zeros_like(p.data)
                big_equation.sub_(generate_Xi(grad, alpha_1), alpha=c_1)
                big_equation.sub_(generate_Xi(grad, alpha_2), alpha=c_2)
                big_equation.mul_(delT)

                p.data.add_(big_equation)
        return loss


class Model(nn.Module):
    """
    Base model for our setup
    """
    def __init__(self):
        super().__init__()
        torch.manual_seed(SEED)
        self.linear = nn.Linear(NI, NO)

    def forward(self, inputs):
        Y_pred = torch.sigmoid(self.linear(inputs))
        return Y_pred


class Operator:
    """
    Main operations are done via this class for simplicity
    """
    def __init__(self, inputs, outputs, weights, bias, optim, split_train_test=None):
        super().__init__()
        self.optim = optim
        self.cost_function = None
        if split_train_test is not None:
            if not ((split_train_test > 0.0) and (split_train_test < 1.0)):
                raise ValueError("split_train_test input must be between 0 and 1 (exclusive), or None")
            else:
                train_samples = int(NS - math.floor((split_train_test * NS)))
                type(inputs)
                self.X_train = inputs[:train_samples][:]
                self.Y_train = outputs[:train_samples]
                self.X_test = inputs[train_samples::][:]
                self.Y_test = outputs[train_samples:]
                type(self.X_train)
        else:
            self.X_train = inputs
            self.Y_train = outputs
            self.X_test = None
            self.Y_test = None
        self.model = Model()
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
                self.optim = FTestimator(self.model.parameters())

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
            W_pred = self.model.linear.state_dict()['weight']
            B_pred = self.model.linear.state_dict()['bias']
            print(f'Weight norm is {torch.linalg.norm(self.W - W_pred, ord=2)}')
            print(f'Bias norm is {torch.linalg.norm(self.B - B_pred, ord=2)}')
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


# Functions, General
def generate_truth(split_train_test=None):
    """
    Generates scaled random (normal) X, weight and bias, with which Y = sigmoid(X * W + B)
    :param split_train_test: If none, all X and Y are training. Float between 0 and 1 indicates
    the proportion of data that should be used for tarining and testing (thus returning (X_train,
    X_test), (Y_train, Y_test), W, B)
    :return: (X_train, X_test), (Y_train, Y_test), W, B. Return type is torch tensor.
    """
    torch.manual_seed(SEED)
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


def generate_Xi(grad, alpha):
    """
    Computes
    Xi(Theta, alpha) = (dJdTheta) / ||dJdTheta||^(1 - alpha)
    for FT Estimator
    :param grad: Torch gradient
    :param alpha: alpha_1 or alpha_2 found in Finite Time Estimator
    :return: Xi
    """
    denominator = grad.norm() ** (1 - alpha)
    return grad / denominator


def save_cost(costs, optim):
    file = FILE_START + "costs_" + optim + "_" + FILE_END
    np.savetxt(file, costs, delimiter='\n')


if __name__ == "__main__":
    X, Y, W, B = generate_truth()
    optimizers = ["SGD", "Adam", "FTE"]
    operators = list()
    for optimizer in optimizers:
        operators.append(Operator(X, Y, W, B, optimizer, 0.2))
    test_Js = list()
    for i in range(len(optimizers)):
        Js, test_J = operators[i].master()
        save_cost(Js, optimizers[i])
        test_Js.append(test_J)
    np.savetxt(FILE_START + "test_costs_" + FILE_END, np.asarray(test_Js), delimiter='\n')
