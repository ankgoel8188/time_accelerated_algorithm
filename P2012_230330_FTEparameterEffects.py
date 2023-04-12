"""
Filename:   P2012_230330_FTEparameterEffects.py
Author:     Turibius Rozario
Email:      s175@umbc.edu
Advisor:    Dr. Ankit Goel
Created:    March 30, 2023
Adjusts alpha_1, alpha_2, c_1, c_2, and DelT to determine effect on
optimization rate using FTE.
"""

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
# Import the custom optimizer
from P20_ClassesFunctions import FTEstimator

# Global Variables
EPOCHS = 1000
FILE_START = "P2012/3_"  # Change the last value
TEST_TYPE = "DelT"  # Change this
NI = 10  # Number of Input features
NO = 5  # Number of Output features
NS = 100  # Number of Samples
SCALAR = 5
ALPHA_1 = 0.5
ALPHA_2 = 2.5
C_1 = 2.5
C_2 = 2.5
DelT = 0.01
TESTS = 5


# Classes
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

    def __init__(self, inputs, outputs, weights, bias, params, seed=1, split_train_test=None):
        super().__init__()
        self.optim = None
        self.cost_function = None
        self.params = {
            "alpha": (params["alpha_1"], params["alpha_2"]),
            "c": (params["c_1"], params["c_2"]),
            "delT": params["DelT"]
        }
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
            self.optim = FTEstimator(self.model.parameters(), **self.params)

    def train_model(self):
        if (self.cost_function is None) or (self.optim is None):
            print("Run .setup() first!")
            save_cost_history = None
        else:
            save_cost_history = np.zeros(EPOCHS)
            for epoch in range(EPOCHS):
                Y_pred = self.model(self.X_train)
                cost = self.cost_function(Y_pred, self.Y_train)
                cost.backward()
                self.optim.step()
                self.optim.zero_grad()
                save_cost_history[epoch] = cost.item()
            print(f'Completed. Cost at epoch {EPOCHS} is {save_cost_history[-1]}')
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


# Functions
def set_parameters(num_changes=5):
    """
    Returns a dictionary containing list of the parameters needed for FTE.
    Requires global constant ALPHA_I, C_I, and DelT to be defined.
    :param num_changes: Number of different values of a specific parameter
    Default: 5. The first value is the baseline.
    :return: Dictionary containing the parameters
    """
    params = {
        "alpha_1": [ALPHA_1],
        "alpha_2": [ALPHA_2],
        "c_1": [C_1],
        "c_2": [C_2],
        "DelT": [DelT]
    }
    Del_alpha_1 = (0.95 - ALPHA_1) / (num_changes - 1)
    Del_alpha_2 = (1.05 - ALPHA_2) / (num_changes - 1)
    Del_c = 0.5
    Del_DelT = 4
    for i in range(1, num_changes):
        params["alpha_1"].append(params["alpha_1"][0] + Del_alpha_1 * i)
        params["alpha_2"].append(params["alpha_2"][0] + Del_alpha_2 * i)
        params["c_1"].append(params["c_1"][0] - Del_c * i)
        params["c_2"].append(params["c_2"][0] - Del_c * i)
        params["DelT"].append(params["DelT"][i-1] * Del_DelT)
    return params


def run_test(params):
    """
    Only 1 item in params should be a list!
    :param params:
    :return:
    """
    costs = list()
    test_costs = list()
    for unfixed_parameter in params:
        if type(params[unfixed_parameter]) == list:
            test_params = params.copy()
            for test_parameter in params[unfixed_parameter]:
                test_params[unfixed_parameter] = test_parameter
                operator = Operator(X, Y, W, B, test_params, split_train_test=0.2)
                cost, test_cost = operator.master()
                costs.append(cost)
                test_costs.append(test_cost)
    return costs, test_costs


def specify_test_parameter(params, test_param):
    """
    Takes in a dictionary of parameters, where each is a list. Transforms it
    (hard copy) such that only the test_param is retained as a list
    :param params: A dictionary, where each element is a list
    :param test_param: String, present in params key
    :return: A dictionary where only one value is a list
    """
    new_params = params.copy()
    for key in new_params:
        if key is not test_param:
            new_params[key] = new_params[key][0]
    return new_params


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


def save_cost(costs, test_num):
    file = FILE_START + TEST_TYPE + "_" + str(test_num) + ".csv"
    np.savetxt(file, costs, delimiter='\n')


if __name__ == "__main__":
    X, Y, W, B = generate_truth()
    parameters = set_parameters(TESTS)
    # print(parameters)
    Js, test_Js = run_test(specify_test_parameter(parameters, TEST_TYPE))
    for i in range(TESTS):
        save_cost(Js[i], i)
    np.savetxt(FILE_START + "test_costs_" + TEST_TYPE + ".csv",
               np.asarray(test_Js), delimiter='\n')
