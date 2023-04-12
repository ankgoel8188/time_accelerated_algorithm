"""
Filename:   P2012_230330_FTEparameterEffects.py
Author:     Turibius Rozario
Email:      s175@umbc.edu
Advisor:    Dr. Ankit Goel
Created:    April 10, 2023
Adjusts alpha_1, alpha_2, c_1, c_2, and DelT to determine effect on
optimization rate using FTE. Similar to P2012, but with more data points
"""

# Imports
import torch
import torch.nn as nn
import math
import numpy as np
# Import the custom optimizer
from P20_ClassesFunctions import FTEstimator, SingleLayer, SingleLayerTruth, split_train_test, set_parameters, \
    run_on_device

# Global Variables
# TIME = 5
# FILE_START = "P2013/1_"  # Change the last value
# TEST_TYPE = "alpha_1"  # Change this
NI = [1, 10, 10]  # Number of Input features
NO = [1, 1, 5]  # Number of Output features
# NS = 50 * (NI * NO + NO)  # Number of Samples
SCALAR = 5
TESTS = ['alpha_1', 'alpha_2', 'c_1', 'c_2', 'DelT']


# Classes
class Operator:
    """
    Main operations are done via this class for simplicity
    """

    def __init__(self, inputs, outputs, params, device='cpu', seed=1, split=None):
        """
        Parameters:
        :param inputs: X (all)
        :param outputs: Y (all)
        :param params: a dictionary containing alpha_1, alpha_2, c_1, c_2, and DelT
        :param epochs_second: Number of epochs per second if DelT was 1
        :param device: 'gpu' or 'cpu'. Which device to run on
        :param seed: for generating random
        :param split: what proportion of data should be for testing (0 to 1)
        """
        super().__init__()
        self.optim = None
        self.cost_function = None
        self.params = {
            "alpha": (params["alpha_1"], params["alpha_2"]),
            "c": (params["c_1"], params["c_2"]),
            "delT": params["DelT"]
        }
        # Extract data size information
        sample_size = inputs.size(dim=0)
        input_size = inputs.size(dim=1)
        output_size = outputs.size(dim=1)
        # Imported function from P20_ClassesFunction:
        (self.X_train, self.Y_train), \
            (self.X_test, self.Y_test) = split_train_test(inputs, outputs, sample_size, split=split)
        # Generate model
        if device == 'gpu':
            self.device = run_on_device(device)
            self.X_train = self.X_train.to(self.device)
            self.Y_train = self.X_train.to(self.device)
            self.X_test = self.X_train.to(self.device)
        else:
            self.device = None
        self.model = SingleLayer(input_size, output_size, device=device, seed=seed).to(self.device)
        # self.epochs_second = epochs_second
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
            cost_history = None
        else:
            epoch = 1
            optimizing = True
            # How many epochs are there in a second
            # seconds_to_epochs = math.ceil(self.epochs_second / self.params["delT"])
            seconds_to_epochs = math.ceil(1 / self.params["delT"])
            cost_history = list()
            print(f'Every {seconds_to_epochs}th epoch is a second!')
            # Keep optimizing until cost does not improve significantly
            while optimizing:
                Y_pred = self.model(self.X_train)
                cost = self.cost_function(Y_pred, self.Y_train)
                cost.backward()
                self.optim.step()
                self.optim.zero_grad()
                epoch = epoch + 1
                # As a sanity check, print cost at every second
                if epoch % seconds_to_epochs == 0:
                    print(f'At {epoch}th epoch, cost is {cost.item()}')
                cost_history.append(cost.item())
                # Check if cost is improving significantly or not
                if epoch > math.ceil(seconds_to_epochs / 4):
                    if (cost_history[- math.ceil(seconds_to_epochs / 4)] - cost_history[-1]) < 0.00001:
                        optimizing = False
            print(f'Completed. Cost at epoch {epoch} and at {epoch / seconds_to_epochs}s is {cost_history[-1]}')
        return cost_history

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
def run_test(params, inputs, outputs):
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
                operator = Operator(inputs, outputs, test_params, split=0.2, device='cpu')
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


def save_cost(costs, test_num, file_name_first, file_name_next):
    file = file_name_first + file_name_next + "_" + str(test_num) + ".csv"
    np.savetxt(file, costs, delimiter='\n')


if __name__ == "__main__":
    file_start = "P2013/"
    parameters = {
        'alpha_1': [0.01],
        'alpha_2': [5],
        'c_1': [1],
        'c_2': [1],
        'DelT': [0.01]
    }
    parameters = set_parameters(parameters)
    for layer_structure in range(0, 3):
        lx = NI[layer_structure]
        ly = NO[layer_structure]
        ns = (lx * ly + ly) * 8
        # If split is not None, then X and Y will be tuples!
        X, Y, W, B = SingleLayerTruth(lx, ly, sample_size=ns, scalar=SCALAR)
        file_start_layer = file_start + str(layer_structure) + "_"
        for test_type in TESTS:
            print("Testing", test_type)
            Js, test_Js = run_test(specify_test_parameter(parameters, test_type), X, Y)
            for j in range(len(TESTS)):
                save_cost(Js[j], j, file_start_layer, test_type)
            np.savetxt(file_start_layer + "test_costs_" + test_type + ".csv",
                       np.asarray(test_Js), delimiter='\n')
    print(parameters)
