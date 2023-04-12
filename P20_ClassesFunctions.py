"""
Filename:   P20_ClassesFunctions.py
Author:     Turibius Rozario
Email:      s175@umbc.edu
Advisor:    Dr. Ankit Goel
Created:    March 27, 2023
Used for reusable classes and functions por Project 20.
"""

# Imports
import torch
from torch.optim import Optimizer
import torch.nn as nn
import torch.nn.functional as F
import math


# Classes
class FTEstimator(Optimizer):
    """
    Use global variables to set parameters or manually.
    :param c: The scalar coefficients for two terms in FT equation (default 1.5)
    :param delT: Discrete time step (default 0.1)
    :param alpha_1: Between 0 and 1 (default 0.5)
    :param alpha_2: Greater than 1 (default 2.5)
    :param params: A torch module's or the neural network's parameters, obtainable by
    my_model.parameters()
    """

    def __init__(self, params, alpha=(0.5, 2.5), c=(1.5, 1.5), delT=0.1):
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


class FxTS_momentum(Optimizer):
    """
    Implements FxTS optimizer with momentum. This has been written by Dr. Kunal Garg.
    Parameters:
    lr (float): learning rate. Default 1e-3
    betas (tuple of two floats): FxTS beta parameters (b1,b2). Default: (0.9,0.9)
    alphas (tuple of two floats): FxTS alpha parameters (a1,a2). Default: (2.1,1.9)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), alphas=(2.1, 1.9), momentum=0.0):
        if lr < 0.0:
            raise ValueError("Invalid learning rate:{}-should be >= 0.0".format(lr))
        if betas[0] < 0.0:
            raise ValueError("Invalid beta param:{}-should be >= 0.0".format(betas[0]))
        if betas[1] < 0.0:
            raise ValueError("Invalid beta param:{}-should be >= 0.0".format(betas[1]))
        if not alphas[0] > 2.0:
            raise ValueError("Invalid alpha param:{}-should be > 2.0".format(alphas[0]))
        if not 1.0 < alphas[1] < 2.0:
            raise ValueError("Invalid alpha param:{}-should be >1., <2.".format(alphas[1]))
        if not 0.0 <= momentum < 1.0:
            raise ValueError("Invalid momentum param:{}-should be >=0., <1.".format(momentum))

        defaults = dict(lr=lr, betas=betas, alphas=alphas, momentum=momentum)
        super(FxTS_momentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(FxTS_momentum, self).__setstate__(state)

    def step(self, closure=None):
        """ Performs a single optimization step.
        Arguments:
        closure (callable, optional): A closure that reevaluates the model
        and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            alpha1, alpha2 = group['alphas']
            lr = group['lr']
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['v'] = torch.zeros_like(p.data)

                v = state['v']
                state['step'] += 1
                v.mul_(momentum).add_((1 - momentum), grad)

                v_norm = v.norm()
                factor = beta1 / (v_norm ** ((alpha1 - 2) / (alpha1 - 1))) + \
                         beta2 / (v_norm ** ((alpha2 - 2) / (alpha2 - 1)))
                v.mul_(factor)

                if grad.norm() > (grad - v).norm():
                    h = 0.2 / (grad.norm() ** ((alpha1 - 2) / (alpha1 - 1))) + \
                        0.2 / (grad.norm() ** ((alpha2 - 2) / (alpha2 - 1)))
                else:
                    h = 1.

                p.data.add_(-h * lr, v)

        return loss


class SingleLayer(nn.Module):
    """
    Simple 1 layer NN with sigmoid or ReLU. Using GPU will only
    result in GPU output
    """

    def __init__(self, input_size, output_size, activation='sigmoid', device='cpu', seed=1):
        """
        Float 64 is used. Parameters:
        :param input_size: lx, or the number of inputs
        :param output_size: ly, or the number of neurons
        :param device: 'cpu' or 'gpu'
        :param seed: initial seed for random
        """
        super().__init__()
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(seed)
        # self.device = run_on_device(device)
        # self.linear = nn.Linear(input_size, output_size, device=self.device)
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation

    def forward(self, inputs):
        Y_pred = None
        if self.activation == 'sigmoid':
            Y_pred = torch.sigmoid(self.linear(inputs))
        return Y_pred


# Functions
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


def run_on_device(device='cpu'):
    """
    :param device: Forces a particular device to be used; 'gpu' or 'cpu'
    :return: The device to run PyTorch on
    """
    if torch.cuda.is_available() and device == 'gpu':
        print("Running on GPU")
        return torch.device('cuda')
    else:
        print("Running on CPU")
        return torch.device('cpu')


def SingleLayerTruth(input_size, output_size, sample_size=1, scalar=5, seed=1, split=None):
    """
    Generates scaled random (normal) X, weight and bias, with which Y = sigmoid(X * W + B)
    :param scalar: Data is normally generated. Scales the variance of the normal.
    :param sample_size: Number of samples to generate (total)
    :param output_size: ly
    :param input_size: lx
    :param seed: Integer to seed random with.
    :param split: If none, all X and Y are training. Float between 0 and 1 indicates
    the proportion of data that should be used for tarining and testing (thus returning (X_train,
    X_test), (Y_train, Y_test), W, B)
    :return: (X_train, X_test), (Y_train, Y_test), W, B. Return type is torch tensor. If no split,
    then X_train, Y_train, W, B
    """
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    inputs = torch.randn((sample_size, input_size)) * scalar
    weights = torch.randn((output_size, input_size))
    biases = torch.randn(output_size)
    outputs = F.linear(inputs, weights, bias=biases)
    outputs = F.sigmoid(outputs)
    if split is not None:
        return split_train_test(inputs, outputs, sample_size, split=split), weights, biases
    else:
        inputs_train = inputs
        outputs_train = outputs
        return inputs_train, outputs_train, weights, biases


def split_train_test(inputs, outputs, sample_size, split=None):
    """
    Splits data to training and testing. split_train_test=None if no data is to be used for
    :param inputs: lx, or the number of inputs
    :param outputs: ly, or the number of neurons/outputs
    :param sample_size: number of samples
    :param split: proportiono of data that should be dedicated for testing
    :return: input_training, output_training, input_testing, output_testing
    """
    if split is not None:
        if (split < 0.0) or (split >= 1):
            raise ValueError("split_train_test input must be between 0 and 1 (exclusive), or None")
        else:
            train_samples = int(sample_size - math.floor((split * sample_size)))
            inputs_train = inputs[:train_samples][:]
            outputs_train = outputs[:train_samples]
            inputs_test = inputs[train_samples::][:]
            outputs_test = outputs[train_samples:]
    else:
        inputs_train = inputs
        outputs_train = outputs
        inputs_test = None
        outputs_test = None
    return (inputs_train, outputs_train), (inputs_test, outputs_test)


def set_parameters(params, Prod_DelT=0.1, num_changes=5):
    """
    Returns a dictionary containing list of the parameters needed for FTE.
    alpha_1 and alpha_2 should start with the lowest value.
    DelT should start with the largest value.
    :param Prod_DelT: Multiplies previous DelT with Prod_DelT to create range.
    :param params: Default parameters, dictionary. Must include:
        alpha_1, alpha_2, c_1, c_2, DelT
    :param num_changes: Number of different values of a specific parameter
    Default: 5. The first value is the baseline.
    :return: Dictionary containing the parameters
    """
    a = math.floor(math.pow(0.9 / params['alpha_1'][0], 1 / (num_changes - 1)))
    for i in range(1, num_changes):
        params["alpha_1"].append(params["alpha_1"][i - 1] * a)
        params["alpha_2"].append(params["alpha_2"][i - 1] * 2)
        params["c_1"].append(params["c_1"][i - 1] + 0.5)
        params["c_2"].append(params["c_2"][i - 1] + 0.5)
        params["DelT"].append(params["DelT"][i - 1] * Prod_DelT)
    return params
