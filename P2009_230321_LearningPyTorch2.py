import torch
import torch.nn as nn
import torch.nn.functional as F


# Classes
class LinearRegressor(nn.Module):
    def __init__(self, cpu_or_gpu):
        """
        Simple class for practicing PyTorch. This works with and without GPU.
        :param cpu_or_gpu: A torch.device()
        """
        self.optimizer = None
        self.cost_function = None
        self.X = None
        self.Y = None
        self.W = None
        self.B = None
        self.device = cpu_or_gpu
        torch.manual_seed(SEED)
        torch.set_default_dtype(torch.float64)
        self.linear = nn.Linear(NI, NO, device=self.device)
        super(LinearRegressor, self).__init__()

    def generate(self):
        self.X = torch.randn((NS, NI), device=self.device) * SCALAR
        self.W = torch.randn((NO, NI), device=self.device)
        self.B = torch.randn(NO, device=self.device)
        self.Y = F.linear(self.X, self.W, bias=self.B).to(self.device)

    def setup(self):
        if (self.X is None) or (self.Y is None):
            print("Run LinearRegressor.generate() first!")
        else:
            self.cost_function = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=ALPHA)

    def predict(self):
        if (self.cost_function is None) or (self.optimizer is None):
            print("Run LinearRegressor.setup() first!")
        else:
            for epoch in range(EPOCHS):
                Y_pred = self.linear(self.X)
                cost = self.cost_function(Y_pred, self.Y)
                cost.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f'Epoch {epoch + 1}, Cost {cost.item()}')
            W_pred = self.linear.state_dict()['weight']
            B_pred = self.linear.state_dict()['bias']
            print(f'Weight norm is {torch.linalg.norm(self.W - W_pred, ord=2)}')
            print(f'Bias norm is {torch.linalg.norm(self.B - B_pred, ord=2)}')

    def master(self):
        self.generate()
        self.setup()
        self.predict()


class NonLinearRegressor(nn.Module):
    def __init__(self, cpu_or_gpu):
        """
        Simple class for practicing PyTorch. This works with and without GPU.
        :param cpu_or_gpu: A torch.device()
        """
        super(NonLinearRegressor, self).__init__()
        self.optimizer = None
        self.cost_function = None
        self.X = None
        self.Y = None
        self.W = None
        self.B = None
        self.device = cpu_or_gpu
        torch.set_default_dtype(torch.float64)
        self.linear = nn.Linear(NI, NO, device=self.device)

    def nonlinear(self, X):
        return torch.sigmoid(self.linear(X))

    def generate(self):
        self.X = torch.randn((NS, NI), device=self.device) * SCALAR
        self.W = torch.randn((NO, NI), device=self.device)
        self.B = torch.randn(NO, device=self.device)
        self.Y = F.linear(self.X, self.W, bias=self.B).to(self.device)
        self.Y = F.sigmoid(self.Y)

    def setup(self):
        if (self.X is None) or (self.Y is None):
            print("Run LinearRegressor.generate() first!")
        else:
            self.cost_function = nn.MSELoss()
            self.optimizer = torch.optim.SGD(self.linear.parameters(), lr=ALPHA)

    def predict(self):
        if (self.cost_function is None) or (self.optimizer is None):
            print("Run LinearRegressor.setup() first!")
        else:
            for epoch in range(EPOCHS):
                Y_pred = self.nonlinear(self.X)
                cost = self.cost_function(Y_pred, self.Y)
                cost.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                print(f'Epoch {epoch + 1}, Cost {cost.item()}')
            W_pred = self.linear.state_dict()['weight']
            B_pred = self.linear.state_dict()['bias']
            print(f'Weight norm is {torch.linalg.norm(self.W - W_pred, ord=2)}')
            print(f'Bias norm is {torch.linalg.norm(self.B - B_pred, ord=2)}')

    def master(self):
        self.generate()
        self.setup()
        self.predict()


# Functions (General)
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


# Global Variables
SEED = 1
EPOCHS = 100
ALPHA = 1  # Learning rate
NI = 2  # Number of Input features
NO = 1  # Number of Output features
NS = 100  # Number of Samples
SCALAR = 2

if __name__ == "__main__":
    linearModel = LinearRegressor(run_on_device('gpu'))
    linearModel.master()
    nonLinearModel = NonLinearRegressor(run_on_device('gpu'))
    nonLinearModel.master()
