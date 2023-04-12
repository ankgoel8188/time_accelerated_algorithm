import torch
from torch.optim import Optimizer


class FxTS_momentum(Optimizer):
    """ Implements FxTS optimizer with momentum
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
