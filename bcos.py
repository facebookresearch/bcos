import torch
from torch.optim import Optimizer

class BCOS(Optimizer):
    def __init__(self, params, lr, beta=0.9, eps=1e-6, 
                 weight_decay=0.1, mode='c', decouple_wd=True): 

        defaults = dict(lr=lr, beta=beta, eps=eps, wd=weight_decay) 
        super().__init__(params, defaults)

        if mode not in ['g', 'm', 'c']:
            raise ValueError(f"BCOS mode {mode} not supported")
        self.mode = mode
        self.decouple_wd = decouple_wd      # True for BCOSW

    def step(self, closure = None):

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            wd = group["wd"]

            for p in group["params"]:
                if not p.requires_grad:
                    continue 

                state = self.state[p]
                g = p.grad

                # initialize optimizer states for specific modes
                if self.mode in ['m', 'c'] and 'm' not in state:
                    state['m'] = g.detach().clone()
                if self.mode in ['g', 'm'] and 'v' not in state:
                    state['v'] = g.detach().square()

                # decoupled weight decay or absorb in gradient
                if self.decouple_wd:    # p := (1 - lr * wd) * p
                    p.data.mul_(1 - lr * wd)
                else:                   # g := g + wd * p
                    g.data.add_(p.data, alpha = wd)
                
                if self.mode in ['m', 'c']:
                    m = state['m']
                    if self.mode == 'c':    # conditional estimator
                        beta_v = 1 - (1 - beta)**2
                        g2 = g.detach().square()
                        v = beta_v * m.square() + (1 - beta_v) * g2
                    # update momentum
                    m.mul_(beta).add_(g.detach(), alpha=1 - beta) 
                    d = m
                else:
                    d = g.detach()
                
                if self.mode in ['g', 'm']:     # EMA estimator
                    v = state['v']
                    v.mul_(beta).add_(d.square(), alpha=1 - beta)

                # BCOS update: p := p - lr * (d / (sqrt(v) + eps))
                p.data.add_(d.div(v.sqrt() + eps), alpha= - lr)
