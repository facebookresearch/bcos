# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.optim import Optimizer

class BCOS(Optimizer):
    def __init__(self, params, lr, beta=0.9, beta2=None, eps=1e-6, 
                 weight_decay=0.1, mode='c', decouple_wd=True): 

        defaults = dict(lr=lr, beta=beta, beta2=beta2, eps=eps, wd=weight_decay) 
        super().__init__(params, defaults)

        if mode not in ['g', 'm', 'c']:
            raise ValueError(f"BCOS mode {mode} not supported")
        self.mode = mode
        self.decouple_wd = decouple_wd      # True for BCOSW

    def step(self, closure = None):

        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            beta2 = group["beta2"]
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
                        betav = 1 - (1 - beta)**2 if beta2 is None else beta2
                        g2 = g.detach().square()
                        v = betav * m.square() + (1 - betav) * g2
                    # update momentum
                    m.mul_(beta).add_(g.detach(), alpha=1 - beta) 
                    d = m
                else:
                    d = g.detach()
                
                if self.mode in ['g', 'm']:     # EMA estimator
                    v = state['v']
                    betav = beta if beta2 is None else beta2
                    v.mul_(betav).add_(d.square(), alpha=1 - betav)

                # BCOS update: p := p - lr * (d / (sqrt(v) + eps))
                p.data.add_(d.div(v.sqrt() + eps), alpha= - lr)
