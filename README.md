# BCOS

A stochastic approximation method (optimizer) with Block-Coordinate Optimal Stepsizes ([paper on arXiv](https://arxiv.org/abs/2507.08963)).

## Installation

Download the file `bcos.py` and `import BCOS from bcos`.

## Usage

Follow the [PyTorch optimizer instructions](https://docs.pytorch.org/docs/stable/optim.html) by first constructing a BCOS optimizer as follows: 

```
optimizer = BCOS(params, lr=0.001, beta=0.9, eps=1e-6, weight_decay=0.1, 
                 mode='c', decouple_wd=True, simple_cond=False)
```

### Parameters

* **params** (*iterable*): iterable of model parameters or iterable of dicts defining parameter groups.
* **lr** (*float*): the learning rate.
* **beta** (*float, optional*): smoothing factor in computing the momentum and [EMA](https://en.wikipedia.org/wiki/Exponential_smoothing) estimators (*default: 0.9*).
* **eps** (*float, optional*): small constant added to the denominator to improve numerical stability (*default: 1e-6*).
* **weight\_decay** (*float, optional*): weight decay regularization strength (*default: 0.1*).
* **mode** (*string, optional*): algorithmic mode of BCOS, must be one of the three choices (*default: 'c'*):
    * 'g': use gradient as search direction and EMA estimator for its 2nd moment (equivalent to [RMSprop](https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html)).
    * 'm': use momentum as search direction and EMA estimator for its 2nd moment (using same beta).
    * 'c': use momentum as search direction and conditional estimator for its 2nd moment.
* **decouple\_wd** (*bool, optional*): whether or not use [decoupled weight decay regularization](https://arxiv.org/abs/1711.05101) (*default: True*).
* **simple\_cond** (*bool, optional*): whether or not use simple alternative in BCOS-c variant (*default: False*)/

## License

BCOS is MIT licensed, as found in the LICENSE file.
