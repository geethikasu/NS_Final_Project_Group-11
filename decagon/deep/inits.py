import torch
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = torch.rand([input_dim, output_dim],minval=-init_range,maxval=init_range, dtype=torch.float32)
    return torch.Tensor(initial, name=name)


def zeros(input_dim, output_dim, name=None):
    """All zeros."""
    initial = torch.zeros((input_dim, output_dim), dtype=torch.float32)
    return torch.Tensor(initial, name=name)


def ones (input_dim, output_dim, name=None):
    """All zeros."""
    initial = torch.ones((input_dim, output_dim), dtype=torch.float32)
    return torch.Tensor(initial, name=name)
