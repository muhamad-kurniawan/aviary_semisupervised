import numpy as np
import torch
from torch import Tensor


def RobustL1Loss(output: Tensor, log_std: Tensor, target: Tensor) -> Tensor:
    """
    Robust L1 loss using a lorentzian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    loss = np.sqrt(2.0) * torch.abs(output - target) * torch.exp(-log_std) + log_std
    return torch.mean(loss)


def RobustL2Loss(output: Tensor, log_std: Tensor, target: Tensor) -> Tensor:
    """
    Robust L2 loss using a Gaussian prior. Allows for estimation
    of an aleatoric uncertainty.
    """
    # NOTE can we scale log_std by something sensible to improve the OOD behaviour?
    loss = 0.5 * torch.pow(output - target, 2.0) * torch.exp(-2.0 * log_std) + log_std
    return torch.mean(loss)