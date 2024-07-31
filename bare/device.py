import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union, Dict, Any

__all__ = ['TwoTerminalDevice']


class TwoTerminalDevice(nn.Module):
    def __init__(self, num_device: int,
                 use_diff: Optional[bool] = False,
                 activation: Optional[nn.Module] = nn.ReLU, ):
        super().__init__()

        self.num_device = num_device
        self.use_diff = use_diff
        self.activation = activation

        self.param = nn.Parameter(torch.ones(2 if use_diff else 3, self.num_device))

    def forward(self, t: float, src: torch.Tensor, des: torch.Tensor) -> torch.Tensor:
        r"""
        Forward function of the device

        Args:
            t (float): time, placeholder at present.
            src (torch.Tensor): source node voltage, shape (..., num_device)
            des (torch.Tensor): destination node voltage, shape (..., num_device)

        Returns:
            torch.Tensor: device current, shape (..., num_device)
        """
        if self.use_diff:
            res = self.activation(self.param[0] * (src - des) + self.param[1])
        else:
            res = self.activation(self.param[0] * src + self.param[1] * des + self.param[2])
        return res
