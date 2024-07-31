import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union, Dict, Any
from device import TwoTerminalDevice

__all__ = ['Circuit']


def _preprocess(device_cfg: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Preprocess the device configuration
    """
    if 'use_diff' not in device_cfg:
        device_cfg['use_diff'] = False

    if 'activation' not in device_cfg:
        device_cfg['act_func'] = nn.Identity()
    elif device_cfg['activation'] == 'relu':
        device_cfg['act_func'] = nn.ReLU()
    elif device_cfg['activation'] == 'tanh':
        device_cfg['act_func'] = nn.Tanh()
    elif device_cfg['activation'] == 'leakyrelu':
        device_cfg['act_func'] = nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation function: {device_cfg['activation']}")
    return device_cfg


class Circuit(nn.Module):
    def __init__(self, net_topo: Union[torch.Tensor, Tuple, List],
                 device_cfg: Dict[str, Any],
                 noise_std: Optional[float] = 0.0):
        super().__init__()

        assert len(net_topo) == 2 and len(net_topo[0]) == len(net_topo[1]), "Invalid topology"

        self.register_buffer('src_node', net_topo[0].to(torch.int64))
        self.register_buffer('des_node', net_topo[1].to(torch.int64))

        self.num_device = len(net_topo[0])
        self.device_cfg = _preprocess(device_cfg)
        self.model = TwoTerminalDevice(self.num_device, self.device_cfg)

        self.noise_std = noise_std

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        # x (aux_v) represents the node voltage, excluding (including) the ground node.
        aux_v = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
        state_i = self.model(t, aux_v[..., self.src_node], aux_v[..., self.des_node])

        # add dummy node for ground
        result = torch.cat((torch.zeros_like(x[..., :1]), torch.zeros_like(x)), dim=-1)

        # Subtract state_i from the source nodes and add it to the destination nodes
        result.scatter_add_(-1, self.src_node.expand_as(state_i), -state_i)
        result.scatter_add_(-1, self.des_node.expand_as(state_i), state_i)

        # remove the value associated with ground node, because its v=0 always.
        result = result[..., 1:]
        return result
