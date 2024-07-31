import torchdiffeq
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union, Optional, Any, Tuple, Callable
import numpy as np
import torch.nn.init as init
from .baseline import trace_jac, FeedForwardNet
import warnings
import matplotlib.pyplot as plt
from math import ceil
import time
import sys


class Device(nn.Module):
    def __init__(self, num_edge, negation, activation):
        super(Device, self).__init__()
        self.num_edge = num_edge
        if negation == True:
            self.param = nn.Parameter(torch.ones(2, self.num_edge))
        else:
            self.param = nn.Parameter(torch.ones(3, self.num_edge))
        self.negation = negation
        self.activation_function = getattr(F, activation) if activation is not None else lambda x: x
    def forward(self, x_src, x_des):
        if self.negation:
            res = self.activation_function(self.param[1] * (x_src - x_des) + self.param[0])
        else:
            res = self.activation_function(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res

class ShiftRelu1(nn.Module):
    def __init__(self, num_edge):
        super(ShiftRelu1, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(2, self.num_edge))

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = self.param[1] * F.relu(x - self.param[0])
        return res

class ShiftLeakyRelu1(nn.Module):
    def __init__(self, num_edge):
        super(ShiftLeakyRelu1, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(2, self.num_edge))

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = self.param[1] * F.leaky_relu(x - self.param[0])
        return res


class ShiftRelu2(nn.Module):
    def __init__(self, num_edge):
        super(ShiftRelu2, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(3, self.num_edge))

    def forward(self, x_src, x_des):
        res = F.relu(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res

class ShiftLeakyRelu2(nn.Module):
    def __init__(self, num_edge):
        super(ShiftLeakyRelu2, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(3, self.num_edge))

    def forward(self, x_src, x_des):
        res = F.leaky_relu(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res

class ShiftTanh1(nn.Module):
    def __init__(self, num_edge):
        super(ShiftTanh1, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(3, self.num_edge))

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = F.tanh(x * self.param[1] + self.param[0])
        return res

class ShiftTanh2(nn.Module):
    def __init__(self, num_edge):
        super(ShiftTanh2, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(3, self.num_edge))

    def forward(self, x_src, x_des):
        res = F.tanh(x_src * self.param[0] + x_des * self.param[1] + self.param[2])
        return res

class Conductance(nn.Module):
    def __init__(self, num_edge):
        super(Conductance, self).__init__()
        self.num_edge = num_edge
        self.param = nn.Parameter(torch.ones(3, self.num_edge))

    def forward(self, x_src, x_des):
        x = x_src - x_des
        res = self.param[0] * x
        return res


def _preprocess_net_topo(net_topo: Union[List, Tuple, np.ndarray, torch.Tensor]) -> Tuple[
    torch.Tensor, torch.Tensor, int, int]:
    if isinstance(net_topo, (list, tuple, np.ndarray)):
        net_topo = torch.Tensor(net_topo).to(torch.int64)
    elif isinstance(net_topo, torch.Tensor):
        net_topo = net_topo.to(torch.int64)
    else:
        raise ValueError(f"Unsupported type of net_topo: {type(net_topo)}")

    if not (len(net_topo.shape) == 2 and net_topo.shape[1] >= 2):
        raise ValueError(f"Unsupported shape of net_topo, expected it to be (N, >=2), got {net_topo.shape}")

    return (net_topo[:, 0], net_topo[:, 1], int(torch.max(net_topo[:, :2]).item()), net_topo.shape[0])

def _preprocess_sim_dict(sim_dict: Dict[str, Any]) -> Dict:
    sim_dict['t_end'] = [float(val) for val in sim_dict['t_end']]
    sim_dict['tol'] = float(sim_dict['tol'])
    sim_dict['min_step'] = float(sim_dict['min_step'])
    sim_dict['first_step'] = float(sim_dict['first_step'])
    sim_dict['step_size'] = float(sim_dict['step_size'])
    return sim_dict

def _divide_time_bins(anchors, time_grids):
    anchors = anchors
    if time_grids[-1] > anchors[-1]:
        raise ValueError(f"The last time grid {time_grids[-1]} is larger than the last anchor {anchors[-1]}")
    if time_grids[0] < anchors[0]:
       raise ValueError(f"The first time grid {time_grids[0]} is smaller than the first anchor {anchors[0]}")

    result = [[] for _ in range(len(anchors)-1)]
    for i in range(len(anchors) - 1):
        left, right = anchors[i], anchors[i+1]
        result[i].append(left)
        for j in range(len(time_grids)):
            if time_grids[j] > left and time_grids[j] < right:
                result[i].append(time_grids[j])
        result[i].append(right)
    return result

def _init_model_param(keyword, model):
    if keyword == 'uniform':
        init_function = init.uniform_
    elif keyword == 'zeros':
        init_function = init.zeros_
    elif keyword == 'ones':
        init_function = init.ones_
    elif keyword == 'xavier':
        init_function = init.xavier_normal_
    elif keyword == 'gauss':
        init_function = lambda x: init.normal_(x, mean = 0.0, std=0.01)
    elif keyword == 'kaiming':
        init_function = init.kaiming_normal_
    else:
        raise NotImplementedError(f"Unsupported initialization: {keyword}")

    if isinstance(model, nn.ParameterList):
        for param in model:
            init_function(param)
    elif isinstance(model, nn.Parameter):
        init_function(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")


class CircuitLayer(nn.Module):
    def __init__(self, net_topo, net_dict):
        super(CircuitLayer, self).__init__()
        self.src_node, self.des_node, self.max_node_index, self.num_edge = _preprocess_net_topo(net_topo)
        self.model = getattr(sys.modules[__name__], net_dict['model']['name'])(self.num_edge, **net_dict['model']['args'])
        self.nfe = torch.tensor(0.0)

        _init_model_param(net_dict['initialization'], self.model.param)

    def prepare(self, device: List[int]) -> None:
        # device = [-1] (CPU) or a list with elements all >=0 , e.g., [0,1,2,]
        if len(device) == 1 and device[0] == -1:
            self.src_indices_list = [self.src_node]
            self.des_indices_list = [self.des_node]
        elif len(device) >= 1 and min(device) >= 0:
            self.src_indices_list = [self.src_node.to(device_index) if device_index in device else None for device_index
                                     in range(max(device) + 1)]
            self.des_indices_list = [self.des_node.to(device_index) if device_index in device else None for device_index
                                     in range(max(device) + 1)]
        else:
            raise ValueError(f"Unsupported device: {device}")

    def forward(self, t, x):

        self.nfe += torch.tensor(1.0)

        # Calculate the RHS of the ODE. Circuit ODE: \dot{v} = f(v). The time variable t won't be used; just placeholder.
        src_node, des_node = self.src_indices_list[x.get_device()], self.des_indices_list[x.get_device()]

        aux_v = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
        state_i = self.model(aux_v[..., src_node], aux_v[..., des_node])

        # add dummy node for ground
        result = torch.cat((torch.zeros_like(x[..., :1]), torch.zeros_like(x)), dim=-1)

        # Subtract state_i from the source nodes and add it to the destination nodes
        result.scatter_add_(-1, src_node.expand_as(state_i), -state_i)
        result.scatter_add_(-1, des_node.expand_as(state_i), state_i)

        return result[..., 1:]

class AugCircuitLayer(CircuitLayer):
    def __init__(self, net_struct, net_dict):
        super(AugCircuitLayer, self).__init__(net_struct, net_dict)

    def forward(self, t, states):
        self.nfe += torch.tensor(1.0)
        x, logp_x = states[0], states[1]
        src_node, des_node = self.src_indices_list[x.get_device()], self.des_indices_list[x.get_device()]

        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            # Calculate the RHS of the ODE. Circuit ODE: \dot{v} = f(v). The time variable t won't be used; just placeholder.
            aux_v = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)
            state_i = self.model(aux_v[..., src_node], aux_v[..., des_node])

            # add dummy node for ground
            result = torch.cat((torch.zeros_like(x[..., :1]), torch.zeros_like(x)), dim=-1)

            # Subtract state_i from the source nodes and add it to the destination nodes
            result.scatter_add_(-1, src_node.expand_as(state_i), -state_i)
            result.scatter_add_(-1, des_node.expand_as(state_i), state_i)

            dx_dt = result[..., 1:]
            dlogp_x_dt = -trace_jac(dx_dt, x).view(x.shape[0], 1)

            return (dx_dt, dlogp_x_dt)


class CircuitBlock(nn.Module):
    def __init__(self, layer_list, sim_dict, residual, odeint, fill):
        super(CircuitBlock, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.sim_dict = _preprocess_sim_dict(sim_dict)
        self.residual = residual
        self.odeint = odeint
        self.fill = fill
        self.set_integration_time(self.sim_dict['t_end'])
        print(self.integration_time)
    def set_integration_time(self, t: List) -> None:
        self.integration_time = _divide_time_bins(self.sim_dict['t_end'], t)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], reverse=False, return_middle=False) -> Tuple[torch.Tensor, List]:
        index = range(0, len(self.layer_list)) if not reverse else range(len(self.layer_list) - 1, -1, -1)
        middle = []
        for i in index:
            integration_time = torch.Tensor(self.integration_time[i]).to(x[0].device if isinstance(x, tuple) else x.device)
            integration_time = integration_time - integration_time[0]
            integration_time = integration_time.flip(0) if reverse else integration_time

            # Remedies for dimension inconsistency, applied without any warnings.
            if x.shape[-1] < self.layer_list[i].max_node_index:
                warnings.warn(f"index = {i} dimension mismatch, received dim = {x.shape[-1]}, current layer max_node_index = {self.layer_list[i].max_node_index}")
                if self.fill == 'zeros' or self.fill == 'zero':
                    x = torch.cat((x, torch.zeros(x.shape[0], self.layer_list[i].max_node_index - x.shape[1]).to(x.device)), dim=-1)
                elif self.fill == 'repeat':
                    x = x.repeat(1, ceil(self.layer_list[i].max_node_index / x.shape[1]))[:, :self.layer_list[i].max_node_index]
                else:
                    raise NotImplementedError(f"Unsupported fill method: {self.fill}")
            elif x.shape[-1] > self.layer_list[i].max_node_index:
                warnings.warn(f"index = {i} dimension mismatch, received dim = {x.shape[-1]}, current layer max_node_index = {self.layer_list[i].max_node_index}")
                x = x[:, :self.layer_list[i].max_node_index]

            out = self.odeint(self.layer_list[i], x, integration_time, rtol=self.sim_dict['tol'],
                              method=self.sim_dict['method'],
                              atol=self.sim_dict['tol'],
                              options={'first_step': self.sim_dict['first_step'],
                                       'min_step': self.sim_dict['min_step'],
                                       'step_size': self.sim_dict['step_size']
                                       })
            x = out[-1]

            if return_middle:
                middle.append(out)

            if self.residual is not None:
                if i == self.residual[0]:
                    residual_store = out[-1]
                if i == self.residual[1]:
                    x = x + residual_store

        return (x, middle)

class AugCircuitBlock(nn.Module):
    def __init__(self, layer_list: List[CircuitLayer], sim_dict: Dict[str, Any], residual, odeint: Callable, fill: Optional[str]= 'zero'):
        super(AugCircuitBlock, self).__init__()
        self.layer_list = nn.ModuleList(layer_list)
        self.sim_dict = _preprocess_sim_dict(sim_dict)
        self.residual = residual
        self.odeint = odeint
        self.fill = fill
        self.set_integration_time(self.sim_dict['t_end'])

    def set_integration_time(self, t: List) -> None:
        self.integration_time = _divide_time_bins(self.sim_dict['t_end'], t)
        print(self.integration_time)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor], reverse, return_middle) -> Tuple[torch.Tensor, List]:
        index = range(0, len(self.layer_list)) if not reverse else range(len(self.layer_list) - 1, -1, -1)
        middle = []
        for i in index:
            integration_time = torch.Tensor(self.integration_time[i]).to(x[0].device if isinstance(x, tuple) else x.device)
            integration_time = integration_time - integration_time[0]
            integration_time = integration_time.flip(0) if reverse else integration_time

            # Careful: Dimension inconsistency
            if x[0].shape[1] != self.layer_list[i].max_node_index:
                warnings.warn(f"Dimension mismatch, previous layer dim = {x[0].shape[1]}, current layer max_node_index = {self.layer_list[i].max_node_index}")
                if i == 0 and x[0].shape[1] < self.layer_list[i].max_node_index:
                    warnings.warn(f"This is at the input level, we will do repeat padding.")
                    x = (x[0].repeat(1, ceil(self.layer_list[i].max_node_index / x[0].shape[1]))[:, :self.layer_list[i].max_node_index], x[1])

            out = self.odeint(self.layer_list[i], x, integration_time, rtol=self.sim_dict['tol'],
                              method=self.sim_dict['method'],
                              atol=self.sim_dict['tol'],
                              options={'first_step': self.sim_dict['first_step'],
                                       'min_step': self.sim_dict['min_step'],
                                       'step_size': self.sim_dict['step_size']
                                       })
            x = tuple(ele[-1] for ele in out)
            if return_middle:
                middle.append(out)

        return (x, middle)
