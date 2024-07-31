import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Union, Optional, Any, Tuple
import sys
def generate_topology(topo_dict: Dict) -> torch.Tensor:
    connection = torch.tensor(getattr(sys.modules[__name__], topo_dict['name'])(**topo_dict['args'])[0])
    return connection


def _pregiven(source_edge_index, num_node_feature, repeat_across_layer=1, repeat_within_layer=1, repeat_to_gnd = 1):
    edge_index = source_edge_index.clone().detach().t() + 1
    edge_index = torch.cat([edge_index, edge_index[:, [1, 0]]], dim=0)
    assert len(edge_index.shape) == 2 and edge_index.shape[1] == 2
    edge_index = edge_index.repeat(repeat_within_layer, 1)
    max_node = edge_index.max().item()
    num_connect = edge_index.shape[0]
    for i in range(num_node_feature-1):
        edge_index = torch.cat([edge_index, edge_index[:num_connect,] + max_node], dim = 0)

    for j in range(1, 1 + max_node):
        for k in range(num_node_feature-1):
            for m in range(repeat_across_layer):
                connection = [j + max_node * k, j + max_node * (k+1)]
                edge_index = torch.cat([edge_index, torch.Tensor(connection).view(1,-1)], dim=0)

    for j in range(int(torch.max(edge_index).item())):
        for _ in range(repeat_to_gnd):
            edge_index = torch.cat([edge_index, torch.Tensor([[j,0],[0,j]])], dim = 0)
    cnt = torch.arange(edge_index.shape[0]).view(-1,1)
    edge_index = torch.cat([edge_index, cnt], dim = 1).long()
    return edge_index, cnt


def _neighbor_index(num_channel, width, height, neighbor_size, padding=False, repeat=1, sharing=False, parallel = 1):
    if sharing == False:
        wrk_tensor = torch.arange(1, 1 + num_channel * width * height).reshape(num_channel, width, height).unsqueeze(
            0).float()
        tmp = torch.nn.functional.unfold(wrk_tensor, kernel_size=neighbor_size,
                                         padding=0 if padding == False else (neighbor_size - 1), stride=1)
        result = []
        cnt = 0
        for i in range(tmp.shape[2]):
            for j in range(tmp.shape[1]):
                for k in range(tmp.shape[1]):
                    if j != k and int(tmp[0, j, i]) != 0 and int(tmp[0, k, i]) != 0:
                        for _ in range(repeat):
                            result.append([int(tmp[0, j, i]), int(tmp[0, k, i]), cnt])
                            cnt += 1
    else:
        wrk_tensor = torch.arange(1, 1 + num_channel * width * height).reshape(num_channel, width, height).unsqueeze(
            0).float()
        tmp = torch.nn.functional.unfold(wrk_tensor, kernel_size=neighbor_size,
                                         padding=0 if padding == False else (neighbor_size - 1), stride=1)
        result = []
        cnt = 0
        for i in range(tmp.shape[2]):
            inner_cnt = 0
            for j in range(tmp.shape[1]):
                for k in range(tmp.shape[1]):
                    if j == k:
                        continue
                    if int(tmp[0, j, i]) != 0 and int(tmp[0, k, i]) != 0:
                        for _ in range(repeat):
                            result.append([int(tmp[0, j, i]), int(tmp[0, k, i]), inner_cnt])
                    inner_cnt += 1
                    cnt = max(cnt, inner_cnt)

    # deal with parallel
    if parallel == 1:
        return result, cnt
    else:
        final_result1 = np.array(result)[:,:2]
        final_result2 = np.array(result)[:,2].reshape(-1,1)
        incre1, incre2 = np.max(final_result1), cnt
        for i in range(1, parallel):
            final_result1 = np.concatenate([final_result1, final_result1 + incre1 * i], axis = 0)
            final_result2 = np.concatenate([final_result2, final_result2 + incre2 * i], axis = 0)
        final_result = np.concatenate([final_result1, final_result2], axis = 1)
        final_cnt = np.max(final_result2) + 1 # or cnt + incre2 * (parallel - 1)
        return final_result.tolist(), final_cnt


def _fully_connect(num_node, include_gnd=True, repeat=[1, 1]):
    edge_matrix = []
    node_start = 0 if include_gnd else 1
    node_end = num_node if include_gnd else num_node + 1
    cnt = 0
    for i in range(node_start, node_end):
        for j in range(node_start, node_end):
            if i != j:
                for k1 in range(repeat[0]):
                    edge_matrix.append([i, j, cnt])
                    cnt += 1
                for k2 in range(repeat[1]):
                    edge_matrix.append([j, i, cnt])
                    cnt += 1
    return edge_matrix, cnt

def _simple_connect(num_node, include_gnd=True, repeat = [1,1]):
    edge_matrix = []
    node_start = 0 if include_gnd else 1
    node_end = num_node if include_gnd else num_node + 1
    cnt = 0
    for i in range(node_start, node_end):
        for k1 in range(repeat[0]):
            edge_matrix.append([i, i+1, cnt])
            cnt += 1
    return edge_matrix, cnt

def _mix_connect(num_channel, width, height, neighbor_size, fully_connect_node, repeat=[1, 0], padding = False, kernel_repeat = 1, sharing = False, parallel = 1):
    result, cnt = _neighbor_index(num_channel, width, height, neighbor_size, padding=padding, repeat=kernel_repeat, sharing=sharing, parallel = parallel)
    max_node = np.max(np.array(result)[:,:2])
    for node in range(1, max_node + 1):
        for k1 in range(repeat[0]):
            result.append([node, 0, cnt])
            cnt += 1
        for k2 in range(repeat[1]):
            result.append([0, node, cnt])
            cnt += 1
    for node in range(max_node + 1, max_node + 1 + fully_connect_node):
        for i in range(max_node + 1):
            for k1 in range(repeat[0]):
                result.append([i, node, cnt])
                cnt += 1
            for k2 in range(repeat[1]):
                result.append([node, i, cnt])
                cnt += 1
    return result, cnt
