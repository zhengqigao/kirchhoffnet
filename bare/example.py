import torch
import torch.nn as nn
import torch.nn.functional as F
from circuit import Circuit
from torchdiffeq import odeint_adjoint as odeint

device_cfg = {'use_diff': False, 'activation': 'relu'}

# 12 devices connecting from node (first row) to node (second row), e.g., first device from node 0 to node 4
# index = 0 (node 0) is reserved for ground node
# Thus, this topology has 7 nodes in total (1 ground node, 6 other nodes), and 12 devices in total
net_topo = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 5, 3, 2, 1, 0],
                         [4, 2, 3, 4, 5, 6, 1, 6, 4, 3, 2, 1]])

circuit = Circuit(net_topo, device_cfg)

batch_size, input_dim, num_t = 10, 6, 100

x = torch.randn(batch_size, input_dim)

# Evaluate the right-hand side of the ODE dx/dt = circuit(t,x) at a single time point t=0.5
output1 = circuit(0.5, x)

# solve the ODE dx/dt = circuit(t,x) at 100 uniform time points in [0.0,1.0] with x as initial values at t=0
output2 = odeint(circuit, x, t=torch.linspace(0, 1, num_t))

