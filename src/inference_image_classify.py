import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.testbench import generate_testbench
from utils.model import CircuitNet
from utils.baseline import FeedForwardNet
from utils.utils import update_metric, parse_scheduler
import os
import time
from utils.utils import set_seed
from utils.topology import generate_topology
import sys

def verbose_print():
    print(f"### Test bench: {config['test_bench']}")
    print(f"### Network: {config['network']['net_struct']}")
    print(f"### Number of parameters: {num_param:.6f} (MiB)")
    print("### GPU: ", args.gpu)
    for i, circuitlayer in enumerate(model.circuit.layer_list):
        print(f"### Layer {i}: #edges = {circuitlayer.num_edge}, max_node={circuitlayer.max_node_index}")
    print("### Detailed configs: ", config)
    print('### Detailed model architecture: ', model)
    print("#" * 30)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--exp_name', type=str, default='mnist', help='stored under ./results/exp_name/')
    parser.add_argument('--gpu', type=int, nargs='+', default=[-1], help='gpu ids')  # Allow multiple GPUs
    parser.add_argument('--till_iter', type=float, default=float('inf'), help = 'till which iteration, default is inf to finish all the test data')
    parser.add_argument('--batch_size', type = int, default=-1, help='batch size for testing, default is -1 to use the batch size in the config file')

    # Parse the command-line arguments
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() and args.gpu[0] >= 0 else 'cpu')
    dp_flag = args.gpu[0] >= 0 and len(args.gpu) > 1
    exp_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), './results', args.exp_name)
    set_seed(42)

    if not os.path.exists(exp_dir):
        print(f"{exp_dir} does not exist, no testing performed, exit.")
        exit(0)

    # find the file with suffix .yaml in the exp_dir
    config_path = None
    for file in os.listdir(exp_dir):
        if file.endswith('.yaml'):
            config_path = os.path.join(exp_dir, file)
            break
    if config_path is None:
        print(f"Cannot find the config file in {exp_dir}!")
        exit(0)

    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    model = CircuitNet(
        circuit_topology=[generate_topology(net_struct) for net_struct in config['network']['net_struct']],
        sim_dict=config['simulation'],
        circuit_dict={'model': config['network']['model'],
                      'initialization': config['network']['initialization'],
                      'residual': config['network']['residual'],
                      'fill': config['network']['fill'],},
        encoder = None,
        projector=config['network']['projector'],
        use_augment=False,
        adjoint=config['network']['adjoint']).to(device)

    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    model_path = os.path.join(exp_dir, 'model.pth')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    verbose_print()

    model.prepare(args.gpu)
    if dp_flag:
        model = nn.DataParallel(model, device_ids=args.gpu)

    train_loader, test_loader, criterion, process_func = generate_testbench(config['test_bench'], (config['batch_size'] if args.batch_size < 0 else args.batch_size) * len(args.gpu),
                                                                            config['num_worker'])

    # testing loop
    test_start_time = time.time()
    model.eval()
    metric_value = {key: 0.0 for key in config['test']['metric']} if len(config['test']['metric']) > 0 else {}
    with torch.no_grad():
        running_loss, total_num = 0.0, 0
        for i, data in enumerate(test_loader):  # Loop through batches of data
            inputs, labels = process_func(data)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, _ = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss

            metric_value = update_metric(metric_value, outputs, labels)

            running_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]
            print(f"Finish {i}-th iteration, batch loss = {loss.item():.3e}")
            if i >= args.till_iter:
                break

    test_end_time = time.time()
    test_loss = running_loss / total_num
    print(f"### Testing finished in {(test_end_time - test_start_time) / 60:.3e} mins, w/ test loss = {test_loss:.3e}")
    if metric_value is not None:
        for key, value in metric_value.items():
            value /= float(total_num)
            metric_value[key] = value
            print(f"### Testing result --- {key}: {value:.6f}")
