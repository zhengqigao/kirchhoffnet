import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.testbench import generate_testbench
from utils.model import CircuitNet
import os
import time
from utils.utils import set_seed, parse_scheduler
from utils.topology import generate_topology

from utils.testbench import PotentialFunc


def pre_experiment(args):
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() and args.gpu[0] >= 0 else 'cpu')
    set_seed(args.seed)

    # generate the experiment directory
    exp_dir = os.path.join('./results', args.exp_name + '_seed_' + str(args.seed))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    # load the config file and save it to the experiment directory
    with open(args.config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    return device, config, exp_dir


def verbose_print():
    print(f"### Test bench: {config['test_bench']}")
    print(f"### Seed: {args.seed}")
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
    parser.add_argument('--exp_name', type=str, default='debug_local', help='stored under ./results/exp_name/')
    parser.add_argument('--config_path', type=str, default='./configs/config_potential1.yaml',
                        help='configuration file path')
    parser.add_argument('--gpu', type=int, nargs='+', default=[-1], help='gpu ids')  # Allow multiple GPUs
    parser.add_argument('--seed', type=int, default=2, help='random seed')

    # Parse the command-line arguments and prepare the experiment
    args = parser.parse_args()
    set_seed(args.seed)
    device, config, exp_dir = pre_experiment(args)
    dp_flag = args.gpu[0] >= 0 and len(args.gpu) > 1

    model = CircuitNet(
        circuit_topology=[generate_topology(net_struct) for net_struct in config['network']['net_struct']],
        sim_dict=config['simulation'],
        circuit_dict={'model': config['network']['model'],
                      'initialization': config['network']['initialization'],
                      'fill': None,
                      'residual': None, },
        encoder = None,
        projector=None,
        use_augment=True,
        adjoint=config['network']['adjoint']).to(device)
    model.prepare(args.gpu)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    verbose_print()

    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.zeros(model.circuit.layer_list[0].max_node_index, ).to(device),
        covariance_matrix=torch.diag(torch.ones(model.circuit.layer_list[0].max_node_index, )).to(device)
    )

    if dp_flag:
        model = nn.DataParallel(model, device_ids=args.gpu)

    potential_func = PotentialFunc(config['test_bench'])

    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['lr']))
    scheduler = parse_scheduler(config['train']['scheduler'], optimizer)

    # Training loop
    loss_list, nfe_list, nan_flag = [], [], False
    for epoch in range(1, 1 + config['train']['num_epoch']):  # Loop over the dataset multiple times
        epoch_start_time = time.time()

        inputs = p_z0.sample([config['batch_size'] * len(args.gpu)]).to(device)
        optimizer.zero_grad()  # Zero the gradients

        (z_t0, logp_diff_t0), _ = model((inputs, torch.zeros(inputs.shape[0], 1).to(inputs.device)),
                                   reverse=False)  # Forward pass

        logp_x = logp_diff_t0.view(-1) + potential_func(z_t0, 1)
        loss = logp_x.mean(0)

        loss.backward()  # Backpropagation

        optimizer.step()  # Update the weights

        print(
            f"{epoch}-th iteration ({(time.time() - epoch_start_time) / 60.0:.2f} mins), batch loss = {loss.item():.3e}")

        loss_list.append(loss.item())
        nfe_list.append(list(map(lambda x: x / inputs.shape[0], model.module.nfe if dp_flag else model.nfe)))

        if epoch % config['train']['save_epoch'] == 0 or epoch == config['train']['num_epoch']:
            torch.save(model.module.state_dict() if dp_flag else model.state_dict(), os.path.join(exp_dir, 'model.pth'))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), np.array(loss_list))
            plt.savefig(os.path.join(exp_dir, 'train_loss.png'))
            plt.close()

    # testing loop
    test_start_time = time.time()
    model.eval()

    with torch.no_grad():
        z = p_z0.sample([config['visualization']['num_sample']]).to(device)
        logp_diff_t0 = torch.zeros(config['visualization']['num_sample'], 1).type(torch.float32).to(device)
        (z_t_samples, logp_diff_t1), _ = model((z, logp_diff_t0), reverse=False)
        logp = logp_diff_t1.view(-1) + p_z0.log_prob(z)

    plt.figure()
    plt.hist2d(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), range=[[-4, 4], [-4, 4]], bins=300,
               density=True, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(exp_dir, 'test_potential1.png'))
    plt.close()

    plt.figure()
    plt.plot(z_t_samples[:min(1000, z_t_samples.shape[0]), 0].cpu(),
             z_t_samples[:min(1000, z_t_samples.shape[0]), 1].cpu(), ".")
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig(os.path.join(exp_dir, 'test_potential2.png'))
    plt.close()

    plt.figure()
    plt.scatter(z_t_samples[:, 0].cpu(), z_t_samples[:, 1].cpu(), c=torch.exp(logp).cpu(), cmap='viridis', marker='.', s = 2)
    plt.colorbar()
    plt.xlim([-4, 4])
    plt.ylim([-4, 4])
    plt.savefig(os.path.join(exp_dir, 'test_potential3.png'))
    plt.close()

    # save the metric value
    metric_value = {}
    metric_value['train_loss_list'] = loss_list
    metric_value['nfe_list'] = nfe_list
    with open(os.path.join(exp_dir, 'metric.yaml'), 'w') as file:
        yaml.dump(metric_value, file)
