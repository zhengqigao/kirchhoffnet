import sys
sys.path.append('../')
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from utils.model import CircuitNet
from utils.topology import generate_topology
import yaml
from utils.utils import set_seed



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--exp_name', type=str, default='potential1',
                        help='stored under ./results/exp_name/')
    parser.add_argument('--gpu', type=int, nargs='+', default=[-1], help='gpu ids')  # Allow multiple GPUs
    parser.add_argument('--seed', type=int, default=22, help='random seed')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() and args.gpu[0] >= 0 else 'cpu')

    exp_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), './results', args.exp_name)
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    # customization
    model = CircuitNet(
        circuit_topology=[generate_topology(net_struct) for net_struct in config['network']['net_struct']],
        sim_dict=config['simulation'],
        circuit_dict={'model': config['network']['model'],
                      'initialization': config['network']['initialization'],
                      'fill': None,
                      'residual': None, },
        encoder=None,
        projector=None,
        use_augment=True,
        adjoint=config['network']['adjoint']).to(device)

    # print number of parameters
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Number of parameters: {num_param:.6f} (MiB)")

    # load model state dict
    model.load_state_dict(torch.load(os.path.join(exp_dir, 'model.pth'), map_location=device))
    model.prepare(args.gpu)

    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.zeros(model.circuit.layer_list[0].max_node_index, ).to(device),
        covariance_matrix=torch.diag(torch.ones(model.circuit.layer_list[0].max_node_index, )).to(device)
    )

    with torch.no_grad():
        num_sample = 10 if config['test_bench'] == 'genmnist' else config['visualization']['num_sample']
        z_t0 = p_z0.sample([num_sample]).to(device)
        logp_diff_t0 = torch.zeros(num_sample, 1).type(torch.float32).to(device)
        (z_t_samples, logp_diff_t1), _ = model((z_t0, logp_diff_t0), reverse=False)
        logp = logp_diff_t1.view(-1) + p_z0.log_prob(z_t0)

    if config['test_bench'] == 'genmnist':
        z_t_samples = z_t_samples.reshape(-1, 28, 28)

        # map z_t_samples to [0, 1]
        z_t_samples = (z_t_samples - z_t_samples.min()) / (z_t_samples.max() - z_t_samples.min())
        z_t_samples = (z_t_samples > 0.5).type(torch.float32)

        for i in range(num_sample):
            plt.figure()
            plt.imshow(z_t_samples[i,...].cpu().numpy(),  cmap='gray')
            plt.show()

    elif config['test_bench'] in ['pinwheel', 'swissroll', 'circle','2spirals', '8gaussians', 'twomoon']:
        plt.figure()
        plt.scatter(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), s=1)

        plt.figure()
        plt.hist2d(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), bins=300, density=True,
                       cmap='viridis')
        plt.colorbar()
        plt.show()

    elif config['test_bench'] in ['potential2', 'potential1']:
        plt.figure()
        plt.hist2d(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), range=[[-4, 4], [-4, 4]], bins=300,
                   density=True, cmap='viridis')
        plt.colorbar()


        plt.figure()
        plt.plot(z_t_samples[:min(1000, z_t_samples.shape[0]), 0].cpu(),
                 z_t_samples[:min(1000, z_t_samples.shape[0]), 1].cpu(), ".")
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])


        plt.figure()
        plt.scatter(z_t_samples[:, 0].cpu(), z_t_samples[:, 1].cpu(), c=torch.exp(logp).cpu(), cmap='viridis',
                    marker='.', s=2)
        plt.colorbar()
        plt.xlim([-4, 4])
        plt.ylim([-4, 4])

        plt.show()

    else:
        raise NotImplementedError

