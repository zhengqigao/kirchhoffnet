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
    parser.add_argument('--exp_name', type=str, default='debug', help='stored under ./results/exp_name/')
    parser.add_argument('--config_path', type=str, default='./configs/config_twomoon.yaml',
                        help='configuration file path')
    parser.add_argument('--gpu', type=int, nargs='+', default=[-1], help='gpu ids')  # Allow multiple GPUs
    parser.add_argument('--seed', type=int, default=42, help='random seed')


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
                      'residual': None,
                      'fill': None},
        encoder = None,
        projector=None,
        use_augment=True,
        adjoint=config['network']['adjoint']).to(device)
    model.prepare(args.gpu)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    verbose_print()


    train_loader, test_loader, criterion, process_func = generate_testbench(config['test_bench'],
                                                                            config['batch_size'] * len(args.gpu),
                                                                            config['num_worker'])

    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['lr']))
    # sgd optimizer
    # optimizer = optim.SGD(model.parameters(), lr=float(config['train']['lr']), momentum=0.9)
    scheduler = parse_scheduler(config['train']['scheduler'], optimizer)

    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.zeros(model.circuit.layer_list[0].max_node_index,).to(device),
        covariance_matrix=torch.diag(torch.ones(model.circuit.layer_list[0].max_node_index,)).to(device)
    )

    if dp_flag:
        model = nn.DataParallel(model, device_ids=args.gpu)
    # Training loodp
    loss_list, nfe_list, nan_flag = [], [], False
    for epoch in range(1, 1 + config['train']['num_epoch']):  # Loop over the dataset multiple times
        running_loss, total_num, epoch_start_time = 0.0, 0, time.time()
        for i, data in enumerate(train_loader):
            inputs = process_func(data).to(device)
            optimizer.zero_grad()  # Zero the gradients

            (z_t0, logp_diff_t0), _ = model((inputs, torch.zeros(inputs.shape[0], 1).to(inputs.device)),
                                       reverse=True)  # Forward pass

            logp_x = p_z0.log_prob(z_t0) - logp_diff_t0.view(-1)
            loss = -logp_x.mean(0)

            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            running_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]

            print(f"{i}-th iteration ({(time.time() - epoch_start_time) / 60.0:.2f} mins), batch loss = {loss.item():.3e}")
            # print gradient
            # print(f"Example of gradients", model.module.circuit.layer_list[0].net_param.grad if dp_flag else model.circuit.layer_list[0].net_param.grad)

        running_loss /= total_num
        epoch_end_time = time.time()
        print(f"Finish {epoch}/{config['train']['num_epoch']} ({(epoch_end_time - epoch_start_time) / 60:.2f} mins), training loss = {running_loss:.3e}")

        loss_list.append(running_loss)
        nfe_list.append(list(map(lambda x: x / total_num, model.module.nfe if dp_flag else model.nfe)))
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
        z_t0 = p_z0.sample([config['visualization']['num_sample']]).to(device)
        logp_diff_t0 = torch.zeros(config['visualization']['num_sample'], 1).type(torch.float32).to(device)
        (z_t_samples, _), _ = model((z_t0, logp_diff_t0), reverse=False)

    if z_t_samples.shape[1] == 2:
        plt.figure()
        plt.scatter(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), s=1)
        plt.savefig(os.path.join(exp_dir, 'test_samples1.png'))
        plt.close()

        plt.figure()
        plt.hist2d(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), bins = 300, density = True, cmap = 'viridis')
        plt.colorbar()
        plt.savefig(os.path.join(exp_dir, 'test_samples2.png'))
        plt.close()
    elif z_t_samples.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=7, azim=-80)
        ax.scatter(z_t_samples[:, 0].cpu().numpy(), z_t_samples[:, 1].cpu().numpy(), z_t_samples[:, 2].cpu().numpy(), c=z_t_samples[:, 2].cpu().numpy(), cmap='viridis')
        plt.show()
    elif config['test_bench'] == 'genmnist':
        z_t_samples = z_t_samples.reshape(-1, 28, 28)
        plt.figure()
        plt.imshow(z_t_samples[0,...].cpu().numpy())
        plt.savefig(os.path.join(exp_dir, 'test_samples1.png'))
        plt.close()

        plt.figure()
        plt.imshow(z_t_samples[1,...].cpu().numpy())
        plt.savefig(os.path.join(exp_dir, 'test_samples2.png'))
        plt.close()
    else:
        raise ValueError(f"The dimension of the samples is {z_t_samples.shape[1]}, which is not supported.")

    # save the metric value
    metric_value = {}
    metric_value['train_loss_list'] = loss_list
    metric_value['nfe_list'] = nfe_list
    with open(os.path.join(exp_dir, 'metric.yaml'), 'w') as file:
        yaml.dump(metric_value, file)