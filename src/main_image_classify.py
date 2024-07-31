import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.testbench import generate_testbench
from utils.model import CircuitNet
from utils.topology import generate_topology
from utils.utils import update_metric, parse_scheduler
import os
import time
from utils.utils import set_seed

def pre_experiment(args):
    device = torch.device(f"cuda:{args.gpu[0]}" if torch.cuda.is_available() and args.gpu[0] >= 0 else 'cpu')

    # generate the experiment directory
    exp_dir = os.path.join('./results', args.exp_name + '_seed_' + str(args.seed))

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
        print(f"{exp_dir} doesn't exist; created!")
    else:
        for file in os.listdir(exp_dir):
            os.remove(os.path.join(exp_dir, file))
        print(f"{exp_dir} already exists; files are all removed.")

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
    parser.add_argument('--exp_name', type=str, default='debug_dropout', help='stored under ./results/exp_name/')
    parser.add_argument('--config_path', type=str, help='configuration file path')
    parser.add_argument('--gpu', type=int, nargs='+', default=[-1], help='gpu ids')
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
                      'residual': config['network']['residual'],
                      'fill': config['network']['fill'],},
        encoder = None,
        projector=config['network']['projector'],
        use_augment=False,
        adjoint=config['network']['adjoint']).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    verbose_print()

    train_loader, test_loader, criterion, process_func = generate_testbench(config['test_bench'],
                                                                            config['batch_size'] * len(args.gpu),
                                                                            config['num_worker'])

    optimizer = optim.AdamW(model.parameters(), lr=float(config['train']['lr']))  # Use appropriate optimizer
    scheduler = parse_scheduler(config['train']['scheduler'], optimizer)

    # Training loop
    model.prepare(args.gpu)
    if dp_flag:
        model = nn.DataParallel(model, device_ids=args.gpu)

    loss_list, nfe_list, nan_flag = [], [], False
    for epoch in range(1, 1 + config['train']['num_epoch']):  # Loop over the dataset multiple times
        running_loss, total_num, epoch_start_time = 0.0, 0, time.time()
        for i, data in enumerate(train_loader):  # Loop through batches of data

            t1 = time.time()
            inputs, labels = process_func(data)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            outputs, _ = model(inputs)  # Forward pass

            t2 = time.time()

            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            t3 = time.time()

            running_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]

            # print(outputs[0, :10])

            print(f"{i}-th iteration in {epoch}-th epoch ({(time.time() - epoch_start_time) / 60.0:.3f} mins), batch loss = {loss.item():.3e}, forward time = {t2 - t1:.2f}, backward time = {t3 - t2:.2f}")

        running_loss /= total_num
        epoch_end_time = time.time()
        print(f"Finish {epoch}/{config['train']['num_epoch']} ({(epoch_end_time - epoch_start_time) / 60:.3f} mins), training loss = {running_loss:.3e}")

        loss_list.append(running_loss)
        nfe_list.append(list(map(lambda x: x/ total_num, model.module.nfe if dp_flag else model.nfe)))

        if epoch % config['train']['save_epoch'] == 0 or epoch == config['train']['num_epoch']:
            torch.save(model.module.state_dict() if dp_flag else model.state_dict(), os.path.join(exp_dir, 'model.pth'))
            plt.figure()
            plt.plot(np.arange(len(loss_list)), np.array(loss_list))
            plt.savefig(os.path.join(exp_dir, 'train_loss.png'))
            plt.close()
        if scheduler:
            scheduler.step()

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
            nan_indices = torch.isnan(outputs)

    test_end_time = time.time()
    test_loss = running_loss / total_num
    print(f"### Testing finished in {(test_end_time - test_start_time) / 60:.3f} mins, w/ test loss = {test_loss:.3e}")
    if metric_value is not None:
        for key, value in metric_value.items():
            value /= float(total_num)
            metric_value[key] = value
            print(f"### Testing result --- {key}: {value:.6f}")
    metric_value['test_loss'] = test_loss
    metric_value['num_param_mib'] = num_param
    metric_value['train_one_epoch_mins'] = (epoch_end_time - epoch_start_time) / 60
    metric_value['train_loss_list'] = loss_list
    metric_value['nfe_list'] = nfe_list
    metric_value['test_total_mins'] = (test_end_time - test_start_time) / 60

    # save the metric value
    with open(os.path.join(exp_dir, 'metric.yaml'), 'w') as file:
        yaml.dump(metric_value, file)
