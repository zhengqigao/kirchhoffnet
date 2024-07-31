import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils.testbench import generate_testbench
from utils.baseline import FeedForwardNet
from utils.utils import update_metric
import os
import time
from utils.utils import set_seed, parse_scheduler
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--exp_name', type=str, default='debug', help='stored under ./results/exp_name/')
    parser.add_argument('--hidden_dim', type=int, nargs='+', help='hidden dimension')
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    parser.add_argument('--test_bench', type=str, default='toyregress', help='test bench')
    parser.add_argument('--metric', type = str, nargs='+', default=[], help='metric to be evaluated')

    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epoch', type=int, default=40, help='epoch')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--scheduler', type = str, default = 'none', help='use stepLR scheduler')

    parser.add_argument('--num_worker', type=int, default=4, help='number of workers')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # Parse the command-line arguments
    args = parser.parse_args()

    exp_dir = os.path.join('./results', args.exp_name + '_seed_' + str(args.seed))
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}") if args.gpu >= 0 else torch.device("cpu")

    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

    train_loader, test_loader, criterion, process_func = generate_testbench(args.test_bench, args.batch_size,
                                                                            args.num_worker)

    # Define the neural network, loss function, and optimizer
    model = FeedForwardNet(args.hidden_dim).to(device)
    num_param = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"### Test bench: {args.test_bench}")
    print(f"### Network: FFN {args.hidden_dim}")
    print(f"### Number of parameters: {num_param:.3e} (MiB)")
    print(model)


    optimizer = optim.AdamW(model.parameters(), args.lr)
    scheduler = parse_scheduler(args.scheduler, optimizer)

    # Training loop
    loss_list = []
    for epoch in range(args.epoch):  # Adjust the number of epochs as needed
        running_loss, total_num , epoch_start_time = 0.0, 0, time.time()
        for i, data in enumerate(train_loader):
            inputs, labels = process_func(data)
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the weights

            running_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]
        running_loss /= total_num
        if scheduler:
            scheduler.step()
        epoch_end_time = time.time()
        loss_list.append(running_loss)
        print(f"Epoch {epoch + 1}, Time: {(epoch_end_time - epoch_start_time)/60:.3e} mins, Loss: {running_loss:.3e}")

    # save the trained model to results/
    torch.save(model.state_dict(), os.path.join(exp_dir, 'model_ffn.pth'))
    plt.figure()
    plt.plot(np.arange(len(loss_list)), np.array(loss_list))
    plt.savefig(os.path.join(exp_dir, 'train_loss.png'))

    # testing loop
    test_start_time = time.time()
    model.eval()
    metric_value = {key: 0.0 for key in args.metric} if len(args.metric) > 0 else {}
    with torch.no_grad():
        running_loss, total_num = 0.0, 0
        for i, data in enumerate(test_loader):  # Loop through batches of data
            inputs, labels = process_func(data)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute the loss

            metric_value = update_metric(metric_value, outputs, labels)

            running_loss += loss.item() * inputs.shape[0]
            total_num += inputs.shape[0]
            nan_indices = torch.isnan(outputs)

    print(labels[:4])
    print(outputs[:4])
    test_end_time = time.time()
    test_loss = running_loss / total_num
    print(f"### Testing finished in {(test_end_time - test_start_time) / 60:.3e} mins, w/ test loss = {test_loss:.3e}")
    if metric_value is not None:
        for key, value in metric_value.items():
            value /= float(total_num)
            metric_value[key] = value
            print(f"### Testing result --- {key}: {value:.3e}")

    metric_value['structure'] = args.hidden_dim
    metric_value['test_loss'] = test_loss
    metric_value['num_param_mib'] = num_param
    metric_value['train_one_epoch_mins'] = (epoch_end_time - epoch_start_time) / 60
    metric_value['train_loss_list'] = loss_list
    metric_value['test_total_mins'] = (test_end_time - test_start_time) / 60

    # save the metric value
    with open(os.path.join(exp_dir, 'metric.yaml'), 'w') as file:
        yaml.dump(metric_value, file)