import random
import numpy as np
import torch


def set_seed(seed):
    # Set random seed for Python's random module
    random.seed(seed)

    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for PyTorch
    torch.manual_seed(seed)

def update_metric(metric_dict, prediction, target):
    # check if metric_dict is empty
    if metric_dict is None or len(metric_dict) == 0:
        return metric_dict
    if 'top1' in metric_dict.keys():
        assert len(prediction.shape) == 2 and len(target.shape) == 1 and prediction.shape[0] == target.shape[
            0], "shape mismatch"
        _, predicted = torch.max(prediction, 1)
        metric_dict['top1'] += (predicted == target).sum().item()
    if 'top5' in metric_dict.keys():
        # Ensure the shapes of prediction and target are compatible
        assert len(prediction.shape) == 2 and len(target.shape) == 1 and prediction.shape[0] == target.shape[
            0], "shape mismatch"
        # Calculate the top-5 accuracy: Check if the correct target is within the top 5 predictions
        _, predicted = torch.topk(prediction, 5, dim=1)
        metric_dict['top5'] += torch.sum(predicted == target.view(-1, 1)).item()

    return metric_dict

def parse_scheduler(keyword, optimizer):
    if keyword == None or keyword == 'none':
        return None
    elif keyword.startswith('steplr'):
        keywords = keyword.split('_')
        step_size, gamma = int(keywords[1]), float(keywords[2])
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif keyword.startswith('cosinelr'):
        keywords = keyword.split('_')
        T_max, eta_min = int(keywords[1]), float(keywords[2])
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    else:
        raise NotImplementedError(f"Unsupported scheduler: {keyword}")