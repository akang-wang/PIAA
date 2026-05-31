import torch
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics.wrappers import ClasswiseWrapper
import random
import numpy as np

def compute_metrics(y_pred, y_true, class_names=None, device='cpu'):
    """
    Use torchmetrics to compute per-class average precision and mean average precision.

    Args:
        y_pred (Tensor or ndarray): shape (N, C), predicted probabilities.
        y_true (Tensor or ndarray): shape (N, C), binary labels (0 or 1).
        class_names (list of str, optional): class names for output readability.
        device (str): 'cpu' or 'cuda'

    Returns:
        dict: { 'ap': per-class AP array, 'map': mean AP scalar }
    """
    # Convert inputs to torch tensors
    if not isinstance(y_pred, torch.Tensor):
        y_pred = torch.tensor(y_pred)
    if not isinstance(y_true, torch.Tensor):
        y_true = torch.tensor(y_true,dtype=torch.int)
    else:
        y_true = y_true.to(torch.int)

    y_pred = y_pred.to(device).float()
    y_true = y_true.to(device)

    num_classes = y_pred.shape[1]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    # Define metric
    metric = ClasswiseWrapper(
        MultilabelAveragePrecision(num_labels=num_classes, average=None),
        labels=class_names
    ).to(device)

    # Update and compute
    metric.update(y_pred, y_true)
    ap_per_class = metric.compute()
    metric.reset()

    # Convert to list or numpy for easy handling
    ap_list = [ap.item() * 100 for ap in ap_per_class.values()]
    mAP = float(torch.mean(torch.tensor(ap_list)))

    return {
        'ap': ap_list,  # list of floats
        'map': mAP  # float
    }

