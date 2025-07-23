from torch.utils.data import DataLoader
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch
from collections import defaultdict
from typing import List, Tuple
import itertools
import numpy as np


def get_lr_weights(
        model: nn.Module,
        loader: DataLoader,
        tuning_mode: str
    ) -> dict:
    """
    Get the weights for each layer in the model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to analyze.
    loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    cfg : DictConfig
        Configuration object containing parameters for tuning.
    tuning_mode : str
        The tuning mode to use, either 'SNR' or 'RGN'.
        SNR: Signal-to-Noise Ratio, computed by 
        (grad^2 / var(grad)).
        RGN: Relative Gradient Norm, computed by
        (norm(grad) / norm(param)).

    Returns
    -------
    dict
        A dictionary with layer names as keys and
        their relative gradient norms as values.
    """
    layer_names = [
        n for n, _ in model.named_parameters() if "bn" not in n
    ] 
    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    partial_loader = itertools.islice(loader, 5)
    xent_grads = []

    for x, y in partial_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

    def _get_grad_norms(model, grads):
        _metrics = defaultdict(list)
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            if tuning_mode == 'SNR':
                tmp = (grad * grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
                _metrics[name] = tmp.mean().item()
            elif tuning_mode == 'RGN':
                _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()
            else:
                raise ValueError(
                    f"Unknown tuning mode: {tuning_mode}. "
                    "Use 'SNR' or 'RGN'."
                )

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = _get_grad_norms(model, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)

    return average_metrics


def configure_optimiser(
        train_loader: DataLoader,
        lr: float,
        wd: float,
        model: nn.Module,
        tuning_mode='SNR',
        binary_lr: bool=True,
        binary_threshold: float=0.95
    ) -> Tuple[List[float], optim.Optimizer]:
    """
    Compute the learning rate weights for each layer
    using the Relative Gradient Norm (RGN) method,
    and configure an Adam optimiser with these weights.

    Parameters
    ----------
    train_loader : DataLoader
        The DataLoader for the training dataset.
    lr : float
        Default learning rate for the optimizer.
    wd : float
        Weight decay for the optimizer.
    model : torch.nn.Module
        The model to optimize.
    tuning_mode : str, optional
        The tuning mode to use, either 'SNR' or 'RGN'.
        Defaults to 'SNR'.
    binary_lr : bool, optional
        If True, the learning rates will be binary (0 or 1)
        Defaults to True.
    binary_threshold : float, optional
        The threshold for binary learning rates.
        If the computed weight is below this threshold,
        it will be set to 0, otherwise to 1.
        Defaults to 0.95.

    Returns
    -------
    opt : torch.optim.Optimizer
        The optimizer configured with the computed learning rates.
    """

    layer_weights = [
        0 for layer, _ in model.named_parameters()
        if 'bn' not in layer
    ]

    weights = get_lr_weights(
        model, train_loader, tuning_mode=tuning_mode
    )

    if binary_lr:
        for k, v in weights.items(): 
            weights[k] = 0.0 if v < binary_threshold else 1.0

    # normalise the weights
    max_weight = max(weights.values())
    for k, v in weights.items(): 
        weights[k] = v / max_weight

    # sum the weights for each layer
    layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
    params = defaultdict()
    for n, p in model.named_parameters():
        if "bn" not in n:
            params[n] = p

    # configure the optimiser with the weights
    params_weights = []
    for param, weight in weights.items():
        params_weights.append({"params": params[param], "lr": weight * lr})

    opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)

    return opt
