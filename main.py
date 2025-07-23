"""
To choose the configuration file to use, run the script with the command:
```
python main.py --config-name=[FILE_NAME].yaml
```
config.yaml is the default file.

You may override the parameters in the configuration file by
passing command line arguments. For example, 
```
python main.py user.wandb_id=your_wandb_id
```

Available parameters:
# - `args.auto_tune`: str, method for tuning the learning rate. Options: 'none', 'RGN', 'eb-criterion'.
    'none' - trying all layers one by one.
    'RGN' - using Relative Gradient Norm to tune the learning rate of each layer.
    'eb-criterion' - using the signal-to-noise ratio to tune the learning rate of each layer.
# - `args.train_mode`: str, the training mode. Options: 'train', 'test'.
# - `args.train_n`: int, number of training examples per class.
# - `args.log_dir` : str, Directory to save the results.
# - `args.epochs`: int, number of epochs to train the model.
# - `args.seed`: int, random seed for reproducibility.
# - `data.model_name`: str, name of the model to load for that dataset.
# - `data.dataset_name`: str, name of the dataset. Options: 'cifar10', 'imagenet-c'.
# - `data.corruption_types`: list of str, types of corruption to apply.
# - `data.severity`: int, severity level of the corruption (1-5).
# - `user.ckpt_dir`: str, directory containing the model checkpoints.
# - `user.root_dir`: str, root directory for datasets.
# - `user.wandb_id`: str, Weights & Biases user ID for logging.
# - `wandb.project` : str, Weights & Biases project name.
# - `wandb.exp_name` : str, name of the experiment for Weights & Biases.
# - `wandb.use`: bool, whether to use Weights & Biases for logging.
# - `wandb.exp_name`: str, name of the experiment for Weights & Biases.
"""

import csv
import itertools
import os
import time
from collections import defaultdict

import pathlib
from datetime import date
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import wandb
from omegaconf import DictConfig

import utils
from .dataset import get_loaders
from .surgical import configure_optimiser


@torch.no_grad()
def test(model, loader, criterion, cfg):
    model.eval()
    all_test_corrects = []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y)
        all_test_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss
    acc = torch.cat(all_test_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss


def train(
        model: torch.nn.Module,
        loader: DataLoader,
        criterion: callable,
        opt: torch.optim.optimizer,
        cfg: DictConfig
    ) -> tuple:
    """
    Train the classification model on the given DataLoader.

    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    criterion : callable
        Loss function to use for training.
    opt : torch.optim.Optimizer
        Optimizer to use for training.
    cfg : DictConfig
        Configuration object containing parameters for tuning.
        Currently a placeholder, but can be used for future configurations.

    Returns
    -------
    acc : float
        Training accuracy.
    total_loss : float
        Average training loss across batches
    """

    all_train_corrects = []
    total_loss = 0.0

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        loss = criterion(logits, y)
        all_train_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    acc = torch.cat(all_train_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()

    return acc, total_loss


@hydra.main(config_path="config", config_name="config")   # loads config/config.yaml
def main(cfg: DictConfig) -> None:
    """
    Main function to tune the model on corrupted datasets.

    The results will be saved to a CSV file in the specified log directory,
    whose name is created based on the current date and the dataset name.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing all the parameters for tuning.
    """
    cfg.args.log_dir = pathlib.Path.cwd()
    cfg.args.log_dir = os.path.join(
        cfg.args.log_dir, "results",
        cfg.data.dataset_name,
        date.today().strftime("%Y.%m.%d"),
        cfg.args.auto_tune
    )
    print(f"Log dir: {cfg.args.log_dir}")
    os.makedirs(cfg.args.log_dir, exist_ok=True)

    tune_options = [
        "first_two_block",
        "second_block",
        "third_block",
        "last",
        "all",
    ]

    if cfg.data.dataset_name == "imagenet-c":
        tune_options.append("fourth_block")
    if cfg.args.auto_tune != 'none':
        tune_options = ["all"]
    if cfg.args.epochs == 0: tune_options = ['all']

    corruption_types = cfg.data.corruption_types
    for corruption_type in corruption_types:
        cfg.wandb.exp_name = f"{cfg.data.dataset_name}_corruption{corruption_type}"
        if cfg.wandb.use:
            utils.setup_wandb(cfg)

        utils.set_seed_everywhere(cfg.args.seed)
        loaders = get_loaders(cfg, corruption_type, cfg.data.severity)

        for tune_option in tune_options:
            tune_metrics = defaultdict(list)
            lr_wd_grid = [
                (1e-1, 1e-4),
                (1e-2, 1e-4),
                (1e-3, 1e-4),
                (1e-4, 1e-4),
                (1e-5, 1e-4),
            ]

            # search across (default) learning rates and weight decays
            for lr, wd in lr_wd_grid:
                dataset_name = (
                    "imagenet"
                    if cfg.data.dataset_name == "imagenet-c"
                    else cfg.data.dataset_name
                )

                model = load_model(
                    cfg.data.model_name,
                    cfg.user.ckpt_dir,
                    dataset_name,
                    ThreatModel.corruptions,
                )

                model = model.cuda()

                # define tuning blocks
                if cfg.data.dataset_name == "cifar10":
                    tune_params_dict = {
                        "all": [model.parameters()],
                        "first_two_block": [
                            model.conv1.parameters(),
                            model.block1.parameters(),
                        ],
                        "second_block": [
                            model.block2.parameters(),
                        ],
                        "third_block": [
                            model.block3.parameters(),
                        ],
                        "last": [model.fc.parameters()],
                    }
                elif cfg.data.dataset_name == "imagenet-c":
                    tune_params_dict = {
                        "all": [model.model.parameters()],
                        "first_second": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                            model.model.layer2.parameters(),
                        ],
                        "first_two_block": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                        ],
                        "second_block": [
                            model.model.layer2.parameters(),
                        ],
                        "third_block": [
                            model.model.layer3.parameters(),
                        ],
                        "fourth_block": [
                            model.model.layer4.parameters(),
                        ],
                        "last": [model.model.fc.parameters()],
                    }

                params_list = list(itertools.chain(*tune_params_dict[tune_option]))

                N = sum(p.numel() for p in params_list if p.requires_grad)

                print(
                    f"\nTrain mode={cfg.args.train_mode}, "
                    f"using {cfg.args.train_n} corrupted images for training. "
                    "\n"
                    f"Re-training {tune_option} ({N} params). "
                    f"lr={lr}, wd={wd}. Corruption {corruption_type}"
                )

                criterion = F.cross_entropy

                for epoch in range(1, cfg.args.epochs + 1):
                    if cfg.args.train_mode == "train":
                        model.train()

                    # choose the learning rates based on the auto_tune option
                    if cfg.args.auto_tune != 'none': 
                        opt = configure_optimiser(
                            loaders['train'], lr, wd, model,
                            tuning_mode = cfg.args.auto_tune
                        )
                    else:
                        # Log rough fraction of parameters being tuned
                        no_weight = 0
                        for elt in model.named_parameters():
                            if elt['lr'] == 0.:
                                no_weight += elt['params'][0].flatten().shape[0]
                        total_params = sum(p.numel() for p in model.parameters())
                        tune_metrics['frac_params'].append((total_params-no_weight)/total_params)
                        print(f"Tuning {(total_params-no_weight)} out of {total_params} total")

                        # optimiser with equal learning rate on all parameters
                        opt = optim.Adam(params_list, lr=lr, weight_decay=wd)

                    acc_tr, loss_tr = train(
                        model, loaders["train"], criterion, opt, cfg
                    )

                    acc_te, loss_te = test(model, loaders["test"], criterion, cfg)
                    acc_val, loss_val = test(model, loaders["val"], criterion, cfg)
                    tune_metrics["acc_train"].append(acc_tr)
                    tune_metrics["acc_val"].append(acc_val)
                    tune_metrics["acc_te"].append(acc_te)

                    print(f"Epoch {epoch:2d} Train acc: {acc_tr:.4f}, Val acc: {acc_val:.4f}")

                    if cfg.wandb.use:
                        log_dict = {
                            f"{tune_option}/train/acc": acc_tr,
                            f"{tune_option}/train/loss": loss_tr,
                            f"{tune_option}/val/acc": acc_val,
                            f"{tune_option}/val/loss": loss_val,
                            f"{tune_option}/test/acc": acc_te,
                            f"{tune_option}/test/loss": loss_te,
                        }
                        wandb.log(log_dict)

                tune_metrics["lr_tested"].append(lr)
                tune_metrics["wd_tested"].append(wd)

            # Get test acc according to best val acc
            best_run_idx = np.argmax(np.array(tune_metrics["acc_val"]))
            best_testacc = tune_metrics["acc_te"][best_run_idx]
            best_lr_wd = best_run_idx // (cfg.args.epochs)

            print(
                f"Best epoch: {best_run_idx % (cfg.args.epochs)}, Test Acc: {best_testacc}"
            )

            data = {
                "corruption_type": corruption_type,
                "train_mode": cfg.args.train_mode,
                "tune_option": tune_option,
                "auto_tune": cfg.args.auto_tune,
                "train_n": cfg.args.train_n,
                "seed": cfg.args.seed,
                "lr": tune_metrics["lr_tested"][best_lr_wd],
                "wd": tune_metrics["wd_tested"][best_lr_wd],
                "val_acc": tune_metrics["acc_val"][best_run_idx],
                "best_testacc": best_testacc,
            }

            recorded = False
            fieldnames = data.keys()
            csv_file_name = f"{cfg.args.log_dir}/results_seed{cfg.args.seed}.csv"
            write_header = True if not os.path.exists(csv_file_name) else False
            while not recorded:
                try:
                    with open(csv_file_name, "a") as f:
                        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0.0)
                        if write_header:
                            csv_writer.writeheader()
                        csv_writer.writerow(data)
                    recorded = True
                except:
                    time.sleep(5)

if __name__ == "__main__":
    main()
