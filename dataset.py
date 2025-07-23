from pathlib import Path
from robustbench.data import load_cifar10c
from torch.utils.data import Subset, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
from omegaconf import DictConfig
from typing import Tuple



def get_loaders(
        cfg: DictConfig, corruption_type: str, severity: int
    ) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Retrieve train, validation, and test datasets for CIFAR-10-C or ImageNet-C.

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing dataset parameters.
        - data.dataset_name : str
            Name of the dataset ('cifar10' or 'imagenet-c').
        - args.train_n : int
            Number of training examples per class.
        - user.root_dir : str
            Root directory for dataset storage.
        - args.batch_size : int
            Batch size for data loaders.
        - args.num_workers : int
            Number of workers for data loading.
    corruption_type : str
        Type of corruption to apply
        (e.g., 'gaussian_noise', 'shot_noise').
    severity : int
        Severity level of the corruption (1-5).

    Returns
    -------
    Tuple[TensorDataset, TensorDataset, TensorDataset]
        - Train dataset
        - Validation dataset
        - Test dataset
    """
    if cfg.data.dataset_name == "cifar10":
        x_corr, y_corr = load_cifar10c(
            10000, severity, cfg.user.root_dir, False, [corruption_type]
        )
        assert cfg.args.train_n <= 9000
        labels = {}
        num_classes = int(max(y_corr)) + 1
        for i in range(num_classes):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i]
        num_ex = cfg.args.train_n // num_classes
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+100])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        
        tr_dataset = TensorDataset(x_corr[tr_idxs], y_corr[tr_idxs])
        val_dataset = TensorDataset(x_corr[val_idxs], y_corr[val_idxs])
        te_dataset = TensorDataset(x_corr[test_idxs], y_corr[test_idxs])
    
    elif cfg.data.dataset_name == "imagenet-c":
        data_root = Path(cfg.user.root_dir)
        image_dir = data_root / "ImageNet-C" / corruption_type / str(severity)
        dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
        assert cfg.args.train_n <= 20000
        labels = {}
        y_corr = dataset.targets
        for i in range(max(y_corr)+1):
            labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
        num_ex = cfg.args.train_n // (max(y_corr)+1)
        tr_idxs = []
        val_idxs = []
        test_idxs = []
        for i in range(len(labels.keys())):
            np.random.shuffle(labels[i])
            tr_idxs.append(labels[i][:num_ex])
            val_idxs.append(labels[i][num_ex:num_ex+10])
            test_idxs.append(labels[i][num_ex+10:num_ex+20])
        tr_idxs = np.concatenate(tr_idxs)
        val_idxs = np.concatenate(val_idxs)
        test_idxs = np.concatenate(test_idxs)
        tr_dataset = Subset(dataset, tr_idxs)
        val_dataset = Subset(dataset, val_idxs)
        te_dataset = Subset(dataset, test_idxs)

    return tr_dataset, val_dataset, te_dataset
