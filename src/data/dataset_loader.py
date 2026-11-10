import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets
from typing import Optional, Tuple
from .transforms import get_transforms


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for ultra-fast CPU unit testing and dry runs
    without requiring network downloads or gigabytes of disk space.
    """

    def __init__(self, num_samples: int = 100, num_classes: int = 10, image_size: int = 224):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        self.data = torch.randn(num_samples, 3, image_size, image_size)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]


def get_dataset_loaders(
    dataset_name: str,
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    image_size: int = 224,
    debug_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Creates and returns train_loader, val_loader, and num_classes.
    Supports 'svhn', 'cifar100', 'flowers102', and 'synthetic'.
    """
    dataset_name = dataset_name.lower()
    os.makedirs(data_dir, exist_ok=True)

    train_transform = get_transforms(dataset_name, image_size=image_size, is_training=True)
    val_transform = get_transforms(dataset_name, image_size=image_size, is_training=False)

    if dataset_name == "synthetic":
        num_classes = 10
        train_set = SyntheticDataset(num_samples=debug_subset or 128, num_classes=num_classes, image_size=image_size)
        val_set = SyntheticDataset(num_samples=debug_subset or 64, num_classes=num_classes, image_size=image_size)

    elif dataset_name == "svhn":
        num_classes = 10
        train_set = datasets.SVHN(
            root=data_dir, split="train", download=True, transform=train_transform
        )
        val_set = datasets.SVHN(
            root=data_dir, split="test", download=True, transform=val_transform
        )

    elif dataset_name == "cifar100":
        num_classes = 100
        train_set = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        val_set = datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=val_transform
        )

    elif dataset_name in ["flowers102", "oxford_flowers"]:
        num_classes = 102
        train_set = datasets.Flowers102(
            root=data_dir, split="train", download=True, transform=train_transform
        )
        val_set = datasets.Flowers102(
            root=data_dir, split="test", download=True, transform=val_transform
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Apply debug subset if specified
    if debug_subset is not None and dataset_name != "synthetic":
        train_size = min(debug_subset, len(train_set))
        val_size = min(debug_subset // 2, len(val_set))
        train_set = Subset(train_set, list(range(train_size)))
        val_set = Subset(val_set, list(range(val_size)))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader, num_classes

