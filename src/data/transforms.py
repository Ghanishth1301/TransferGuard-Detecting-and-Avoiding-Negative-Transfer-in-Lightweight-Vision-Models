"""
src/data/transforms.py
Domain-specific image augmentation and preprocessing pipelines.

Authors: Arvind Yadav, Ghanishth Saini, Aakash Jagtap
"""

from torchvision import transforms
from typing import Tuple


def get_transforms(
    dataset_name: str,
    image_size: int = 224,
    is_training: bool = True,
) -> transforms.Compose:
    """
    Returns appropriate image transformations.
    
    CRITICAL CV NOTE: For digit recognition datasets like SVHN, RandomHorizontalFlip
    MUST NOT be used, as horizontal mirroring invalidates digits (e.g. 6 becomes an inverted glyph, 3 flips).
    """
    dataset_name = dataset_name.lower()
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_training:
        transform_list = []
        if dataset_name == "svhn":
            # No horizontal flip for digits! Mild rotation and scaling only.
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.RandomRotation(degrees=7),
                transforms.ColorJitter(brightness=0.15, contrast=0.15),
            ])
        elif dataset_name in ["cifar100", "cifar10"]:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(image_size, padding=4, padding_mode="reflect"),
            ])
        elif dataset_name in ["flowers102", "oxford_flowers"]:
            transform_list.extend([
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
            ])
        else:
            transform_list.extend([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
            ])

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        # Validation / Test transforms
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]

    return transforms.Compose(transform_list)

