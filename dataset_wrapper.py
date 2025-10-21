from torchvision.datasets import CIFAR10, MNIST
from typing import Tuple, Any, Optional
import torch

class CIFAR10Wrapper(CIFAR10):
    """Wrapper for CIFAR10 dataset that adds placeholder captions and rule indices"""
    def __init__(self, root: str, train: bool = True, transform: Optional[Any] = None, target_transform: Optional[Any] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        # Create placeholder captions and rule indices
        self.captions = [f"Image of {self.classes[label]}" for label in self.targets]
        self.rule_indices = [[0] * len(self.classes) for _ in range(len(self.targets))]  # Initialize with zeros

    def __getitem__(self, index: int) -> Tuple[Any, Any, str, list]:
        img, target = super().__getitem__(index)
        caption = self.captions[index]
        rule_indices = self.rule_indices[index]
        return img, target, caption, rule_indices

class MNISTWrapper(MNIST):
    """Wrapper for MNIST dataset that adds placeholder captions and rule indices"""
    def __init__(self, root: str, train: bool = True, transform: Optional[Any] = None, target_transform: Optional[Any] = None, download: bool = False):
        super().__init__(root, train, transform, target_transform, download)
        # Create placeholder captions and rule indices
        self.captions = [f"Digit {target}" for target in self.targets]
        self.rule_indices = [[0] * 10 for _ in range(len(self.targets))]  # Initialize with zeros

    def __getitem__(self, index: int) -> Tuple[Any, Any, str, list]:
        img, target = super().__getitem__(index)
        caption = self.captions[index]
        rule_indices = self.rule_indices[index]
        return img, target, caption, rule_indices

