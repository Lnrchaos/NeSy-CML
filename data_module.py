import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from dataset_wrapper import CIFAR10Wrapper, MNISTWrapper
from typing import Tuple, Optional, Dict, Any, List
from text_encoder import TextEncoder
import random
from model_spec import ModelSpec
import numpy as np

class ContinualDataset(Dataset):
    """Dataset wrapper for continual learning with meta-task sampling"""
    def __init__(self, dataset: Dataset, spec: ModelSpec, transform: Optional[transforms.Compose] = None):
        self.dataset = dataset
        self.spec = spec
        self.transform = transform
        self.meta_batch_size = spec.meta_batch_size
        self.memory_size = spec.memory_size
        
        # Initialize text encoder with device configuration
        self.text_encoder = TextEncoder(model_name="openai/clip-vit-base-patch32")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize multimodal memory buffers
        self.memory_images = []  # Store image tensors
        self.memory_texts = []  # Store text embeddings
        self.memory_labels = []  # Store corresponding labels
        self.memory_rule_indices = []  # Store rule indices for each sample
        
        # Configure worker settings based on environment
        self.worker_config = {
            'num_workers': min(4, os.cpu_count()),  # Use up to 4 workers but not more than CPU cores
            'pin_memory': True if torch.cuda.is_available() else False,  # Only pin memory if using GPU
            'prefetch_factor': 2  # Number of batches to prefetch
        }
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def visualize_batch(self, batch_size: int = 8, mode: str = 'dataset') -> None:
        """
        Visualize a batch of images from either the dataset or memory buffer
        
        Args:
            batch_size: Number of samples to visualize
            mode: 'dataset' or 'memory' to specify which source to visualize
        """
        import matplotlib.pyplot as plt
        
        if mode == 'dataset':
            indices = np.random.choice(len(self), size=batch_size, replace=False)
            samples = [self[i] for i in indices]
            title_prefix = 'Dataset'
        elif mode == 'memory':
            if len(self.memory_images) == 0:
                print("Memory is empty!")
                return
            indices = np.random.choice(len(self.memory_images), size=min(batch_size, len(self.memory_images)), replace=False)
            samples = [(self.memory_images[i], self.memory_labels[i], None) for i in indices]
            title_prefix = 'Memory Buffer'
        else:
            raise ValueError("mode must be either 'dataset' or 'memory'")
        
        # Create a grid of images
        fig, axes = plt.subplots(2, batch_size//2, figsize=(15, 6))
        axes = axes.flatten()
        
        for i, (image, label, _) in enumerate(samples):
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0)  # Convert from CxHxW to HxWxC
                image = image.numpy()
            
            axes[i].imshow(image)
            axes[i].set_title(f'{title_prefix} - Label: {label}')
            axes[i].axis('off')
        
        plt.suptitle(f'Visualization of {batch_size} samples from {title_prefix}')
        plt.tight_layout()
        plt.show()

    def summarize_dataset(self) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset and memory buffer
        
        Returns:
            Dictionary containing dataset statistics and memory information
        """
        from collections import Counter
        import torch
        
        # Get dataset statistics
        dataset_stats = {
            'total_samples': len(self.dataset),
            'memory_size': self.memory_size,
            'current_memory_size': len(self.memory_images)
        }
        
        # Get label distribution from dataset
        try:
            labels = [self.dataset[i][1] for i in range(len(self.dataset))]
            label_dist = dict(Counter(labels))
            dataset_stats['label_distribution'] = label_dist
            dataset_stats['unique_labels'] = len(label_dist)
        except:
            dataset_stats['label_distribution'] = 'Could not retrieve label distribution'
            
        # Get memory statistics if available
        if self.memory_images:
            memory_stats = {
                'memory_samples': len(self.memory_images),
                'memory_labels': dict(Counter(self.memory_labels))
            }
            dataset_stats['memory_stats'] = memory_stats
        else:
            dataset_stats['memory_stats'] = 'Memory buffer is empty'
            
        # Print summary
        print("\nDataset Summary:")
        print("-" * 50)
        print(f"Total samples: {dataset_stats['total_samples']}")
        print(f"Memory size: {dataset_stats['memory_size']}")
        print(f"Current memory samples: {dataset_stats.get('current_memory_size', 0)}")
        
        print("\nLabel Distribution:")
        print("-" * 50)
        if isinstance(dataset_stats['label_distribution'], dict):
            for label, count in sorted(dataset_stats['label_distribution'].items()):
                print(f"Label {label}: {count} samples")
        else:
            print(dataset_stats['label_distribution'])
            
        if 'memory_stats' in dataset_stats:
            print("\nMemory Buffer Stats:")
            print("-" * 50)
            print(f"Memory samples: {dataset_stats['memory_stats']['memory_samples']}")
            print("Memory label distribution:")
            for label, count in sorted(dataset_stats['memory_stats']['memory_labels'].items()):
                print(f"Label {label}: {count} samples")
                
        return dataset_stats
        plt.show()
        
    def get_meta_batch(self, n_way: int, k_shot: int, query_size: int):
        """
        Sample a meta-batch of N-way K-shot tasks with image-text pairs and memory replay
        
        Args:
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            query_size: Number of query examples per class
            
        Returns:
            Tuple containing (support_images, support_labels, support_text_embeddings, support_rule_indices,
                           query_images, query_labels, query_text_embeddings, query_rule_indices)
        """
        # Sample classes for the meta-task
        classes = torch.randperm(len(self.dataset.classes))[:n_way]
        
        # Initialize buffers
        support_images, support_labels, support_text_embeddings, support_rule_indices = [], [], [], []
        query_images, query_labels, query_text_embeddings, query_rule_indices = [], [], [], []
        
        # Sample from dataset
        for class_idx in classes:
            class_samples = torch.where(torch.tensor(self.dataset.targets) == class_idx)[0]
            
            # Sample support set
            support_idx = torch.randperm(len(class_samples))[:k_shot]
            support_samples = class_samples[support_idx]
            
            # Sample query set
            remaining_idx = torch.randperm(len(class_samples))[k_shot:k_shot+query_size]
            query_samples = class_samples[remaining_idx]
            
            # Get samples with text embeddings
            for idx in support_samples:
                img, label, text_embedding, rule_idx = self.__getitem__(idx)
                support_images.append(img)
                support_labels.append(label)
                support_text_embeddings.append(text_embedding)
                support_rule_indices.append(rule_idx)
            
            for idx in query_samples:
                img, label, text_embedding, rule_idx = self.__getitem__(idx)
                query_images.append(img)
                query_labels.append(label)
                query_text_embeddings.append(text_embedding)
                query_rule_indices.append(rule_idx)
        
        return (
            torch.stack(support_images), torch.tensor(support_labels), torch.stack(support_text_embeddings), support_rule_indices,
            torch.stack(query_images), torch.tensor(query_labels), torch.stack(query_text_embeddings), query_rule_indices
        )
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, torch.Tensor, list]:
        """
        Get a single sample from the dataset with text encoding
        
        Args:
            index: Index of the sample to retrieve
            
        Returns:
            Tuple containing (image, label, text_embedding, rule_indices)
        """
        image, label, caption, rule_indices = self.dataset[index]
        if self.transform:
            image = self.transform(image)
            
        # Encode caption to text embedding
        text_embedding = self.text_encoder.encode_single_text(caption)
        return image, label, text_embedding, rule_indices

    def get_memory_sample(self, index: int) -> Tuple[torch.Tensor, int, str, list]:
        """
        Get a single sample from the memory buffer
        
        Args:
            index: Index of the sample to retrieve from memory
            
        Returns:
            Tuple containing (image, label, caption, rule_indices)
        """
        if index >= len(self.memory_images):
            raise IndexError(f"Memory index {index} out of bounds")
        
        # Return the sample from memory
        return (
            self.memory_images[index],
            self.memory_labels[index],
            self.memory_texts[index],
            self.memory_rule_indices[index]
        )

class DataModule:
    """Data module for handling dataset loading and transformations"""
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.train_transform = self._get_train_transforms()
        self.val_transform = self._get_val_transforms()
        
    def _get_train_transforms(self) -> transforms.Compose:
        """Get training data transformations"""
        return transforms.Compose([
            transforms.RandomResizedCrop(self.spec.input_shape[1:]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _get_val_transforms(self) -> transforms.Compose:
        """Get validation data transformations"""
        return transforms.Compose([
            transforms.Resize(self.spec.input_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, dataset_name: str = "cifar10") -> Tuple[DataLoader, DataLoader]:
        """
        Prepare and return train and validation dataloaders with meta-batch sampling
        
        Args:
            dataset_name: Name of the dataset to load
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load dataset using wrappers
        if dataset_name == "cifar10":
            train_dataset = CIFAR10Wrapper(
                root='./data',
                train=True,
                download=True
            )
            val_dataset = CIFAR10Wrapper(
                root='./data',
                train=False,
                download=True
            )
        elif dataset_name == "mnist":
            train_dataset = MNISTWrapper(
                root='./data',
                train=True,
                download=True
            )
            val_dataset = MNISTWrapper(
                root='./data',
                train=False,
                download=True
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Wrap with ContinualDataset
        train_dataset = ContinualDataset(
            train_dataset,
            self.spec,
            transform=self.train_transform
        )
        val_dataset = ContinualDataset(
            val_dataset,
            self.spec,
            transform=self.val_transform
        )
        
        # Create dataloaders with meta-batch sampling
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.spec.meta_batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=lambda x: train_dataset.get_meta_batch(
                n_way=self.spec.n_way,
                k_shot=self.spec.k_shot,
                query_size=self.spec.query_size
            )
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.spec.meta_batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=lambda x: val_dataset.get_meta_batch(
                n_way=self.spec.n_way,
                k_shot=self.spec.k_shot,
                query_size=self.spec.query_size
            )
        )
        
        # Store references to datasets for memory updates
        self.train_dataset = train_dataset
        
        return train_loader, val_loader

    def update_memory(self, samples: torch.Tensor, labels: torch.Tensor, captions: list, rule_indices: list):
        """Update the memory buffer with new samples"""
        # Update memory in train dataset
        if hasattr(self, 'train_dataset'):
            self.train_dataset.memory_images.extend(samples)
            self.train_dataset.memory_labels.extend(labels.tolist())
            self.train_dataset.memory_texts.extend(captions)
            self.train_dataset.memory_rule_indices.extend(rule_indices)
            
            # If memory exceeds capacity, remove oldest samples
            if len(self.train_dataset.memory_images) > self.spec.memory_size:
                num_to_remove = len(self.train_dataset.memory_images) - self.spec.memory_size
                self.train_dataset.memory_images = self.train_dataset.memory_images[num_to_remove:]
                self.train_dataset.memory_labels = self.train_dataset.memory_labels[num_to_remove:]
                self.train_dataset.memory_texts = self.train_dataset.memory_texts[num_to_remove:]
                self.train_dataset.memory_rule_indices = self.train_dataset.memory_rule_indices[num_to_remove:]

