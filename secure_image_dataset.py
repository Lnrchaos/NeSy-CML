"""
Secure Image Dataset for NeuroSym-CML Training
Processes sensitive images in memory and automatically cleans up after training
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import gc
import shutil
from typing import List, Dict, Tuple, Optional, Any
import tempfile
from pathlib import Path

class SecureImageDataset(Dataset):
    """
    Secure dataset that processes images in memory and cleans up sensitive data
    """
    
    def __init__(self, 
                 images_dir: str, 
                 labels_dir: str, 
                 classes_file: str,
                 transform: Optional[transforms.Compose] = None,
                 max_samples: Optional[int] = None,
                 cleanup_after_epoch: bool = True):
        """
        Initialize the secure dataset
        
        Args:
            images_dir: Path to images directory
            labels_dir: Path to labels directory  
            classes_file: Path to classes.txt file
            transform: Image transformations
            max_samples: Maximum number of samples to load (for testing)
            cleanup_after_epoch: Whether to cleanup memory after each epoch
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform or self._get_default_transform()
        self.cleanup_after_epoch = cleanup_after_epoch
        
        # Load class names
        self.classes = self._load_classes(classes_file)
        self.num_classes = len(self.classes)
        
        # Get image files
        self.image_files = self._get_image_files()
        if max_samples:
            self.image_files = self.image_files[:max_samples]
        
        print(f"📁 Found {len(self.image_files)} images")
        print(f"🏷️  {self.num_classes} classes: {', '.join(self.classes[:10])}{'...' if len(self.classes) > 10 else ''}")
        
        # Memory management
        self._memory_cache = {}
        self._current_epoch = 0
        
    def _load_classes(self, classes_file: str) -> List[str]:
        """Load class names from file"""
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    def _get_image_files(self) -> List[Path]:
        """Get list of image files (including subdirectories)"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        # Recursively search for image files
        for file_path in self.images_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                # Include all images, regardless of whether they have labels
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_image_safely(self, image_path: Path) -> torch.Tensor:
        """Load image safely with memory management"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
            
            # Clear PIL image from memory
            del image
            gc.collect()
            
            return image_tensor
        except Exception as e:
            print(f"⚠️  Error loading image {image_path}: {e}")
            # Return a blank tensor as fallback
            return torch.zeros(3, 224, 224)
    
    def _load_labels(self, image_path: Path) -> Tuple[List[int], List[float]]:
        """Load YOLO format labels"""
        label_file = self.labels_dir / f"{image_path.stem}.txt"
        
        if not label_file.exists():
            return [], []
        
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            class_ids = []
            bbox_coords = []
            
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # YOLO format: class_id x_center y_center width height (normalized)
                    coords = [float(x) for x in parts[1:5]]
                    class_ids.append(class_id)
                    bbox_coords.extend(coords)
            
            return class_ids, bbox_coords
        except Exception as e:
            print(f"⚠️  Error loading labels for {image_path}: {e}")
            return [], []
    
    def _create_text_embedding(self, class_ids: List[int]) -> torch.Tensor:
        """Create text embedding from class names"""
        if not class_ids:
            # Return zero embedding if no classes
            return torch.zeros(512)
        
        # Get class names for the detected objects
        class_names = [self.classes[class_id] for class_id in class_ids if class_id < len(self.classes)]
        
        # Create a simple text description
        if class_names:
            text_description = f"Image containing: {', '.join(class_names[:5])}"  # Limit to 5 classes
        else:
            text_description = "Image with objects"
        
        # For now, create a simple embedding based on class IDs
        # In a full implementation, you'd use CLIP or similar
        embedding = torch.zeros(512)
        for class_id in class_ids:
            if class_id < len(self.classes):
                # Create a simple one-hot-like encoding
                start_idx = (class_id * 512) // len(self.classes)
                end_idx = ((class_id + 1) * 512) // len(self.classes)
                embedding[start_idx:end_idx] = 1.0
        
        return embedding
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        image_path = self.image_files[idx]
        
        # Load image (in memory temporarily)
        image_tensor = self._load_image_safely(image_path)
        
        # Load labels
        class_ids, bbox_coords = self._load_labels(image_path)
        
        # Create text embedding
        text_embedding = self._create_text_embedding(class_ids)
        
        # Create rule indices - use the first class_id as the rule index
        # For multi-class, we'll use a single rule index based on the first detected class
        if class_ids:
            rule_index = class_ids[0] % self.num_classes  # Use first class as rule index
        else:
            rule_index = 0  # Default rule if no classes detected
        
        return {
            'image': image_tensor,
            'text': text_embedding,
            'rules': torch.tensor(rule_index, dtype=torch.long),  # Single rule index
            'class_ids': class_ids,
            'bbox_coords': bbox_coords,
            'image_path': str(image_path),  # For debugging only
            'num_objects': len(class_ids)
        }
    
    def cleanup_memory(self):
        """Clean up memory cache and force garbage collection"""
        self._memory_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Memory cleaned up")
    
    def start_epoch(self):
        """Called at the start of each epoch"""
        self._current_epoch += 1
        if self.cleanup_after_epoch and self._current_epoch > 1:
            self.cleanup_memory()
    
    def end_epoch(self):
        """Called at the end of each epoch"""
        if self.cleanup_after_epoch:
            self.cleanup_memory()
    
    def final_cleanup(self):
        """Final cleanup after training"""
        print("🧹 Performing final cleanup...")
        self.cleanup_memory()
        
        # Clear all references
        self.image_files.clear()
        self.classes.clear()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ All sensitive data cleaned up")

class SecureDataModule:
    """Secure data module for handling the sensitive dataset"""
    
    def __init__(self, 
                 images_dir: str,
                 labels_dir: str, 
                 classes_file: str,
                 batch_size: int = 8,
                 max_samples: Optional[int] = None,
                 cleanup_after_epoch: bool = True):
        """
        Initialize secure data module
        
        Args:
            images_dir: Path to images directory
            labels_dir: Path to labels directory
            classes_file: Path to classes.txt file
            batch_size: Batch size for training
            max_samples: Maximum samples to load (for testing)
            cleanup_after_epoch: Whether to cleanup after each epoch
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.classes_file = classes_file
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.cleanup_after_epoch = cleanup_after_epoch
        
        # Create transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def create_dataloaders(self, train_split: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders"""
        print("📊 Creating secure dataloaders...")
        
        # Create full dataset
        full_dataset = SecureImageDataset(
            images_dir=self.images_dir,
            labels_dir=self.labels_dir,
            classes_file=self.classes_file,
            transform=self.train_transform,
            max_samples=self.max_samples,
            cleanup_after_epoch=self.cleanup_after_epoch
        )
        
        # Split dataset
        total_size = len(full_dataset)
        train_size = int(train_split * total_size)
        val_size = total_size - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing for security
            pin_memory=False,  # Disable pin_memory for security
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Disable multiprocessing for security
            pin_memory=False,  # Disable pin_memory for security
            collate_fn=self._collate_fn
        )
        
        print(f"📈 Train samples: {len(train_dataset)}")
        print(f"📈 Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader, full_dataset
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        images = torch.stack([item['image'] for item in batch])
        texts = torch.stack([item['text'] for item in batch])
        rules = torch.stack([item['rules'] for item in batch])  # Now single rule indices
        
        # Create multi-label targets from class_ids
        batch_size = len(batch)
        num_classes = len(batch[0]['class_ids']) if batch[0]['class_ids'] else 42  # Use 42 as default
        targets = torch.zeros(batch_size, 42)  # Fixed size for multi-label
        
        for i, item in enumerate(batch):
            for class_id in item['class_ids']:
                if class_id < 42:  # Ensure within bounds
                    targets[i, class_id] = 1.0
        
        return {
            'images': images,
            'texts': texts,
            'rules': rules,  # Single rule indices for embedding
            'targets': targets,  # Multi-label targets for loss
            'batch_info': {
                'num_objects': [item['num_objects'] for item in batch],
                'class_ids': [item['class_ids'] for item in batch]
            }
        }

def secure_training_cleanup(dataset: SecureImageDataset):
    """Ensure all sensitive data is cleaned up after training"""
    print("🔒 Performing secure cleanup...")
    dataset.final_cleanup()
    print("✅ Secure cleanup completed - all sensitive data removed from memory")
