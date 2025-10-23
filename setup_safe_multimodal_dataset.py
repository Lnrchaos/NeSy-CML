#!/usr/bin/env python3
"""
Complete setup script for safe multimodal dataset
Creates synthetic images and downloads public domain images
"""

import os
import sys
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from create_safe_image_dataset import SafeImageDatasetCreator
from download_public_images import PublicImageDownloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_complete_dataset():
    """Set up the complete safe multimodal dataset"""
    logger.info("🚀 Setting up complete safe multimodal dataset...")
    
    # Create base directory
    base_dir = Path("dataset/safe_images")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Step 1: Creating synthetic images...")
    # Create synthetic images
    creator = SafeImageDatasetCreator(output_dir=base_dir)
    creator.create_dataset()
    
    print("\n🌐 Step 2: Downloading public domain images...")
    # Download public domain images
    downloader = PublicImageDownloader(output_dir=base_dir / "downloaded")
    try:
        downloader.download_all()
    except Exception as e:
        logger.warning(f"Some downloads failed: {e}")
        logger.info("Continuing with synthetic images only...")
    
    print("\n📋 Step 3: Creating dataset summary...")
    create_dataset_summary(base_dir)
    
    print("\n✅ Safe multimodal dataset setup complete!")
    print(f"📁 Location: {base_dir}")
    print("🔒 Safe for public GitHub repositories")
    print("🤖 Ready for multimodal AI training")

def create_dataset_summary(base_dir):
    """Create a summary of the dataset"""
    summary = {
        "dataset_name": "Safe Multimodal Training Dataset",
        "description": "Comprehensive safe image dataset for multimodal AI training",
        "categories": {},
        "total_images": 0,
        "license": "Mixed - See individual category licenses",
        "safe_for_github": True
    }
    
    # Count images in each category
    for category_dir in base_dir.rglob("*"):
        if category_dir.is_dir() and category_dir != base_dir:
            png_files = list(category_dir.glob("*.png"))
            jpg_files = list(category_dir.glob("*.jpg"))
            total_files = len(png_files) + len(jpg_files)
            
            if total_files > 0:
                rel_path = category_dir.relative_to(base_dir)
                summary["categories"][str(rel_path)] = {
                    "count": total_files,
                    "formats": ["PNG", "JPG"] if jpg_files else ["PNG"]
                }
                summary["total_images"] += total_files
    
    # Save summary
    import json
    with open(base_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n📊 Dataset Summary:")
    print(f"   Total Images: {summary['total_images']}")
    print(f"   Categories: {len(summary['categories'])}")
    for category, info in summary['categories'].items():
        print(f"   - {category}: {info['count']} images")

def update_multimodal_trainer():
    """Update the multimodal trainer to use safe images"""
    logger.info("📝 Updating multimodal trainer configuration...")
    
    # Check if train_multimodal_xavious.py exists
    trainer_file = Path("train_multimodal_xavious.py")
    if not trainer_file.exists():
        logger.warning("train_multimodal_xavious.py not found - skipping trainer update")
        return
    
    # Create a configuration note
    config_note = """
# Safe Image Dataset Configuration
# Add this to your multimodal trainer config:

SAFE_IMAGE_CONFIG = {
    'image_dataset_path': 'dataset/safe_images',
    'image_categories': [
        'geometric', 'colors', 'numbers', 'letters', 
        'chess_pieces', 'programming', 'patterns', 
        'charts', 'educational', 'downloaded/picsum',
        'downloaded/unsplash', 'downloaded/nature',
        'downloaded/objects', 'downloaded/abstract'
    ],
    'image_size': (224, 224),
    'safe_for_github': True,
    'license_compliant': True
}

# Usage in your trainer:
# dataset = MultiModalDataset(
#     data_dir='dataset', 
#     tokenizer=tokenizer,
#     image_config=SAFE_IMAGE_CONFIG
# )
"""
    
    with open("safe_image_config.py", 'w') as f:
        f.write(config_note)
    
    logger.info("✅ Created safe_image_config.py with configuration")

def main():
    """Main function"""
    print("🎨 Safe Multimodal Dataset Setup")
    print("=" * 50)
    print("This script creates a comprehensive, safe image dataset")
    print("suitable for public GitHub repositories and AI training.")
    print()
    
    try:
        setup_complete_dataset()
        update_multimodal_trainer()
        
        print("\n🎉 Setup Complete!")
        print("\nNext steps:")
        print("1. Review the dataset in dataset/safe_images/")
        print("2. Check dataset_summary.json for details")
        print("3. Use safe_image_config.py in your trainer")
        print("4. Run your multimodal training!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"\n❌ Setup failed: {e}")

if __name__ == "__main__":
    main()