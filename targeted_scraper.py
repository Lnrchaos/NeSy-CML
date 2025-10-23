#!/usr/bin/env python3
"""
Targeted Web Scraper for Specific Bing URLs
Scrapes images from specific Bing search URLs and labels them using NeuroSym-CML
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import json
import cv2
import requests
from bs4 import BeautifulSoup
import time
import random
from typing import List, Dict, Any, Tuple
import shutil
from datetime import datetime
import urllib.parse
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Import the original NeuroSym-CML components
from meta_model import HybridModel, ModelSpec

class TargetedScraper:
    """Scrape images from specific Bing URLs and label them using NeuroSym-CML"""
    
    def __init__(self, neurosym_checkpoint_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load NeuroSym-CML model
        if neurosym_checkpoint_path:
            self.neurosym_model = self._load_neurosym_model(neurosym_checkpoint_path)
        else:
            checkpoint_path = "checkpoints/secure_checkpoint_epoch_8.pt"
            if os.path.exists(checkpoint_path):
                self.neurosym_model = self._load_neurosym_model(checkpoint_path)
            else:
                print("❌ No NeuroSym-CML checkpoint found!")
                return
        
        # Define class mappings
        self.class_mapping = {
            0: 'abdomen', 1: 'arm', 2: 'ass', 3: 'black_penis', 4: 'bra', 5: 'bra_panties',
            6: 'breast', 7: 'clothed_chest', 8: 'doggystyle', 9: 'exposed_chest', 10: 'exposing_body',
            11: 'exposing_chest', 12: 'exposing_pussy', 13: 'eye', 14: 'face', 15: 'foot',
            16: 'giving_blowjob', 17: 'hand', 18: 'interracial_blowjob', 19: 'interracial_sex',
            20: 'leg', 21: 'lingerie', 22: 'missionary', 23: 'mouth', 24: 'nipple', 25: 'nose',
            26: 'panties', 27: 'pants', 28: 'penis', 29: 'pierced_nipple', 30: 'pussy',
            31: 'sex', 32: 'shirt', 33: 'shoe', 34: 'shorts', 35: 'shoulder', 36: 'skirt',
            37: 'sock', 38: 'tattoo', 39: 'wearing_clothes', 40: 'white_penis', 41: 'wrist'
        }
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Scraping configuration
        self.scraping_config = {
            'delay_between_requests': 2.0,
            'timeout': 15,
            'max_retries': 3,
            'user_agents': [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ]
        }
        
        print(f"✅ Targeted scraper initialized")
        print(f"🎯 NeuroSym-CML model loaded for labeling")
        print(f"🎯 Scraping config: {self.scraping_config}")
    
    def _load_neurosym_model(self, checkpoint_path: str):
        """Load NeuroSym-CML model from checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            config = checkpoint.get('config', {})
            
            model_spec = ModelSpec(
                neural_architecture=config.get('neural_architecture', 'custom_cnn'),
                num_classes=config.get('num_classes', 42),
                input_size=config.get('input_size', 224)
            )
            
            model = HybridModel(model_spec)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(state_dict, strict=False)
            
            print(f"✅ NeuroSym-CML model loaded from {checkpoint_path}")
            return model
            
        except Exception as e:
            print(f"❌ Error loading NeuroSym-CML model: {e}")
            return None
    
    def scrape_specific_urls(self, urls: List[str], output_dir: str = "targeted_scraped", max_images_per_url: int = 100) -> Dict[str, Any]:
        """Scrape images from specific Bing URLs"""
        print(f"🔍 Scraping {len(urls)} specific URLs")
        print(f"📊 Max images per URL: {max_images_per_url}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_scraped_images = []
        url_results = {}
        
        for i, url in enumerate(urls):
            print(f"\n--- Scraping URL {i+1}/{len(urls)} ---")
            print(f"🌐 URL: {url}")
            
            # Create subdirectory for this URL
            url_dir = os.path.join(output_dir, f"url_{i+1:02d}")
            os.makedirs(url_dir, exist_ok=True)
            
            # Scrape images from this URL
            scraped_images = self._scrape_bing_url(url, url_dir, max_images_per_url)
            
            url_results[f"url_{i+1:02d}"] = {
                'url': url,
                'images_found': len(scraped_images),
                'output_dir': url_dir
            }
            
            all_scraped_images.extend(scraped_images)
            
            print(f"✅ Scraped {len(scraped_images)} images from URL {i+1}")
        
        # Save scraping results
        scraping_metadata = {
            'total_urls': len(urls),
            'total_images': len(all_scraped_images),
            'url_results': url_results,
            'scraping_timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "scraping_results.json"), 'w') as f:
            json.dump(scraping_metadata, f, indent=2)
        
        print(f"\n✅ URL scraping completed!")
        print(f"📊 Total images scraped: {len(all_scraped_images)}")
        print(f"📁 Results saved to: {output_dir}")
        
        return scraping_metadata
    
    def _scrape_bing_url(self, url: str, output_dir: str, max_images: int) -> List[str]:
        """Scrape images from a specific Bing URL"""
        scraped_images = []
        
        # Set up Chrome driver
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_argument('--disable-extensions')
        chrome_options.add_argument('--disable-plugins')
        chrome_options.add_argument('--disable-images')
        chrome_options.add_argument(f'--user-agent={random.choice(self.scraping_config["user_agents"])}')
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"❌ Chrome driver not available: {e}")
            print("💡 Install Chrome and chromedriver")
            return []
        
        try:
            print(f"🌐 Loading URL: {url}")
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Scroll to load more images
            print("📜 Scrolling to load more images...")
            for scroll in range(5):
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
                print(f"   Scroll {scroll + 1}/5")
            
            # Find image elements
            print("🔍 Finding image elements...")
            image_elements = driver.find_elements(By.CSS_SELECTOR, "img[src*='http']")
            print(f"📊 Found {len(image_elements)} image elements")
            
            # Download images
            for i, img_element in enumerate(image_elements[:max_images]):
                try:
                    img_url = img_element.get_attribute('src')
                    if not img_url or 'data:' in img_url or 'blob:' in img_url:
                        continue
                    
                    # Clean URL
                    if '?' in img_url:
                        img_url = img_url.split('?')[0]
                    
                    # Download image
                    img_filename = f"scraped_{i:03d}.jpg"
                    img_path = os.path.join(output_dir, img_filename)
                    
                    if self._download_image(img_url, img_path):
                        scraped_images.append(img_path)
                        print(f"✅ Downloaded: {img_filename}")
                    else:
                        print(f"❌ Failed to download: {img_url[:50]}...")
                    
                    # Delay between requests
                    time.sleep(self.scraping_config['delay_between_requests'])
                    
                except Exception as e:
                    print(f"❌ Error downloading image {i}: {e}")
                    continue
            
        except Exception as e:
            print(f"❌ Error during scraping: {e}")
        
        finally:
            driver.quit()
        
        return scraped_images
    
    def _download_image(self, url: str, filepath: str) -> bool:
        """Download image from URL with retries"""
        for attempt in range(self.scraping_config['max_retries']):
            try:
                headers = {
                    'User-Agent': random.choice(self.scraping_config['user_agents']),
                    'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Connection': 'keep-alive',
                    'Referer': 'https://www.bing.com/',
                    'Sec-Fetch-Dest': 'image',
                    'Sec-Fetch-Mode': 'no-cors',
                    'Sec-Fetch-Site': 'cross-site'
                }
                
                response = requests.get(url, headers=headers, timeout=self.scraping_config['timeout'])
                response.raise_for_status()
                
                # Check if it's actually an image
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'webp']):
                    print(f"⚠️ Not an image: {content_type}")
                    return False
                
                # Save image
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                # Verify it's a valid image
                try:
                    with Image.open(filepath) as img:
                        img.verify()
                    return True
                except Exception:
                    os.remove(filepath)
                    return False
                    
            except Exception as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt < self.scraping_config['max_retries'] - 1:
                    time.sleep(2)
                else:
                    return False
        
        return False
    
    def label_scraped_images(self, image_paths: List[str], output_dir: str, threshold: float = 0.1) -> Dict[str, Any]:
        """Label scraped images using NeuroSym-CML"""
        print(f"🧠 Labeling {len(image_paths)} scraped images")
        print(f"🎯 Threshold: {threshold}")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labeled_images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "metadata"), exist_ok=True)
        
        labeled_count = 0
        labeling_results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"\n--- Labeling {i+1}/{len(image_paths)}: {os.path.basename(image_path)} ---")
            
            try:
                # Label the image
                labeling_result = self._label_image(image_path, threshold)
                if not labeling_result:
                    print(f"❌ Failed to label {image_path}")
                    continue
                
                # Copy labeled image
                labeled_filename = f"labeled_{i:03d}_{os.path.basename(image_path)}"
                labeled_path = os.path.join(output_dir, "labeled_images", labeled_filename)
                shutil.copy2(image_path, labeled_path)
                
                # Save labeling metadata
                metadata = {
                    'original_path': image_path,
                    'labeled_path': labeled_path,
                    'labeling_result': labeling_result,
                    'labeling_timestamp': datetime.now().isoformat()
                }
                
                metadata_file = os.path.join(output_dir, "metadata", f"labeled_{i:03d}_meta.json")
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                labeling_results.append(metadata)
                labeled_count += 1
                
                # Print summary
                print(f"✅ Labeled: {len(labeling_result['all_predictions'])} predictions")
                if labeling_result['top_prediction']:
                    print(f"   Top: {labeling_result['top_prediction']['class_name']} ({labeling_result['top_prediction']['confidence']:.3f})")
                
            except Exception as e:
                print(f"❌ Error labeling {image_path}: {e}")
                continue
        
        # Save overall results
        overall_metadata = {
            'total_images': len(image_paths),
            'labeled_images': labeled_count,
            'labeling_threshold': threshold,
            'labeling_timestamp': datetime.now().isoformat(),
            'results': labeling_results
        }
        
        with open(os.path.join(output_dir, "labeling_results.json"), 'w') as f:
            json.dump(overall_metadata, f, indent=2)
        
        print(f"\n✅ Labeling completed!")
        print(f"📊 Labeled: {labeled_count}/{len(image_paths)} images")
        print(f"📁 Results saved to: {output_dir}")
        
        return overall_metadata
    
    def _label_image(self, image_path: str, threshold: float = 0.1) -> Dict[str, Any]:
        """Label a single image using NeuroSym-CML"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Create dummy inputs for NeuroSym-CML
            batch_size = input_tensor.size(0)
            text_embeddings = torch.zeros(batch_size, 512).to(self.device)
            rule_indices = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                self.neurosym_model.eval()
                outputs = self.neurosym_model(input_tensor, text_embeddings, rule_indices)
                probabilities = F.softmax(outputs, dim=1)
            
            # Extract predictions above threshold
            probs = probabilities[0].cpu().numpy()
            predictions = []
            
            for class_id, prob in enumerate(probs):
                if prob > threshold:
                    predictions.append({
                        'class_id': class_id,
                        'class_name': self.class_mapping.get(class_id, f'class_{class_id}'),
                        'confidence': float(prob),
                        'category': self._get_category(class_id)
                    })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Categorize predictions
            anatomical_features = [p for p in predictions if p['category'] == 'anatomical']
            clothing_items = [p for p in predictions if p['category'] == 'clothing']
            body_parts = [p for p in predictions if p['category'] == 'body']
            
            return {
                'image_path': image_path,
                'all_predictions': predictions,
                'anatomical_features': anatomical_features,
                'clothing_items': clothing_items,
                'body_parts': body_parts,
                'top_prediction': predictions[0] if predictions else None,
                'confidence_distribution': probs.tolist()
            }
            
        except Exception as e:
            print(f"❌ Error labeling {image_path}: {e}")
            return None
    
    def _get_category(self, class_id: int) -> str:
        """Get category for a class ID"""
        anatomical_classes = {6: 'breast', 24: 'nipple', 29: 'pierced_nipple', 38: 'tattoo'}
        clothing_classes = {4: 'bra', 26: 'panties', 27: 'pants', 32: 'shirt', 36: 'skirt', 21: 'lingerie'}
        body_classes = {0: 'abdomen', 1: 'arm', 13: 'eye', 14: 'face', 15: 'foot', 17: 'hand', 20: 'leg', 35: 'shoulder', 41: 'wrist'}
        
        if class_id in anatomical_classes:
            return 'anatomical'
        elif class_id in clothing_classes:
            return 'clothing'
        elif class_id in body_classes:
            return 'body'
        else:
            return 'other'
    
    def create_training_dataset(self, scraped_images: List[str], output_dir: str = "training_dataset") -> Dict[str, Any]:
        """Create training dataset from scraped and labeled images"""
        print(f"🎯 Creating training dataset from {len(scraped_images)} images")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "anatomical"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "clothing"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "body_parts"), exist_ok=True)
        
        # Label all images
        labeling_results = self.label_scraped_images(scraped_images, output_dir)
        
        # Categorize images
        anatomical_count = 0
        clothing_count = 0
        body_count = 0
        
        for result in labeling_results['results']:
            labeling_result = result['labeling_result']
            
            # Copy to appropriate category
            if labeling_result['anatomical_features']:
                anatomical_count += 1
                self._copy_to_category(result['labeled_path'], os.path.join(output_dir, "anatomical"))
            
            if labeling_result['clothing_items']:
                clothing_count += 1
                self._copy_to_category(result['labeled_path'], os.path.join(output_dir, "clothing"))
            
            if labeling_result['body_parts']:
                body_count += 1
                self._copy_to_category(result['labeled_path'], os.path.join(output_dir, "body_parts"))
        
        # Create dataset summary
        dataset_metadata = {
            'total_images': len(scraped_images),
            'labeled_images': labeling_results['labeled_images'],
            'anatomical_images': anatomical_count,
            'clothing_images': clothing_count,
            'body_images': body_count,
            'creation_timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "dataset_metadata.json"), 'w') as f:
            json.dump(dataset_metadata, f, indent=2)
        
        print(f"\n✅ Training dataset created!")
        print(f"📊 Total images: {len(scraped_images)}")
        print(f"📊 Anatomical features: {anatomical_count}")
        print(f"📊 Clothing items: {clothing_count}")
        print(f"📊 Body parts: {body_count}")
        print(f"📁 Dataset saved to: {output_dir}")
        
        return dataset_metadata
    
    def _copy_to_category(self, source_path: str, target_dir: str):
        """Copy image to category directory"""
        try:
            filename = os.path.basename(source_path)
            target_path = os.path.join(target_dir, filename)
            shutil.copy2(source_path, target_path)
        except Exception as e:
            print(f"❌ Error copying {source_path}: {e}")

def main():
    """Main function for targeted scraping"""
    print("🎯 Targeted Web Scraper + NeuroSym-CML Labeler")
    print("=" * 60)
    
    # Initialize scraper
    scraper = TargetedScraper()
    
    # Define the specific URLs you provided
    urls = [
        "https://www.bing.com/images/search?q=hannah+Hays&form=HDRSC3&first=1",
        "https://www.bing.com/images/search?q=hannah+hays+topless&FORM=HDRSC3",
        "https://www.bing.com/images/search?q=hannah+hays+completely+naked&qs=n&form=QBIR&sp=-1&lq=0&pq=hannah+hays+completely+naked&sc=0-28&cvid=332F5D49472448C7B23EA00816A0505C&first=1"
    ]
    
    print(f"🎯 Target URLs:")
    for i, url in enumerate(urls, 1):
        print(f"   {i}. {url}")
    
    # Scrape images from URLs
    max_images_per_url = int(input("\nEnter max images per URL (default 50): ") or "50")
    output_dir = "hannah_hays_scraped"
    
    print(f"\n🔍 Starting scraping process...")
    scraping_results = scraper.scrape_specific_urls(urls, output_dir, max_images_per_url)
    
    if scraping_results['total_images'] > 0:
        # Label all scraped images
        print(f"\n🧠 Starting labeling process...")
        all_images = []
        for url_result in scraping_results['url_results'].values():
            all_images.extend([f for f in os.listdir(url_result['output_dir']) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Create training dataset
        print(f"\n🎯 Creating training dataset...")
        dataset_metadata = scraper.create_training_dataset(all_images, "hannah_hays_training_dataset")
        
        print(f"\n✅ Complete process finished!")
        print(f"📊 Scraped: {scraping_results['total_images']} images")
        print(f"📊 Labeled: {dataset_metadata['labeled_images']} images")
        print(f"📊 Anatomical: {dataset_metadata['anatomical_images']} images")
        print(f"📊 Clothing: {dataset_metadata['clothing_images']} images")
        print(f"📊 Body parts: {dataset_metadata['body_images']} images")
    else:
        print("❌ No images scraped")

if __name__ == "__main__":
    main()
