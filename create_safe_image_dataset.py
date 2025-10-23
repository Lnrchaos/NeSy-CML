#!/usr/bin/env python3
"""
Create a safe, appropriate image dataset for multimodal training
Uses public domain and creative commons images suitable for GitHub
"""

import os
import requests
import json
import time
import hashlib
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import logging
from urllib.parse import urlparse
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SafeImageDatasetCreator:
    """Create appropriate images for multimodal training"""
    
    def __init__(self, output_dir="dataset/safe_images"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_synthetic_images(self):
        """Create synthetic images with text labels"""
        logger.info("Creating synthetic images...")
        
        categories = {
            "geometric": ["circle", "square", "triangle", "rectangle", "pentagon"],
            "colors": ["red", "blue", "green", "yellow", "purple", "orange"],
            "numbers": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "letters": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
            "chess_pieces": ["♔", "♕", "♖", "♗", "♘", "♙"],
            "programming": ["def", "class", "if", "for", "while", "return"],
        }
        
        for category, items in categories.items():
            category_dir = self.output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for i, item in enumerate(items):
                self._create_text_image(item, category_dir / f"{item}_{i:03d}.png", category)
                
    def _create_text_image(self, text, filepath, category):
        """Create an image with text"""
        # Create image
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fallback to default
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        
        # Get text size and center it
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (img_size[0] - text_width) // 2
        y = (img_size[1] - text_height) // 2
        
        # Choose color based on category
        colors = {
            "geometric": "blue",
            "colors": "black",
            "numbers": "green",
            "letters": "red",
            "chess_pieces": "brown",
            "programming": "purple"
        }
        
        draw.text((x, y), text, fill=colors.get(category, "black"), font=font)
        
        # Add some geometric shapes for visual interest
        if category == "geometric":
            self._add_geometric_shape(draw, img_size, text)
        
        img.save(filepath)
        
    def _add_geometric_shape(self, draw, img_size, shape_name):
        """Add geometric shapes to images"""
        margin = 50
        
        if shape_name == "circle":
            draw.ellipse([margin, margin, img_size[0]-margin, img_size[1]-margin], 
                        outline="blue", width=3)
        elif shape_name == "square":
            draw.rectangle([margin, margin, img_size[0]-margin, img_size[1]-margin], 
                          outline="blue", width=3)
        elif shape_name == "triangle":
            points = [
                (img_size[0]//2, margin),
                (margin, img_size[1]-margin),
                (img_size[0]-margin, img_size[1]-margin)
            ]
            draw.polygon(points, outline="blue", width=3)
            
    def create_pattern_images(self):
        """Create pattern-based images"""
        logger.info("Creating pattern images...")
        
        pattern_dir = self.output_dir / "patterns"
        pattern_dir.mkdir(exist_ok=True)
        
        patterns = [
            "stripes_horizontal",
            "stripes_vertical", 
            "checkerboard",
            "dots",
            "grid"
        ]
        
        for i, pattern in enumerate(patterns):
            self._create_pattern_image(pattern, pattern_dir / f"{pattern}_{i:03d}.png")
            
    def _create_pattern_image(self, pattern_type, filepath):
        """Create pattern images"""
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        if pattern_type == "stripes_horizontal":
            for y in range(0, img_size[1], 20):
                if (y // 20) % 2 == 0:
                    draw.rectangle([0, y, img_size[0], y+20], fill="lightblue")
                    
        elif pattern_type == "stripes_vertical":
            for x in range(0, img_size[0], 20):
                if (x // 20) % 2 == 0:
                    draw.rectangle([x, 0, x+20, img_size[1]], fill="lightgreen")
                    
        elif pattern_type == "checkerboard":
            square_size = 28
            for x in range(0, img_size[0], square_size):
                for y in range(0, img_size[1], square_size):
                    if ((x // square_size) + (y // square_size)) % 2 == 0:
                        draw.rectangle([x, y, x+square_size, y+square_size], fill="lightgray")
                        
        elif pattern_type == "dots":
            for x in range(20, img_size[0], 40):
                for y in range(20, img_size[1], 40):
                    draw.ellipse([x-10, y-10, x+10, y+10], fill="red")
                    
        elif pattern_type == "grid":
            for x in range(0, img_size[0], 30):
                draw.line([x, 0, x, img_size[1]], fill="black", width=2)
            for y in range(0, img_size[1], 30):
                draw.line([0, y, img_size[0], y], fill="black", width=2)
        
        img.save(filepath)
        
    def create_chart_images(self):
        """Create simple chart/diagram images"""
        logger.info("Creating chart images...")
        
        chart_dir = self.output_dir / "charts"
        chart_dir.mkdir(exist_ok=True)
        
        # Simple bar chart
        self._create_bar_chart(chart_dir / "bar_chart_001.png")
        
        # Simple pie chart representation
        self._create_pie_chart(chart_dir / "pie_chart_001.png")
        
    def _create_bar_chart(self, filepath):
        """Create a simple bar chart image"""
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw bars
        bar_width = 30
        bar_heights = [50, 80, 120, 90, 110]
        colors = ["red", "blue", "green", "yellow", "purple"]
        
        start_x = 30
        for i, (height, color) in enumerate(zip(bar_heights, colors)):
            x = start_x + i * (bar_width + 10)
            y = img_size[1] - 30 - height
            draw.rectangle([x, y, x + bar_width, img_size[1] - 30], fill=color)
            
        img.save(filepath)
        
    def _create_pie_chart(self, filepath):
        """Create a simple pie chart representation"""
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw pie slices as rectangles (simplified)
        center = (img_size[0] // 2, img_size[1] // 2)
        radius = 80
        
        colors = ["red", "blue", "green", "yellow"]
        angles = [0, 90, 180, 270]
        
        for i, (color, angle) in enumerate(zip(colors, angles)):
            # Simple representation with rectangles
            if i == 0:
                draw.rectangle([center[0], center[1]-radius//2, center[0]+radius, center[1]+radius//2], fill=color)
            elif i == 1:
                draw.rectangle([center[0]-radius//2, center[1], center[0]+radius//2, center[1]+radius], fill=color)
            elif i == 2:
                draw.rectangle([center[0]-radius, center[1]-radius//2, center[0], center[1]+radius//2], fill=color)
            else:
                draw.rectangle([center[0]-radius//2, center[1]-radius, center[0]+radius//2, center[1]], fill=color)
                
        img.save(filepath)
        
    def create_metadata(self):
        """Create metadata file for the images"""
        logger.info("Creating metadata...")
        
        metadata = {
            "dataset_name": "Safe Multimodal Training Dataset",
            "description": "Synthetic images for multimodal AI training - safe for public repositories",
            "categories": {
                "geometric": "Basic geometric shapes and forms",
                "colors": "Color names and representations", 
                "numbers": "Numeric digits 0-9",
                "letters": "Alphabetic characters A-J",
                "chess_pieces": "Chess piece symbols",
                "programming": "Programming keywords",
                "patterns": "Visual patterns and textures",
                "charts": "Simple charts and diagrams"
            },
            "image_format": "PNG",
            "image_size": "224x224",
            "total_images": 0,
            "license": "Public Domain - Created synthetically for AI training"
        }
        
        # Count total images
        total = 0
        for category_dir in self.output_dir.iterdir():
            if category_dir.is_dir():
                total += len(list(category_dir.glob("*.png")))
        
        metadata["total_images"] = total
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def download_public_domain_images(self):
        """Download safe public domain images from various sources"""
        logger.info("Downloading public domain images...")
        
        # Create download directory
        download_dir = self.output_dir / "downloaded"
        download_dir.mkdir(exist_ok=True)
        
        # Safe image sources with public domain content
        image_sources = [
            # Unsplash (requires API key but has free tier)
            {
                "name": "nature",
                "urls": [
                    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=224&h=224&fit=crop",  # Mountain
                    "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=224&h=224&fit=crop",  # Forest
                    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=224&h=224&fit=crop",  # Lake
                ]
            },
            # Pixabay-style URLs (these are example patterns)
            {
                "name": "objects", 
                "urls": [
                    "https://cdn.pixabay.com/photo/2016/03/27/18/10/bear-1283347_960_720.jpg",
                    "https://cdn.pixabay.com/photo/2017/05/08/13/15/spring-bird-2295434_960_720.jpg",
                ]
            }
        ]
        
        # Instead of downloading from external sources (which may have rate limits),
        # let's create more diverse synthetic images
        self._create_nature_inspired_images(download_dir)
        self._create_object_images(download_dir)
        self._create_abstract_art_images(download_dir)
        
    def _create_nature_inspired_images(self, output_dir):
        """Create nature-inspired synthetic images"""
        nature_dir = output_dir / "nature"
        nature_dir.mkdir(exist_ok=True)
        
        # Create simple nature scenes
        scenes = [
            ("sky", "lightblue", "white"),
            ("grass", "green", "darkgreen"), 
            ("sunset", "orange", "red"),
            ("ocean", "blue", "darkblue"),
            ("forest", "darkgreen", "brown")
        ]
        
        for i, (name, color1, color2) in enumerate(scenes):
            self._create_gradient_image(name, color1, color2, nature_dir / f"{name}_{i:03d}.png")
            
    def _create_object_images(self, output_dir):
        """Create simple object representations"""
        objects_dir = output_dir / "objects"
        objects_dir.mkdir(exist_ok=True)
        
        objects = [
            ("book", "brown", (50, 50, 150, 200)),
            ("cup", "white", (80, 100, 120, 180)),
            ("ball", "red", (70, 70, 150, 150)),
            ("box", "gray", (60, 80, 160, 140)),
            ("tree", "green", (100, 50, 120, 200))
        ]
        
        for i, (name, color, rect) in enumerate(objects):
            self._create_simple_object(name, color, rect, objects_dir / f"{name}_{i:03d}.png")
            
    def _create_abstract_art_images(self, output_dir):
        """Create abstract art-style images"""
        abstract_dir = output_dir / "abstract"
        abstract_dir.mkdir(exist_ok=True)
        
        for i in range(10):
            self._create_random_abstract(abstract_dir / f"abstract_{i:03d}.png")
            
    def _create_gradient_image(self, name, color1, color2, filepath):
        """Create a gradient image"""
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Simple gradient effect
        for y in range(img_size[1]):
            ratio = y / img_size[1]
            # Simple color interpolation
            if color1 == "lightblue" and color2 == "white":
                r = int(173 + (255 - 173) * ratio)
                g = int(216 + (255 - 216) * ratio) 
                b = int(230 + (255 - 230) * ratio)
            elif color1 == "green" and color2 == "darkgreen":
                r = int(0 + (0 - 0) * ratio)
                g = int(128 + (100 - 128) * ratio)
                b = int(0 + (0 - 0) * ratio)
            elif color1 == "orange" and color2 == "red":
                r = int(255 + (255 - 255) * ratio)
                g = int(165 + (0 - 165) * ratio)
                b = int(0 + (0 - 0) * ratio)
            elif color1 == "blue" and color2 == "darkblue":
                r = int(0 + (0 - 0) * ratio)
                g = int(0 + (0 - 0) * ratio)
                b = int(255 + (139 - 255) * ratio)
            else:  # forest
                r = int(34 + (139 - 34) * ratio)
                g = int(139 + (69 - 139) * ratio)
                b = int(34 + (19 - 34) * ratio)
                
            draw.line([(0, y), (img_size[0], y)], fill=(r, g, b))
            
        # Add text label
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        draw.text((10, 10), name.title(), fill="black", font=font)
        img.save(filepath)
        
    def _create_simple_object(self, name, color, rect, filepath):
        """Create simple object representation"""
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw the object
        if name == "book":
            draw.rectangle(rect, fill=color, outline="black", width=2)
            # Add lines for pages
            for i in range(3):
                y = rect[1] + 20 + i * 15
                draw.line([(rect[0] + 10, y), (rect[2] - 10, y)], fill="black")
        elif name == "cup":
            draw.rectangle(rect, fill=color, outline="black", width=2)
            # Add handle
            draw.arc([rect[2], rect[1] + 20, rect[2] + 20, rect[3] - 20], 270, 90, fill="black", width=3)
        elif name == "ball":
            draw.ellipse(rect, fill=color, outline="black", width=2)
        elif name == "box":
            draw.rectangle(rect, fill=color, outline="black", width=2)
            # Add 3D effect
            draw.polygon([(rect[2], rect[1]), (rect[2] + 15, rect[1] - 15), 
                         (rect[2] + 15, rect[3] - 15), (rect[2], rect[3])], 
                        fill="darkgray", outline="black")
        elif name == "tree":
            # Tree trunk
            trunk_rect = (rect[0] + 30, rect[3] - 50, rect[0] + 40, rect[3])
            draw.rectangle(trunk_rect, fill="brown")
            # Tree top
            draw.ellipse((rect[0], rect[1], rect[2], rect[3] - 30), fill=color, outline="darkgreen")
            
        # Add label
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
            
        draw.text((10, img_size[1] - 30), name.title(), fill="black", font=font)
        img.save(filepath)
        
    def _create_random_abstract(self, filepath):
        """Create random abstract art"""
        img_size = (224, 224)
        img = Image.new('RGB', img_size, color='white')
        draw = ImageDraw.Draw(img)
        
        # Random shapes and colors
        colors = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
        
        # Draw random rectangles
        for _ in range(5):
            x1 = np.random.randint(0, img_size[0] // 2)
            y1 = np.random.randint(0, img_size[1] // 2)
            x2 = np.random.randint(x1 + 20, img_size[0])
            y2 = np.random.randint(y1 + 20, img_size[1])
            color = np.random.choice(colors)
            draw.rectangle([x1, y1, x2, y2], fill=color, outline="black")
            
        # Draw random circles
        for _ in range(3):
            x = np.random.randint(20, img_size[0] - 20)
            y = np.random.randint(20, img_size[1] - 20)
            r = np.random.randint(10, 30)
            color = np.random.choice(colors)
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline="black")
            
        img.save(filepath)
        
    def create_educational_images(self):
        """Create educational/academic images"""
        logger.info("Creating educational images...")
        
        edu_dir = self.output_dir / "educational"
        edu_dir.mkdir(exist_ok=True)
        
        # Math symbols
        math_symbols = ["∑", "∫", "π", "∞", "√", "±", "≤", "≥", "≠", "≈"]
        for i, symbol in enumerate(math_symbols):
            self._create_text_image(symbol, edu_dir / f"math_{symbol}_{i:03d}.png", "math")
            
        # Science symbols  
        science_symbols = ["⚛", "🔬", "🧪", "⚗", "🔭", "🌡", "⚡", "🌍", "🌙", "⭐"]
        for i, symbol in enumerate(science_symbols):
            self._create_text_image(symbol, edu_dir / f"science_{i:03d}.png", "science")
            
    def create_dataset(self):
        """Create the complete safe image dataset"""
        logger.info("Creating comprehensive safe image dataset for multimodal training...")
        
        # Create synthetic images
        self.create_synthetic_images()
        self.create_pattern_images() 
        self.create_chart_images()
        self.create_educational_images()
        
        # Create nature-inspired and object images
        self.download_public_domain_images()
        
        # Create metadata
        self.create_metadata()
        
        logger.info(f"✅ Comprehensive safe image dataset created in {self.output_dir}")
        logger.info("This dataset is appropriate for public GitHub repositories")
        logger.info("Includes: synthetic, patterns, charts, educational, nature-inspired, objects, and abstract art")

def main():
    """Main function"""
    creator = SafeImageDatasetCreator()
    creator.create_dataset()
    
    print("\n🎨 Safe Image Dataset Created!")
    print("📁 Location: dataset/safe_images/")
    print("📊 Categories: geometric, colors, numbers, letters, chess_pieces, programming, patterns, charts")
    print("✅ Safe for public repositories - no explicit content")
    print("🔧 Ready for multimodal AI training")

if __name__ == "__main__":
    main()