#!/usr/bin/env python3
"""
Filtered Occlusion Testing for NeuroSym-CML
Tests specific anatomical features with class filtering to avoid false positives
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import json
from typing import List, Dict, Any, Tuple, Set
from simple_occlusion_test import SimpleOcclusionTester

class FilteredOcclusionTester:
    """Test model with filtered classes to focus on relevant anatomical features"""
    
    def __init__(self, checkpoint_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the secure checkpoint
        if checkpoint_path:
            self.tester = SimpleOcclusionTester(checkpoint_path=checkpoint_path)
        else:
            self.tester = SimpleOcclusionTester()
        
        self.model = self.tester.model
        
        # Define the actual class mappings
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
        
        # Define test-specific class filters
        self.breast_nipple_classes = {6: 'breast', 24: 'nipple', 29: 'pierced_nipple'}
        self.clothing_classes = {4: 'bra', 26: 'panties', 27: 'pants', 32: 'shirt', 36: 'skirt', 39: 'wearing_clothes'}
        self.safe_classes = {7: 'clothed_chest', 39: 'wearing_clothes'}
        self.body_parts = {0: 'abdomen', 1: 'arm', 6: 'breast', 14: 'face', 20: 'leg', 35: 'shoulder', 41: 'wrist'}
        
        # Classes to exclude (likely false positives for this test)
        self.exclude_classes = {
            2: 'ass', 3: 'black_penis', 8: 'doggystyle', 10: 'exposing_body', 11: 'exposing_chest',
            12: 'exposing_pussy', 16: 'giving_blowjob', 18: 'interracial_blowjob', 19: 'interracial_sex',
            22: 'missionary', 28: 'penis', 30: 'pussy', 31: 'sex', 40: 'white_penis'
        }
        
        print(f"✅ Filtered tester initialized")
        print(f"🎯 Focus classes: {list(self.breast_nipple_classes.values())}")
        print(f"🎯 Clothing classes: {list(self.clothing_classes.values())}")
        print(f"🚫 Excluded classes: {list(self.exclude_classes.values())}")
    
    def predict_filtered(self, image_path: str, focus_classes: Set[int] = None, 
                        exclude_classes: Set[int] = None, threshold: float = 0.01) -> Dict[str, Any]:
        """Get filtered predictions focusing on relevant classes"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.tester.transform(image).unsqueeze(0).to(self.device)
            
            # Create dummy inputs for the model
            batch_size = input_tensor.size(0)
            text_embeddings = torch.zeros(batch_size, 512).to(self.device)
            rule_indices = torch.zeros(batch_size, dtype=torch.long).to(self.device)
            
            # Get model predictions
            with torch.no_grad():
                self.model.eval()
                outputs = self.model(input_tensor, text_embeddings, rule_indices)
                probabilities = F.softmax(outputs, dim=1)
            
            # Apply filters
            if focus_classes is None:
                focus_classes = set(self.breast_nipple_classes.keys()) | set(self.clothing_classes.keys()) | set(self.body_parts.keys())
            
            if exclude_classes is None:
                exclude_classes = set(self.exclude_classes.keys())
            
            # Get filtered predictions
            probs = probabilities[0].cpu().numpy()
            predictions = []
            
            for class_id, prob in enumerate(probs):
                # Skip excluded classes
                if class_id in exclude_classes:
                    continue
                
                # Only include focus classes if specified
                if focus_classes and class_id not in focus_classes:
                    continue
                
                if prob > threshold:
                    predictions.append({
                        'class_id': class_id,
                        'class_name': self.class_mapping.get(class_id, f'class_{class_id}'),
                        'confidence': float(prob),
                        'is_breast_nipple': class_id in self.breast_nipple_classes,
                        'is_clothing': class_id in self.clothing_classes,
                        'is_safe': class_id in self.safe_classes,
                        'is_body_part': class_id in self.body_parts
                    })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                'image_path': image_path,
                'predictions': predictions,
                'breast_nipple_detected': [p for p in predictions if p['is_breast_nipple']],
                'clothing_detected': [p for p in predictions if p['is_clothing']],
                'safe_detected': [p for p in predictions if p['is_safe']],
                'body_parts_detected': [p for p in predictions if p['is_body_part']],
                'top_prediction': predictions[0] if predictions else None
            }
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None
    
    def test_breast_nipple_occlusion(self, ground_zero_path: str, occluded_path: str) -> Dict[str, Any]:
        """Test specifically for breast/nipple detection through clothing"""
        print(f"\n🔍 Breast/Nipple occlusion test: {os.path.basename(ground_zero_path)} vs {os.path.basename(occluded_path)}")
        
        # Focus only on breast/nipple and clothing classes
        focus_classes = set(self.breast_nipple_classes.keys()) | set(self.clothing_classes.keys()) | set(self.safe_classes.keys())
        
        # Get filtered predictions
        gz_result = self.predict_filtered(ground_zero_path, focus_classes=focus_classes)
        occ_result = self.predict_filtered(occluded_path, focus_classes=focus_classes)
        
        if not gz_result or not occ_result:
            return None
        
        # Analyze breast/nipple detection
        analysis = {
            'ground_zero': gz_result,
            'occluded': occ_result,
            'breast_nipple_analysis': self._analyze_breast_nipple_detection(gz_result, occ_result)
        }
        
        return analysis
    
    def _analyze_breast_nipple_detection(self, gz_result: Dict, occ_result: Dict) -> Dict[str, Any]:
        """Analyze breast/nipple detection specifically"""
        
        gz_breast_nipple = gz_result['breast_nipple_detected']
        occ_breast_nipple = occ_result['breast_nipple_detected']
        gz_clothing = gz_result['clothing_detected']
        occ_clothing = occ_result['clothing_detected']
        
        # Check if model detects breast/nipple through clothing
        hidden_detection = len(occ_breast_nipple) > 0
        clothing_detection = len(occ_clothing) > 0
        
        # Calculate confidence scores
        gz_confidence = max([p['confidence'] for p in gz_breast_nipple], default=0.0)
        occ_confidence = max([p['confidence'] for p in occ_breast_nipple], default=0.0)
        
        # Determine if model is truly inferring hidden features
        if hidden_detection and clothing_detection:
            inference_quality = "EXCELLENT"
            score = 1.0
        elif hidden_detection:
            inference_quality = "GOOD"
            score = 0.7
        elif clothing_detection:
            inference_quality = "PARTIAL"
            score = 0.4
        else:
            inference_quality = "POOR"
            score = 0.0
        
        return {
            'hidden_detection': hidden_detection,
            'clothing_detection': clothing_detection,
            'gz_breast_nipple': [p['class_name'] for p in gz_breast_nipple],
            'occ_breast_nipple': [p['class_name'] for p in occ_breast_nipple],
            'gz_clothing': [p['class_name'] for p in gz_clothing],
            'occ_clothing': [p['class_name'] for p in occ_clothing],
            'gz_confidence': gz_confidence,
            'occ_confidence': occ_confidence,
            'inference_quality': inference_quality,
            'inference_score': score,
            'analysis': self._get_detailed_analysis(gz_result, occ_result, hidden_detection, clothing_detection)
        }
    
    def _get_detailed_analysis(self, gz_result: Dict, occ_result: Dict, 
                              hidden_detection: bool, clothing_detection: bool) -> str:
        """Get detailed analysis of the detection results"""
        
        if hidden_detection and clothing_detection:
            return "✅ EXCELLENT: Model detects both clothing AND hidden breast/nipple features through occlusion"
        elif hidden_detection:
            return "✅ GOOD: Model detects hidden breast/nipple features through clothing"
        elif clothing_detection:
            return "⚠️ PARTIAL: Model detects clothing but not hidden anatomical features"
        else:
            return "❌ POOR: Model fails to detect relevant features"
    
    def visualize_filtered_results(self, analysis: Dict, save_path: str = None):
        """Create visualization of filtered results"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Filtered Breast/Nipple Occlusion Analysis', fontsize=16, fontweight='bold')
        
        gz_result = analysis['ground_zero']
        occ_result = analysis['occluded']
        breast_analysis = analysis['breast_nipple_analysis']
        
        # Load images
        gz_img = Image.open(gz_result['image_path']).convert('RGB')
        occ_img = Image.open(occ_result['image_path']).convert('RGB')
        
        # Ground zero image
        axes[0, 0].imshow(gz_img)
        gz_text = f"Ground Zero\n"
        if gz_result['breast_nipple_detected']:
            gz_text += f"Breast/Nipple: {', '.join([p['class_name'] for p in gz_result['breast_nipple_detected']])}\n"
        if gz_result['clothing_detected']:
            gz_text += f"Clothing: {', '.join([p['class_name'] for p in gz_result['clothing_detected']])}"
        axes[0, 0].set_title(gz_text)
        axes[0, 0].axis('off')
        
        # Occluded image
        axes[0, 1].imshow(occ_img)
        occ_text = f"Occluded\n"
        if occ_result['breast_nipple_detected']:
            occ_text += f"Breast/Nipple: {', '.join([p['class_name'] for p in occ_result['breast_nipple_detected']])}\n"
        if occ_result['clothing_detected']:
            occ_text += f"Clothing: {', '.join([p['class_name'] for p in occ_result['clothing_detected']])}"
        axes[0, 1].set_title(occ_text)
        axes[0, 1].axis('off')
        
        # Confidence comparison
        all_classes = set()
        for p in gz_result['predictions']:
            all_classes.add(p['class_name'])
        for p in occ_result['predictions']:
            all_classes.add(p['class_name'])
        
        gz_scores = {p['class_name']: p['confidence'] for p in gz_result['predictions']}
        occ_scores = {p['class_name']: p['confidence'] for p in occ_result['predictions']}
        
        classes = list(all_classes)[:8]  # Show top 8 classes
        gz_values = [gz_scores.get(c, 0) for c in classes]
        occ_values = [occ_scores.get(c, 0) for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[1, 0].bar(x - width/2, gz_values, width, label='Ground Zero', alpha=0.8, color='skyblue')
        axes[1, 0].bar(x + width/2, occ_values, width, label='Occluded', alpha=0.8, color='lightcoral')
        axes[1, 0].set_xlabel('Classes')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_title('Filtered Confidence Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Analysis summary
        summary_text = f"""
        Filtered Analysis:
        
        Hidden Detection: {breast_analysis['hidden_detection']}
        Clothing Detection: {breast_analysis['clothing_detection']}
        Inference Quality: {breast_analysis['inference_quality']}
        Inference Score: {breast_analysis['inference_score']:.2f}
        
        Ground Zero:
        • Breast/Nipple: {', '.join(breast_analysis['gz_breast_nipple'])}
        • Clothing: {', '.join(breast_analysis['gz_clothing'])}
        
        Occluded:
        • Breast/Nipple: {', '.join(breast_analysis['occ_breast_nipple'])}
        • Clothing: {', '.join(breast_analysis['occ_clothing'])}
        
        Analysis:
        {breast_analysis['analysis']}
        """
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Filtered visualization saved: {save_path}")
        
        plt.show()
    
    def run_filtered_tests(self, test_pairs: List[Tuple[str, str]], output_dir: str = "filtered_results"):
        """Run filtered tests on all image pairs"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🧪 Running filtered breast/nipple occlusion tests on {len(test_pairs)} pairs...")
        
        results = []
        
        for i, (gz_path, occ_path) in enumerate(test_pairs):
            print(f"\n--- Filtered Test {i+1} ---")
            
            analysis = self.test_breast_nipple_occlusion(gz_path, occ_path)
            if analysis:
                results.append(analysis)
                
                # Save visualization
                save_path = os.path.join(output_dir, f"filtered_test_{i+1:03d}.png")
                self.visualize_filtered_results(analysis, save_path)
                
                # Print summary
                breast_analysis = analysis['breast_nipple_analysis']
                print(f"✅ Inference quality: {breast_analysis['inference_quality']}")
                print(f"📊 Analysis: {breast_analysis['analysis']}")
        
        # Generate summary report
        self._generate_filtered_summary(results, output_dir)
        
        return results
    
    def _generate_filtered_summary(self, results: List[Dict], output_dir: str):
        """Generate summary of filtered test results"""
        if not results:
            return
        
        # Calculate statistics
        excellent_count = sum(1 for r in results if r['breast_nipple_analysis']['inference_quality'] == 'EXCELLENT')
        good_count = sum(1 for r in results if r['breast_nipple_analysis']['inference_quality'] == 'GOOD')
        hidden_detections = sum(1 for r in results if r['breast_nipple_analysis']['hidden_detection'])
        clothing_detections = sum(1 for r in results if r['breast_nipple_analysis']['clothing_detection'])
        
        print(f"\n📊 FILTERED TEST SUMMARY")
        print(f"=" * 50)
        print(f"Total tests: {len(results)}")
        print(f"Excellent inference: {excellent_count}/{len(results)} ({excellent_count/len(results)*100:.1f}%)")
        print(f"Good inference: {good_count}/{len(results)} ({good_count/len(results)*100:.1f}%)")
        print(f"Hidden feature detections: {hidden_detections}/{len(results)} ({hidden_detections/len(results)*100:.1f}%)")
        print(f"Clothing detections: {clothing_detections}/{len(results)} ({clothing_detections/len(results)*100:.1f}%)")
        
        # Save detailed results
        with open(os.path.join(output_dir, "filtered_results.json"), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📄 Detailed results saved: {os.path.join(output_dir, 'filtered_results.json')}")

def main():
    """Main function for filtered occlusion testing"""
    print("🔍 Filtered Breast/Nipple Occlusion Testing")
    print("=" * 60)
    
    # Initialize tester
    tester = FilteredOcclusionTester()
    
    # Define test image pairs
    ground_zero_dir = "occlusion_test/ground_zero"
    occlusion_dir = "occlusion_test/occlusion_img"
    
    test_pairs = []
    
    # Create all possible combinations
    gz_images = [f for f in os.listdir(ground_zero_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    occ_images = [f for f in os.listdir(occlusion_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for gz_img in gz_images:
        for occ_img in occ_images:
            gz_path = os.path.join(ground_zero_dir, gz_img)
            occ_path = os.path.join(occlusion_dir, occ_img)
            test_pairs.append((gz_path, occ_path))
    
    print(f"📁 Found {len(gz_images)} ground zero images")
    print(f"📁 Found {len(occ_images)} occlusion images")
    print(f"🧪 Created {len(test_pairs)} test pairs")
    
    if not test_pairs:
        print("❌ No test images found!")
        return
    
    # Run filtered tests
    results = tester.run_filtered_tests(test_pairs, "filtered_results")
    
    print(f"\n✅ Filtered testing completed!")
    print(f"🎯 This test focuses specifically on:")
    print(f"   - Breast/nipple detection through clothing")
    print(f"   - Clothing detection (bra, shirt, etc.)")
    print(f"   - Excludes irrelevant classes (penis, sex acts, etc.)")
    print(f"   - Tests true hidden feature inference")

if __name__ == "__main__":
    main()
