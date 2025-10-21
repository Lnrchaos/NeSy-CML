import torch
from typing import Dict, Any, Optional
from meta_model import HybridModel
from data_module import DataLoader

def evaluate(model: HybridModel, dataloader: DataLoader, criterion: Optional[torch.nn.Module] = None) -> Dict[str, float]:
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The HybridModel to evaluate
        dataloader: DataLoader containing the evaluation data
        criterion: Loss function to use (defaults to CrossEntropyLoss)
        
    Returns:
        Dict containing evaluation metrics:
        - 'total_loss': Total loss across all batches
        - 'total_accuracy': Overall accuracy
        - 'neural_loss': Loss from neural component
        - 'neural_accuracy': Accuracy from neural component
        - 'symbolic_loss': Loss from symbolic component
        - 'symbolic_accuracy': Accuracy from symbolic component
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    device = next(model.parameters()).device
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    neural_loss = 0
    neural_correct = 0
    
    symbolic_loss = 0
    symbolic_correct = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get predictions from both components
            neural_outputs, symbolic_outputs = model(inputs)
            
            # Calculate losses
            neural_batch_loss = criterion(neural_outputs, targets)
            symbolic_batch_loss = criterion(symbolic_outputs, targets)
            
            # Update loss metrics
            neural_loss += neural_batch_loss.item()
            symbolic_loss += symbolic_batch_loss.item()
            total_loss += (neural_batch_loss.item() + symbolic_batch_loss.item()) / 2
            
            # Calculate accuracies
            neural_preds = torch.argmax(neural_outputs, dim=1)
            symbolic_preds = torch.argmax(symbolic_outputs, dim=1)
            
            neural_batch_correct = (neural_preds == targets).sum().item()
            symbolic_batch_correct = (symbolic_preds == targets).sum().item()
            
            # Update accuracy metrics
            neural_correct += neural_batch_correct
            symbolic_correct += symbolic_batch_correct
            total_correct += (neural_batch_correct + symbolic_batch_correct) // 2
            
            total_samples += targets.size(0)
    
    # Calculate final metrics
    total_accuracy = total_correct / total_samples
    neural_accuracy = neural_correct / total_samples
    symbolic_accuracy = symbolic_correct / total_samples
    
    # Normalize losses
    total_loss /= len(dataloader)
    neural_loss /= len(dataloader)
    symbolic_loss /= len(dataloader)
    
    # Log results
    print(f"\nEvaluation Results:")
    print(f"Total Loss: {total_loss:.4f}")
    print(f"Total Accuracy: {total_accuracy:.4f}")
    print(f"\nNeural Component:")
    print(f"Loss: {neural_loss:.4f}")
    print(f"Accuracy: {neural_accuracy:.4f}")
    print(f"\nSymbolic Component:")
    print(f"Loss: {symbolic_loss:.4f}")
    print(f"Accuracy: {symbolic_accuracy:.4f}")
    
    return {
        'total_loss': total_loss,
        'total_accuracy': total_accuracy,
        'neural_loss': neural_loss,
        'neural_accuracy': neural_accuracy,
        'symbolic_loss': symbolic_loss,
        'symbolic_accuracy': symbolic_accuracy
    }

