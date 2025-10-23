#!/usr/bin/env python3
"""
Debug script to check tensor shapes in poetry training
"""

import torch
from modular_symbolic_controller import create_symbolic_controller

def debug_shapes():
    print("🔍 Debugging Poetry Training Tensor Shapes")
    print("=" * 50)
    
    # Test parameters
    batch_size = 6
    max_length = 256
    input_size = 256
    hidden_size = 32
    num_rules = 50
    num_fuzzy_sets = 5
    
    print(f"Batch size: {batch_size}")
    print(f"Max length (text encoding): {max_length}")
    print(f"Input size for controller: {input_size}")
    print(f"Hidden size: {hidden_size}")
    print(f"Num rules: {num_rules}")
    print(f"Num fuzzy sets: {num_fuzzy_sets}")
    
    # Create test text encodings (simulating PoetryDataset output)
    text_encodings = torch.randint(0, 1000, (batch_size, max_length), dtype=torch.long)
    text_encodings_float = text_encodings.float()
    
    print(f"\nText encodings shape: {text_encodings.shape}")
    print(f"Text encodings float shape: {text_encodings_float.shape}")
    
    # Create symbolic controller
    try:
        symbolic_controller = create_symbolic_controller(
            controller_type='fuzzy_logic',
            num_rules=num_rules,
            input_size=input_size,
            hidden_size=hidden_size,
            num_fuzzy_sets=num_fuzzy_sets
        )
        print(f"✅ Symbolic controller created successfully")
        print(f"Controller type: {symbolic_controller.get_controller_type()}")
        
        # Test forward pass
        print(f"\n🧪 Testing forward pass...")
        rule_indices, symbolic_state = symbolic_controller(text_encodings_float)
        print(f"✅ Forward pass successful!")
        print(f"Rule indices shape: {rule_indices.shape}")
        print(f"Symbolic state keys: {list(symbolic_state.keys())}")
        
    except Exception as e:
        print(f"❌ Error in symbolic controller: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Let's check the controller internals
        try:
            symbolic_controller = create_symbolic_controller(
                controller_type='fuzzy_logic',
                num_rules=num_rules,
                input_size=input_size,
                hidden_size=hidden_size,
                num_fuzzy_sets=num_fuzzy_sets
            )
            
            print(f"\n🔍 Controller internals:")
            print(f"Fuzzifier: {symbolic_controller.fuzzifier}")
            
            # Check first layer of fuzzifier
            first_layer = symbolic_controller.fuzzifier[0]
            print(f"First layer input size: {first_layer.in_features}")
            print(f"First layer output size: {first_layer.out_features}")
            
        except Exception as e2:
            print(f"❌ Error creating controller: {e2}")

if __name__ == "__main__":
    debug_shapes()