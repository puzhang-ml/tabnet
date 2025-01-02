import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import List, Dict
import json

from tabnet_block import TabNet

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def generate_sample_data(batch_size: int = 8192) -> List[tf.Tensor]:
    """Generate sample preprocessed data as a list of tensors.
    Total input dimension is 900, split across 300 features (3 dims each)"""
    features = []
    for _ in range(300):
        # Each feature has 3 dimensions
        features.append(tf.random.normal((batch_size, 3)))
    return features

def main():
    # Configuration for large-scale TabNet
    # Parameters adjusted for 900-dimensional input with 300 features
    config = {
        "feature_dim": 256,    # Increased feature dimension for complex data
        "output_dim": 1,       # Binary classification
        "num_decision_steps": 5,  # Increased steps for more complex feature interactions
        "relaxation_factor": 1.5,
        "sparsity_coefficient": 1e-4,  # Adjusted for larger feature space
        "virtual_batch_size": 512,  # Virtual batch size for ghost batch norm
        "num_groups": 2,      # Increased groups for better feature grouping
        "epsilon": 1e-5
    }
    
    # Generate sample data
    print("Generating sample data...")
    sample_features = generate_sample_data()
    
    # Print feature shapes
    print("\nFeature shapes:")
    print(f"Number of features: {len(sample_features)}")
    print(f"Each feature shape: {sample_features[0].shape}")
    print(f"Total input dimension: {len(sample_features) * sample_features[0].shape[-1]}")
    print(f"Batch size: {sample_features[0].shape[0]}")
    
    # Initialize model
    print("\nInitializing TabNet model...")
    
    # Create input layers
    input_layers = []
    for i in range(300):
        input_layers.append(tf.keras.layers.Input(shape=(3,), name=f'feature_{i}'))
    
    # Initialize TabNet model
    model = TabNet(
        feature_dim=config["feature_dim"],
        output_dim=config["output_dim"],
        num_decision_steps=config["num_decision_steps"],
        relaxation_factor=config["relaxation_factor"],
        sparsity_coefficient=config["sparsity_coefficient"],
        virtual_batch_size=config["virtual_batch_size"],
        num_groups=config["num_groups"],
        epsilon=config["epsilon"]
    )
    
    # Build model
    outputs = model(input_layers)
    full_model = tf.keras.Model(inputs=input_layers, outputs=outputs, name='TabNet')
    
    # Print model summary
    print("\nModel Architecture:")
    full_model.summary()
    
    # Test forward pass
    print("\nPerforming forward pass...")
    try:
        output = full_model(sample_features, training=True)
        print("Output shape:", output.shape)
        print("Forward pass successful!")
    except Exception as e:
        print("Error during forward pass:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

if __name__ == "__main__":
    main()
