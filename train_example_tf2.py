import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import List
import json

from tabnet_block import TabNet

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def generate_sample_data(batch_size: int = 32) -> List[tf.Tensor]:
    """Generate sample preprocessed data as a list of tensors"""
    return [
        tf.random.normal((batch_size, 1)),  # numeric_feature1
        tf.random.normal((batch_size, 1)),  # numeric_feature2
        tf.random.normal((batch_size, 8)),  # embedding1
        tf.random.normal((batch_size, 16)), # embedding2
        tf.random.normal((batch_size, 12))  # categorical_embedding
    ]

def main():
    # Sample configuration
    config = {
        "tabnet_feature_dim": 64,    # Reduced from 512 for this example
        "tabnet_output_dim": 1,      # Binary classification
        "tabnet_num_decision_steps": 3,
        "tabnet_relaxation_factor": 1.5,
        "tabnet_batch_momentum": 0.98,
        "tabnet_batch_size": 128
    }
    
    # Generate sample data
    print("Generating sample data...")
    sample_features = generate_sample_data()
    
    # Print feature shapes
    print("\nFeature shapes:")
    for i, feat in enumerate(sample_features):
        print(f"Feature {i}: {feat.shape}")
    
    # Initialize model
    print("\nInitializing TabNet model...")
    model = TabNet(
        feature_dim=config["tabnet_feature_dim"],
        output_dim=config["tabnet_output_dim"],
        n_steps=config["tabnet_num_decision_steps"],
        gamma=config["tabnet_relaxation_factor"],
        momentum=config["tabnet_batch_momentum"],
        virtual_batch_size=config["tabnet_batch_size"]
    )
    
    # Test forward pass
    print("\nPerforming forward pass...")
    try:
        output = model(sample_features, training=True)
        print("Output shape:", output.shape)
        print("Forward pass successful!")
    except Exception as e:
        print("Error during forward pass:")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        raise

if __name__ == "__main__":
    main()
