import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import Dict
import json

from tabnet_block import TabNet, create_feature_config

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def generate_sample_data(batch_size: int = 32) -> Dict[str, tf.Tensor]:
    """Generate sample preprocessed data similar to preprocessor_keyed_out"""
    return {
        'numeric_feature1': tf.random.normal((batch_size, 1)),
        'numeric_feature2': tf.random.normal((batch_size, 1)),
        'embedding1': tf.random.normal((batch_size, 8)),
        'embedding2': tf.random.normal((batch_size, 16)),
        'categorical_embedding': tf.random.normal((batch_size, 12))
    }

def main():
    # Sample configuration
    config = {
        "tabnet_feature_dim": 512,
        "tabnet_output_dim": 64,
        "tabnet_num_decision_steps": 1,
        "tabnet_relaxation_factor": 1.5,
        "tabnet_batch_momentum": 0.98,
        "tabnet_batch_size": 128
    }
    
    # Generate sample data
    sample_data = generate_sample_data()
    
    # Create feature configuration
    feature_config = create_feature_config(sample_data)
    print("\nFeature Configuration:")
    print(json.dumps(feature_config, indent=2))
    
    # Initialize model
    model = TabNet(
        feature_config=feature_config,
        feature_dim=config.get("tabnet_feature_dim", 512),
        output_dim=config.get("tabnet_output_dim", 64),
        n_steps=config.get("tabnet_num_decision_steps", 1),
        gamma=config.get("tabnet_relaxation_factor", 1.5),
        momentum=config.get("tabnet_batch_momentum", 0.98),
        virtual_batch_size=config.get("tabnet_batch_size", 128)
    )
    
    # Test forward pass
    output = model(sample_data, training=True)
    print("\nOutput shape:", output.shape)

if __name__ == "__main__":
    main()
