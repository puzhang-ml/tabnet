"""
Modified from the original TabNet implementation by DreamQuark:
https://github.com/dreamquark-ai/tabnet

MIT License

Copyright (c) 2019 DreamQuark
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple

from tabnet_block import TabNet

def generate_sample_data(num_samples: int = 1000) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """Generate synthetic data with clear feature group patterns.
    
    Args:
        num_samples: Number of samples to generate.
        
    Returns:
        Tuple of (features_dict, labels) where features are grouped by type.
    """
    # Generate base features that will influence both embeddings and numerics
    base = tf.random.normal((num_samples, 4))
    
    # Create features with clear prefix grouping
    features = {
        # Embedding features (should be grouped together)
        'embedding_1': tf.concat([
            base[:, :2],
            tf.random.normal((num_samples, 14))
        ], axis=1),  # 16 dims
        'embedding_2': tf.concat([
            base[:, 2:],
            tf.random.normal((num_samples, 14))
        ], axis=1),  # 16 dims
        
        # Numeric features (should be grouped together)
        'numeric_1': 0.3 * base[:, 0:1] + 0.7 * tf.random.normal((num_samples, 1)),
        'numeric_2': 0.3 * base[:, 1:2] + 0.7 * tf.random.normal((num_samples, 1)),
        
        # Categorical features (should be grouped together)
        'categorical_1': tf.concat([
            0.3 * base[:, :2],
            tf.random.normal((num_samples, 6))
        ], axis=1),  # 8 dims
        'categorical_2': tf.concat([
            0.3 * base[:, 2:],
            tf.random.normal((num_samples, 6))
        ], axis=1)   # 8 dims
    }
    
    # Generate labels based on a combination of features
    logits = (
        0.5 * tf.reduce_sum(features['embedding_1'][:, :2], axis=1) +
        0.3 * tf.reduce_sum(features['embedding_2'][:, :2], axis=1) +
        0.2 * tf.squeeze(features['numeric_1']) +
        0.1 * tf.reduce_sum(features['categorical_1'][:, :2], axis=1)
    )
    
    # Convert to binary labels
    labels = tf.cast(tf.nn.sigmoid(logits) > 0.5, tf.float32)
    
    return features, labels

def create_dataset(features: Dict[str, tf.Tensor], 
                  labels: tf.Tensor, 
                  batch_size: int,
                  shuffle: bool = True) -> tf.data.Dataset:
    """Create TensorFlow dataset from features dictionary and labels.
    
    Args:
        features: Dictionary of input features.
        labels: Target labels.
        batch_size: Batch size for training.
        shuffle: Whether to shuffle the dataset.
        
    Returns:
        TensorFlow dataset.
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10000)
    return dataset.batch(batch_size)

def main():
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Training parameters
    BATCH_SIZE = 256
    EPOCHS = 50
    N_TRAIN = 10000
    N_VALID = 2000
    LEARNING_RATE = 0.02
    EARLY_STOPPING_PATIENCE = 10
    
    print("Generating training data...")
    train_features, train_labels = generate_sample_data(N_TRAIN)
    
    print("Generating validation data...")
    valid_features, valid_labels = generate_sample_data(N_VALID)
    
    # Create datasets
    train_dataset = create_dataset(train_features, train_labels, BATCH_SIZE)
    valid_dataset = create_dataset(valid_features, valid_labels, BATCH_SIZE)
    
    # Calculate total feature dimension
    total_dims = sum(tensor.shape[-1] for tensor in train_features.values())
    print(f"\nTotal feature dimensions: {total_dims}")
    
    # Initialize model
    # Feature groups will be automatically inferred from dictionary keys
    print("\nInitializing TabNet model...")
    model = TabNet(
        feature_dim=total_dims,  # Will be used for validation
        output_dim=1,
        n_d=8,  # Width of the decision prediction layer
        n_a=8,  # Width of the attention embedding
        n_steps=3,  # Number of decision steps
        gamma=1.3,  # Feature reusage coefficient
        epsilon=1e-5,
        momentum=0.7
    )
    
    # Initialize optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Training loop
    print("\nStarting training...")
    best_valid_auc = 0
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        train_loss = 0
        train_batches = 0
        
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                # Forward pass (first pass will trigger automatic feature grouping)
                y_pred, masks, _ = model(x_batch, training=True)
                y_pred = tf.squeeze(y_pred)
                
                # Calculate loss
                loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        y_batch,
                        tf.nn.sigmoid(y_pred)
                    )
                )
            
            # Backpropagation
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss += loss.numpy()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        valid_preds = []
        valid_labels_list = []
        
        for x_batch, y_batch in valid_dataset:
            y_pred, _, _ = model(x_batch, training=False)
            y_pred = tf.nn.sigmoid(tf.squeeze(y_pred))
            valid_preds.extend(y_pred.numpy())
            valid_labels_list.extend(y_batch.numpy())
        
        valid_preds = np.array(valid_preds)
        valid_labels_list = np.array(valid_labels_list)
        valid_auc = roc_auc_score(valid_labels_list, valid_preds)
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid AUC: {valid_auc:.4f}")
        print(f"Learning Rate: {optimizer.learning_rate.numpy():.6f}")
        
        # Early stopping
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("\nEarly stopping triggered!")
                break
    
    print(f"\nTraining finished! Best validation AUC: {best_valid_auc:.4f}")

if __name__ == "__main__":
    main()