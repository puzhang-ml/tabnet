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

def generate_sample_data(num_samples=256):
    """Generate sample data with meaningful patterns."""
    # Create correlated features with clear grouping structure
    base = tf.random.normal((num_samples, 4))
    
    # Create features with clear prefix grouping
    features = {
        'embedding_1': tf.concat([base[:, :2], tf.random.normal((num_samples, 14))], axis=1),  # 16 dims
        'embedding_2': tf.concat([base[:, 2:], tf.random.normal((num_samples, 14))], axis=1),  # 16 dims
        'numeric_1': 0.3 * base[:, 0:1] + 0.7 * tf.random.normal((num_samples, 1)),  # 1 dim
        'numeric_2': 0.3 * base[:, 1:2] + 0.7 * tf.random.normal((num_samples, 1)),  # 1 dim
        'categorical_1': tf.concat([0.3 * base[:, :2], tf.random.normal((num_samples, 6))], axis=1),  # 8 dims
        'categorical_2': tf.concat([0.3 * base[:, 2:], tf.random.normal((num_samples, 6))], axis=1)   # 8 dims
    }
    
    # Generate labels based on a combination of features
    logits = (
        0.5 * tf.reduce_sum(features['embedding_1'][:, :2], axis=1) +
        0.3 * tf.reduce_sum(features['embedding_2'][:, :2], axis=1) +
        0.2 * tf.squeeze(features['numeric_1']) +
        0.1 * tf.reduce_sum(features['categorical_1'][:, :2], axis=1)
    )
    probs = tf.nn.sigmoid(logits)
    labels = tf.cast(probs > 0.5, tf.float32)
    
    return features, labels

def create_dataset(features, labels, batch_size):
    """Create TensorFlow dataset from features dictionary and labels."""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(buffer_size=10000).batch(batch_size)

def main():
    # Parameters
    BATCH_SIZE = 1024
    EPOCHS = 50
    N_SAMPLES = 10000
    LEARNING_RATE = 0.01
    
    print("Generating data...")
    train_features, train_labels = generate_sample_data(N_SAMPLES)
    valid_features, valid_labels = generate_sample_data(N_SAMPLES // 5)
    
    # Create datasets
    train_dataset = create_dataset(train_features, train_labels, BATCH_SIZE)
    valid_dataset = create_dataset(valid_features, valid_labels, BATCH_SIZE)
    
    # Calculate total feature dimension
    total_dims = sum(tensor.shape[-1] for tensor in train_features.values())
    
    # Initialize model without explicit grouped_features
    # The model will infer groups automatically from feature names
    print("\nInitializing TabNet model...")
    model = TabNet(
        feature_dim=total_dims,
        output_dim=1,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        epsilon=1e-5,
        momentum=0.7
    )
    
    # Initialize optimizer with learning rate schedule
    initial_learning_rate = LEARNING_RATE
    decay_steps = 1000
    decay_rate = 0.9
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Training loop
    print("\nStarting training...")
    best_valid_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        train_loss = 0
        train_batches = 0
        for x_batch, y_batch in train_dataset:
            with tf.GradientTape() as tape:
                # First forward pass will trigger automatic feature grouping
                outputs = model(x_batch, training=True)
                y_pred = outputs[0] if isinstance(outputs, tuple) else outputs
                y_pred = tf.squeeze(y_pred)
                
                # Calculate loss
                bce_loss = tf.reduce_mean(
                    tf.keras.losses.binary_crossentropy(y_batch, y_pred)
                )
                sparsity_loss = outputs[2] if isinstance(outputs, tuple) else 0.0
                loss = bce_loss + sparsity_loss
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_loss += loss.numpy()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        valid_preds = []
        valid_labels_list = []
        for x_batch, y_batch in valid_dataset:
            outputs = model(x_batch, training=False)
            y_pred = outputs[0] if isinstance(outputs, tuple) else outputs
            y_pred = tf.squeeze(y_pred)
            valid_preds.extend(y_pred.numpy())
            valid_labels_list.extend(y_batch.numpy())
        
        valid_preds = np.array(valid_preds)
        valid_labels_list = np.array(valid_labels_list)
        valid_auc = roc_auc_score(valid_labels_list, valid_preds)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid AUC: {valid_auc:.4f}")
        print(f"Learning Rate: {lr_schedule(optimizer.iterations).numpy():.6f}")
        
        # Early stopping
        if valid_auc > best_valid_auc:
            best_valid_auc = valid_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    print(f"\nTraining finished! Best validation AUC: {best_valid_auc:.4f}")

if __name__ == "__main__":
    main()