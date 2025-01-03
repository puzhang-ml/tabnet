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
    # Create correlated features
    base = tf.random.normal((num_samples, 4))
    features = {
        'embedding_1': tf.concat([base[:, :2], tf.random.normal((num_samples, 14))], axis=1),
        'embedding_2': tf.concat([base[:, 2:], tf.random.normal((num_samples, 14))], axis=1),
        'numeric_1': 0.3 * base[:, 0:1] + 0.7 * tf.random.normal((num_samples, 1)),
        'numeric_2': 0.3 * base[:, 1:2] + 0.7 * tf.random.normal((num_samples, 1)),
        'categorical_1': tf.concat([0.3 * base[:, :2], tf.random.normal((num_samples, 6))], axis=1),
        'categorical_2': tf.concat([0.3 * base[:, 2:], tf.random.normal((num_samples, 6))], axis=1)
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
    
    # Define feature groups using indices
    # embedding_1: 0-15, embedding_2: 16-31
    # numeric_1: 32, numeric_2: 33
    # categorical_1: 34-41, categorical_2: 42-49
    grouped_features = [
        list(range(0, 32)),  # embeddings
        [32, 33],  # numeric
        list(range(34, 50))  # categorical
    ]
    
    # Initialize model
    print("\nInitializing TabNet model...")
    model = TabNet(
        output_dim=1,
        num_decision_steps=5,
        feature_dim=64,  # Increased feature dimension
        relaxation_factor=1.5,
        sparsity_coefficient=1e-4,  # Increased sparsity
        bn_momentum=0.9,
        grouped_features=grouped_features
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
