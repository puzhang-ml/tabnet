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

def create_dataset(features: List[tf.Tensor], batch_size: int) -> tf.data.Dataset:
    """Create TensorFlow dataset from features list"""
    # Generate random binary labels for this example
    labels = tf.cast(tf.random.uniform((features[0].shape[0],)) > 0.5, tf.float32)
    
    # Convert list of features to a dictionary
    features_dict = {f'feature_{i}': tensor for i, tensor in enumerate(features)}
    
    # Create dataset from the dictionary and labels
    dataset = tf.data.Dataset.from_tensor_slices((features_dict, labels))
    return dataset.shuffle(buffer_size=10000).batch(batch_size)

@tf.function
def train_step(model: tf.keras.Model, 
              optimizer: tf.keras.optimizers.Optimizer,
              features: Dict[str, tf.Tensor], 
              labels: tf.Tensor) -> tf.Tensor:
    """Single training step"""
    with tf.GradientTape() as tape:
        # Get model outputs including masks and sparsity loss during training
        output, masks, sparsity_loss = model(features, training=True)
        # Binary cross-entropy loss
        bce_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, output))
        # Total loss includes sparsity regularization
        total_loss = bce_loss + config["sparsity_coefficient"] * sparsity_loss
        
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, output

def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[float, float]:
    """Evaluate model on dataset"""
    all_labels = []
    all_preds = []
    total_loss = 0
    num_batches = 0
    
    for features, labels in dataset:
        # During evaluation, only get the output predictions
        predictions = model(features, training=False)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, predictions))
        
        all_labels.extend(labels.numpy())
        all_preds.extend(predictions.numpy())
        total_loss += loss.numpy()
        num_batches += 1
        
    avg_loss = total_loss / num_batches
    auc = roc_auc_score(all_labels, all_preds)
    return avg_loss, auc

def main():
    # Configuration for large-scale TabNet
    # Parameters adjusted for 900-dimensional input with 300 features
    global config  # Make config accessible to train_step
    config = {
        "feature_dim": 256,    # Increased feature dimension for complex data
        "output_dim": 1,       # Binary classification
        "num_decision_steps": 5,  # Increased steps for more complex feature interactions
        "relaxation_factor": 1.5,
        "sparsity_coefficient": 1e-4,  # Adjusted for larger feature space
        "virtual_batch_size": 512,  # Virtual batch size for ghost batch norm
        "momentum": 0.02,      # Momentum for batch normalization
        "epsilon": 1e-5
    }
    
    # Training parameters
    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Generate sample data
    print("Generating sample data...")
    train_features = generate_sample_data(batch_size=8192)
    test_features = generate_sample_data(batch_size=2048)
    
    # Create datasets
    train_dataset = create_dataset(train_features, BATCH_SIZE)
    test_dataset = create_dataset(test_features, BATCH_SIZE)
    
    # Initialize model
    print("\nInitializing TabNet model...")
    model = TabNet(
        feature_dim=config["feature_dim"],
        output_dim=config["output_dim"],
        num_decision_steps=config["num_decision_steps"],
        relaxation_factor=config["relaxation_factor"],
        sparsity_coefficient=config["sparsity_coefficient"],
        virtual_batch_size=config["virtual_batch_size"],
        momentum=config["momentum"]
    )
    
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    best_auc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        # Training
        epoch_loss = 0
        num_batches = 0
        for features, labels in train_dataset:
            loss, _ = train_step(model, optimizer, features, labels)
            epoch_loss += loss
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        
        # Evaluation
        test_loss, test_auc = evaluate_model(model, test_dataset)
        
        print(f"Epoch {epoch + 1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test AUC: {test_auc:.4f}")
        
        # Early stopping
        if test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print("\nEarly stopping triggered!")
            break
    
    print(f"\nTraining completed. Best Test AUC: {best_auc:.4f}")
    
    # Final evaluation
    test_loss, test_auc = evaluate_model(model, test_dataset)
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()
