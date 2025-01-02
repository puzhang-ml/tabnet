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
    """Generate sample data for training."""
    # Generate scalar features
    scalar1 = tf.random.normal((num_samples,))
    scalar2 = tf.random.normal((num_samples,))
    
    # Generate embedding features
    embedding1 = tf.random.normal((num_samples, 16))
    embedding2 = tf.random.normal((num_samples, 32))
    
    # Generate continuous features
    continuous1 = tf.random.normal((num_samples, 10))
    continuous2 = tf.random.normal((num_samples, 20))
    
    # Create feature dictionary
    features = {
        'scalar1': scalar1,
        'scalar2': scalar2,
        'embedding1': embedding1,
        'embedding2': embedding2,
        'continuous1': continuous1,
        'continuous2': continuous2
    }
    
    # Calculate total feature dimension
    total_dim = (
        1 +  # scalar1
        1 +  # scalar2
        16 + # embedding1
        32 + # embedding2
        10 + # continuous1
        20   # continuous2
    )
    
    # Define feature groups
    current_idx = 2  # After two scalar features
    embedding1_start = current_idx
    embedding1_end = embedding1_start + 16
    embedding2_start = embedding1_end
    embedding2_end = embedding2_start + 32
    
    grouped_features = [
        list(range(embedding1_start, embedding1_end)),  # embedding1 group
        list(range(embedding2_start, embedding2_end))   # embedding2 group
    ]
    
    return features, total_dim, grouped_features

def create_dataset(features: Dict[str, tf.Tensor], batch_size: int) -> tf.data.Dataset:
    """Create TensorFlow dataset from features dictionary"""
    # Generate random binary labels for this example
    labels = tf.cast(tf.random.uniform((features['scalar1'].shape[0],)) > 0.5, tf.float32)
    
    # Create dataset from the dictionary and labels
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(buffer_size=10000).batch(batch_size)

@tf.function
def train_step(model, optimizer, features, labels):
    """Single training step."""
    with tf.GradientTape() as tape:
        # Forward pass
        output, _, sparsity_loss = model(features, training=True)
        
        # Compute BCE loss
        bce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(labels, output)
        )
        
        # Total loss is BCE loss plus sparsity loss
        # Note: sparsity_coefficient is already applied in the model
        total_loss = bce_loss + sparsity_loss
        
    # Compute gradients and update weights
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
    """Main training function."""
    print("\nGenerating sample data...")
    features, total_dim, grouped_features = generate_sample_data()
    
    # Model configuration
    config = {
        "feature_dim": total_dim,  # Total dimension of concatenated features
        "output_dim": 1,           # Binary classification
        "num_decision_steps": 5,
        "relaxation_factor": 1.5,
        "sparsity_coefficient": 1e-5,
        "bn_virtual_bs": 128,
        "bn_momentum": 0.02
    }
    
    print("\nInitializing TabNet model...")
    model = TabNet(
        feature_dim=config["feature_dim"],
        output_dim=config["output_dim"],
        num_decision_steps=config["num_decision_steps"],
        relaxation_factor=config["relaxation_factor"],
        sparsity_coefficient=config["sparsity_coefficient"],
        bn_virtual_bs=config["bn_virtual_bs"],
        bn_momentum=config["bn_momentum"],
        epsilon=1e-5,
        grouped_features=grouped_features
    )
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    
    # Create datasets
    train_dataset = create_dataset(features, batch_size=256)
    test_dataset = create_dataset(features, batch_size=256)
    
    print("\nStarting training...")
    best_auc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(100):
        # Training
        epoch_loss = 0
        num_batches = 0
        
        for features_batch, labels_batch in train_dataset:
            loss, _ = train_step(model, optimizer, features_batch, labels_batch)
            epoch_loss += loss
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Evaluation
        test_loss, test_auc = evaluate_model(model, test_dataset)
        
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}, "
              f"Test Loss = {test_loss:.4f}, Test AUC = {test_auc:.4f}")
        
        # Early stopping
        if test_auc > best_auc:
            best_auc = test_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered. Best Test AUC: {best_auc:.4f}")
            break

if __name__ == "__main__":
    main()
