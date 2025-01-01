import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple
import json

from tabnet_block import TabNet, create_feature_config

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def generate_sample_data(n_samples: int = 1000) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
    """
    Generate sample data with binary classification target
    Returns features dictionary and labels
    """
    # Generate features
    features = {
        'numeric_feature1': tf.random.normal((n_samples, 1)),
        'numeric_feature2': tf.random.normal((n_samples, 1)),
        'embedding1': tf.random.normal((n_samples, 8)),
        'embedding2': tf.random.normal((n_samples, 16)),
        'categorical_embedding': tf.random.normal((n_samples, 12))
    }
    
    # Generate target (based on some features)
    logits = (tf.reduce_mean(features['embedding1'], axis=1) + 
             features['numeric_feature1'][:, 0] * 0.5 +
             tf.reduce_sum(features['embedding2'][:, :4], axis=1) * 0.1)
    probabilities = tf.sigmoid(logits)
    labels = tf.cast(probabilities > 0.5, tf.float32)
    
    return features, labels

def create_dataset(features: Dict[str, tf.Tensor], labels: tf.Tensor, 
                  batch_size: int) -> tf.data.Dataset:
    """Create TensorFlow dataset from features dictionary and labels"""
    return tf.data.Dataset.from_tensor_slices((features, labels))\
        .shuffle(buffer_size=1000)\
        .batch(batch_size)

@tf.function
def train_step(model: tf.keras.Model, 
              optimizer: tf.keras.optimizers.Optimizer,
              features: Dict[str, tf.Tensor], 
              labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Single training step"""
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels, predictions))
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, predictions

def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> Tuple[float, float]:
    """Evaluate model on dataset"""
    all_labels = []
    all_preds = []
    total_loss = 0
    num_batches = 0
    
    for features, labels in dataset:
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
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    N_SAMPLES = 5000
    
    # Configuration
    config = {
        "tabnet_feature_dim": 64,  # Reduced from 512 for this example
        "tabnet_output_dim": 1,    # Binary classification
        "tabnet_num_decision_steps": 3,
        "tabnet_relaxation_factor": 1.5,
        "tabnet_batch_momentum": 0.98,
        "tabnet_batch_size": 128
    }
    
    print("Generating data...")
    # Generate train and test data
    train_features, train_labels = generate_sample_data(N_SAMPLES)
    test_features, test_labels = generate_sample_data(N_SAMPLES // 5)
    
    # Create datasets
    train_dataset = create_dataset(train_features, train_labels, BATCH_SIZE)
    test_dataset = create_dataset(test_features, test_labels, BATCH_SIZE)
    
    # Create feature configuration
    feature_config = create_feature_config(train_features)
    print("\nFeature Configuration:")
    print(json.dumps(feature_config, indent=2))
    
    # Initialize model
    print("\nInitializing TabNet model...")
    model = TabNet(
        feature_config=feature_config,
        feature_dim=config["tabnet_feature_dim"],
        output_dim=config["tabnet_output_dim"],
        n_steps=config["tabnet_num_decision_steps"],
        gamma=config["tabnet_relaxation_factor"],
        momentum=config["tabnet_batch_momentum"],
        virtual_batch_size=config["tabnet_batch_size"]
    )
    
    # Initialize optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
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