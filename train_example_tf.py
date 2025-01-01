import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple
import json

from tabnet_core_tf import TabNetEncoder

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)

def generate_sample_feature_dict(batch_size: int = 2) -> Dict[str, tf.Tensor]:
    """
    Generate sample feature dictionary with mixed dimensionality
    Always generates batches of 2 examples
    """
    return {
        # Regular features (1D)
        'age': tf.random.normal((batch_size, 1)),
        'income': tf.random.normal((batch_size, 1)),
        
        # Embedding vectors (Multi-dimensional)
        'user_embedding': tf.random.normal((batch_size, 8)),  # 8-dim embedding
        'item_embedding': tf.random.normal((batch_size, 16)),  # 16-dim embedding
        'category_embedding': tf.random.normal((batch_size, 12)),  # 12-dim embedding
        
        # Another 1D feature
        'duration': tf.random.normal((batch_size, 1))
    }

def create_feature_config(feature_dict: Dict[str, tf.Tensor]) -> dict:
    """
    Create feature configuration from a sample feature dictionary
    """
    config = {}
    total_dims = 0
    
    for feature_name, tensor in feature_dict.items():
        feature_dims = tensor.shape[-1]  # Get the last dimension
        config[feature_name] = {
            'start_idx': total_dims,
            'end_idx': total_dims + feature_dims,
            'dims': feature_dims
        }
        total_dims += feature_dims
    
    config['total_dims'] = total_dims
    return config

def create_group_matrix(feature_config: dict) -> np.ndarray:
    """
    Create group matrix where each feature (1D or embedding) is treated as one group
    """
    total_dims = feature_config['total_dims']
    n_features = len(feature_config) - 1  # Subtract 1 for 'total_dims' key
    group_matrix = np.zeros((total_dims, n_features))
    
    for idx, (feature_name, info) in enumerate(feature_config.items()):
        if feature_name != 'total_dims':
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            group_matrix[start_idx:end_idx, idx] = 1
            
    return group_matrix

class FeatureProcessor:
    def __init__(self, feature_config: dict):
        self.feature_config = feature_config
        self.total_dims = feature_config['total_dims']
        
    def process_features(self, feature_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Convert dictionary of features into a single concatenated tensor
        Ensures batch size is always 2
        """
        features_list = []
        
        for feature_name, info in self.feature_config.items():
            if feature_name != 'total_dims':
                tensor = feature_dict[feature_name]
                # Verify batch size is 2
                batch_size = tf.shape(tensor)[0]
                tf.debugging.assert_equal(batch_size, 2, 
                    message=f"Expected batch size 2 for feature {feature_name}, got {batch_size}")
                features_list.append(tensor)
        
        return tf.concat(features_list, axis=1)

class TabNetBinary(tf.keras.Model):
    def __init__(self, feature_config: dict, n_d=8, n_a=8):
        super(TabNetBinary, self).__init__()
        self.feature_config = feature_config
        self.feature_processor = FeatureProcessor(feature_config)
        
        # Create group matrix
        group_matrix = create_group_matrix(feature_config)
        
        self.tabnet = TabNetEncoder(
            input_dim=feature_config['total_dims'],
            output_dim=1,
            n_d=n_d,
            n_a=n_a,
            n_steps=3,
            gamma=1.3,
            n_independent=2,
            n_shared=2,
            virtual_batch_size=2,  # Set to match our batch size
            momentum=0.02,
            group_attention_matrix=group_matrix
        )
        self.fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, x: Dict[str, tf.Tensor], training=None):
        # Process dictionary input into tensor
        x_processed = self.feature_processor.process_features(x)
        steps_output, M_loss = self.tabnet(x_processed, training=training)
        last_step = steps_output[-1]
        out = self.fc(last_step)
        return out, M_loss

@tf.function
def train_step(model, optimizer, x_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred, M_loss = model(x_batch, training=True)
        loss = tf.keras.losses.binary_crossentropy(y_batch, y_pred) + 0.001 * M_loss
        
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, y_pred

def main():
    # Parameters
    BATCH_SIZE = 2  # Fixed batch size of 2
    N_PAIRS = 5000  # Number of pairs (total samples will be 10000)
    EPOCHS = 100
    
    # Generate sample feature dictionary
    print("Generating sample feature dictionary...")
    sample_features = generate_sample_feature_dict(BATCH_SIZE)
    
    # Create feature configuration
    feature_config = create_feature_config(sample_features)
    print("\nFeature Configuration:")
    print(json.dumps(feature_config, indent=2))
    
    # Create datasets
    def generate_batch():
        while True:
            features = generate_sample_feature_dict(BATCH_SIZE)
            # Generate paired labels
            labels = tf.cast(tf.random.uniform((BATCH_SIZE, 1)) > 0.5, tf.float32)
            yield features, labels
    
    train_dataset = tf.data.Dataset.from_generator(
        generate_batch,
        output_signature=(
            {k: tf.TensorSpec(shape=(BATCH_SIZE, v.shape[-1]), dtype=tf.float32) 
             for k, v in sample_features.items()},
            tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
        )
    ).take(N_PAIRS)
    
    test_dataset = tf.data.Dataset.from_generator(
        generate_batch,
        output_signature=(
            {k: tf.TensorSpec(shape=(BATCH_SIZE, v.shape[-1]), dtype=tf.float32) 
             for k, v in sample_features.items()},
            tf.TensorSpec(shape=(BATCH_SIZE, 1), dtype=tf.float32)
        )
    ).take(N_PAIRS // 5)
    
    # Initialize model
    print("\nInitializing TabNet model with feature groups...")
    model = TabNetBinary(feature_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
    
    # Training loop
    print("Starting training...")
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        total_loss = 0
        num_batches = 0
        
        # Training
        for x_batch, y_batch in train_dataset:
            loss, _ = train_step(model, optimizer, x_batch, y_batch)
            total_loss += loss
            num_batches += 1
            
        # Validation
        val_preds = []
        val_true = []
        
        for x_batch, y_batch in test_dataset:
            y_pred, _ = model(x_batch, training=False)
            val_preds.extend(y_pred.numpy())
            val_true.extend(y_batch.numpy())
            
        val_auc = roc_auc_score(val_true, val_preds)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss/num_batches:.4f}, Val AUC = {val_auc:.4f}")
            # Print example predictions for a batch
            if epoch == 0:
                for x_batch, y_batch in train_dataset.take(1):
                    y_pred, _ = model(x_batch, training=False)
                    print("\nExample batch predictions:")
                    print(f"True labels: {y_batch.numpy().flatten()}")
                    print(f"Predictions: {y_pred.numpy().flatten()}\n")
    
    print(f"Training completed. Best AUC: {best_val_auc:.4f}")

if __name__ == "__main__":
    main() 