"""
Example of using TabNet encoder outputs with other features.
Shows how to create a hybrid model combining TabNet with other features.
"""

import tensorflow as tf
import numpy as np
from tabnet_block import TabNet, TabNetEncoder

class HybridModel(tf.keras.Model):
    def __init__(
        self,
        n_d=64,
        n_a=64,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=512,
        momentum=0.02
    ):
        super().__init__()
        # Store parameters for later use
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
        # These will be built in build() method
        self.tabnet_encoder = None
        self.additional_dense = tf.keras.layers.Dense(32)
        self.concat_bn = tf.keras.layers.BatchNormalization()
        self.final_dense = tf.keras.layers.Dense(1)

    def build(self, input_shape):
        # Get input dimensions from the tabnet_features shape
        tabnet_features_shape = input_shape['tabnet_features']
        if isinstance(tabnet_features_shape, dict):
            # Calculate total input dimension from dictionary of features
            input_dim = sum(shape[-1] for shape in tabnet_features_shape.values())
        else:
            input_dim = tabnet_features_shape[-1]
            
        # Now we can create the TabNetEncoder with proper input_dim
        self.tabnet_encoder = TabNetEncoder(
            input_dim=input_dim,
            output_dim=self.n_d * self.n_steps,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            virtual_batch_size=self.virtual_batch_size,
            momentum=self.momentum
        )
        
        super().build(input_shape)

    def call(self, inputs, training=None):
        # Split inputs into tabnet features and additional features
        tabnet_features = inputs['tabnet_features']
        additional_features = inputs['additional_features']
        
        # Process tabnet features
        tabnet_output, masks = self.tabnet_encoder(tabnet_features, training=training)
        
        # Process additional features
        additional_output = self.additional_dense(additional_features)
        
        # Combine features
        combined = tf.concat([tabnet_output, additional_output], axis=-1)
        combined = self.concat_bn(combined, training=training)
        
        # Final prediction
        output = self.final_dense(combined)
        
        return output, masks

# Generate synthetic data with both tabnet and additional features
def generate_hybrid_data(n_samples=10000):
    # TabNet features (same as train_example.py)
    continuous = np.random.normal(0, 1, (n_samples, 100))
    cat_5 = np.eye(5)[np.random.randint(0, 5, (n_samples, 100))]
    cat_4 = np.eye(4)[np.random.randint(0, 4, (n_samples, 100))]
    
    tabnet_features = {
        'continuous': continuous.astype(np.float32),
        'categorical_5': cat_5.reshape(n_samples, -1).astype(np.float32),
        'categorical_4': cat_4.reshape(n_samples, -1).astype(np.float32)
    }
    
    # Additional features (e.g., from another model or source)
    additional_features = np.random.normal(0, 1, (n_samples, 20)).astype(np.float32)
    
    # Combined features
    features = {
        'tabnet_features': tabnet_features,
        'additional_features': additional_features
    }
    
    # Create target using both feature sets
    y = (np.sum(continuous[:, :10], axis=1) + 
         np.sum(additional_features[:, :5], axis=1) > 0).astype(np.float32)
    
    return features, y.reshape(-1, 1)

# Training parameters (same as train_example.py)
BATCH_SIZE = 8192
EPOCHS = 100
LR = 0.02

# Generate data
train_features, y_train = generate_hybrid_data(100000)
val_features, y_val = generate_hybrid_data(20000)

# Create model
model = HybridModel(
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    virtual_batch_size=512,
    momentum=0.02
)

# Create datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, y_train))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_features, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

# Optimizer with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
gradient_clip_norm = 2.0

# Training step
@tf.function
def train_step(features, y):
    with tf.GradientTape() as tape:
        y_pred, masks = model(features, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_pred))
        
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, masks

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = []
    for x_batch, y_batch in train_dataset:
        loss, masks = train_step(x_batch, y_batch)
        epoch_loss.append(float(loss))
    
    # Validation
    val_preds = []
    val_true = []
    for x_val, y_val in val_dataset:
        pred, _ = model(x_val, training=False)
        val_preds.extend(pred.numpy())
        val_true.extend(y_val.numpy())
    
    val_auc = tf.keras.metrics.AUC()(val_true, val_preds)
    print(f"Epoch {epoch+1}, Loss: {np.mean(epoch_loss):.4f}, Val AUC: {val_auc:.4f}")
    
    # Monitor feature importance every 10 epochs
    if (epoch + 1) % 10 == 0:
        _, masks = model(next(iter(val_dataset))[0])
        masks = masks.numpy()
        
        print("\nTabNet Feature Group Importance:")
        for name, indices in model.tabnet_encoder.feature_groups.items():
            importance = masks[:, :, indices].mean()
            print(f"{name}: {importance:.4f}")

# Example of getting intermediate representations
sample_batch = next(iter(val_dataset))
tabnet_output, masks = model.tabnet_encoder(sample_batch[0]['tabnet_features'])
print("\nTabNet encoder output shape:", tabnet_output.shape)
print("Feature masks shape:", masks.shape) 