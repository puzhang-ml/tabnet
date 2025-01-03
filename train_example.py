"""
Modified from the original TabNet implementation by DreamQuark:
https://github.com/dreamquark-ai/tabnet

MIT License

Copyright (c) 2019 DreamQuark
"""

import tensorflow as tf
import numpy as np
from tabnet_block import TabNet

# Generate synthetic data with high dimensionality
def generate_data(n_samples=10000):
    # Example with 300 features:
    # - 100 continuous features (dim=1 each)
    # - 100 categorical features with 5 categories (dim=5 each)
    # - 100 categorical features with 4 categories (dim=4 each)
    continuous = np.random.normal(0, 1, (n_samples, 100))
    cat_5 = np.eye(5)[np.random.randint(0, 5, (n_samples, 100))]
    cat_4 = np.eye(4)[np.random.randint(0, 4, (n_samples, 100))]
    
    features = {
        'continuous': continuous.astype(np.float32),
        'categorical_5': cat_5.reshape(n_samples, -1).astype(np.float32),  # 100 * 5 = 500 dims
        'categorical_4': cat_4.reshape(n_samples, -1).astype(np.float32)   # 100 * 4 = 400 dims
    }
    
    # Create target (example: based on some features)
    y = (np.sum(continuous[:, :10], axis=1) > 0).astype(np.float32)
    
    return features, y.reshape(-1, 1)

# Hyperparameters for large feature set
BATCH_SIZE = 8192
EPOCHS = 100
LR = 0.02

# Generate data first
train_features, y_train = generate_data(100000)  # 100k samples
val_features, y_val = generate_data(20000)      # 20k samples

# Create model with dynamic feature inference
model = TabNet(
    # feature_columns will be inferred from first input
    output_dim=1,
    n_d=64,
    n_a=64,
    n_steps=5,
    gamma=1.5,
    n_independent=2,
    n_shared=2,
    virtual_batch_size=512,
    momentum=0.02
)

# First forward pass will trigger feature inference
sample_batch = next(iter(train_dataset))
_, _ = model(sample_batch[0], training=False)

print("\nInferred Feature Dimensions:")
print(model.feature_columns)

print("\nFeature Groups:")
print(model.feature_groups)

# Create dataset with large batch size
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, y_train))
train_dataset = train_dataset.shuffle(10000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_features, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

# Optimizer with gradient clipping
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
gradient_clip_norm = 2.0

# Training step with gradient clipping
@tf.function
def train_step(features, y):
    with tf.GradientTape() as tape:
        y_pred, masks = model(features, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_pred))
        
    gradients = tape.gradient(loss, model.trainable_variables)
    # Clip gradients to handle high dimensionality
    gradients, _ = tf.clip_by_global_norm(gradients, gradient_clip_norm)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, masks

# Training loop with feature importance monitoring
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
        
        print("\nFeature Group Importance:")
        for name, indices in model.feature_groups.items():
            importance = masks[:, :, indices].mean()
            print(f"{name}: {importance:.4f}")