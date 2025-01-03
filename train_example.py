"""
Modified from the original TabNet implementation by DreamQuark:
https://github.com/dreamquark-ai/tabnet

MIT License

Copyright (c) 2019 DreamQuark
"""

import tensorflow as tf
import numpy as np
from tabnet_block import TabNet

# Generate synthetic data with grouped features
def generate_data(n_samples=1000):
    # Numeric features
    num_features = np.random.normal(0, 1, (n_samples, 3))
    
    # Categorical features (one-hot encoded)
    cat1 = np.eye(4)[np.random.randint(0, 4, n_samples)]
    cat2 = np.eye(3)[np.random.randint(0, 3, n_samples)]
    
    # Create target using combination of features
    y = (num_features[:, 0] + cat1[:, 0] > 1).astype(np.float32)
    
    # Create dictionary of features
    features = {
        'numeric': num_features.astype(np.float32),
        'categorical_1': cat1.astype(np.float32),
        'categorical_2': cat2.astype(np.float32)
    }
    
    return features, y.reshape(-1, 1)

# Feature columns definition
feature_columns = {
    'numeric': 3,
    'categorical_1': 4,
    'categorical_2': 3
}

# Generate data
train_features, y_train = generate_data(10000)
val_features, y_val = generate_data(2000)

# Create model without specifying feature columns
model = TabNet(
    output_dim=1,
    n_d=8,
    n_a=8,
    n_steps=3
)

# First call will infer feature dimensions
features = {
    'numeric': tf.random.normal((BATCH_SIZE, 3)),
    'categorical_1': tf.random.uniform((BATCH_SIZE, 4)),
    'categorical_2': tf.random.uniform((BATCH_SIZE, 3))
}

# Model will automatically build with correct feature dimensions
out, masks = model(features, training=True)

print("\nInferred feature columns:")
print(model.feature_columns)

print("\nFeature groups:")
print(model.feature_groups)

# Training parameters
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.02

# Create dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_features, y_train))
train_dataset = train_dataset.shuffle(1000).batch(BATCH_SIZE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_features, y_val))
val_dataset = val_dataset.batch(BATCH_SIZE)

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

# Training step
@tf.function
def train_step(features, y):
    with tf.GradientTape() as tape:
        y_pred, masks = model(features, training=True)
        loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, y_pred))
        
    gradients = tape.gradient(loss, model.trainable_variables)
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

# Analyze feature importance per group
_, masks = model(val_features)
masks = masks.numpy()

print("\nFeature Group Importance:")
for name, indices in model.feature_groups.items():
    importance = masks[:, :, indices].mean()
    print(f"{name}: {importance:.4f}")