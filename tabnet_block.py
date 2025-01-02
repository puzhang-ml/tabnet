import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple

class GBN(tf.keras.layers.Layer):
    """
    Ghost Batch Normalization
    """
    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()
        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        
    def call(self, x, training=None):
        # During model building, just use regular batch normalization
        if x.shape[0] is None:
            return self.bn(x, training=training)
            
        # Get actual batch size
        batch_size = tf.shape(x)[0]
        
        # If batch size is smaller than virtual_batch_size, use regular BN
        if self.virtual_batch_size is None or batch_size <= self.virtual_batch_size:
            return self.bn(x, training=training)
            
        # Calculate number of splits and split size
        n_splits = tf.cast(tf.math.ceil(batch_size / self.virtual_batch_size), tf.int32)
        split_size = tf.cast(tf.math.ceil(batch_size / n_splits), tf.int32)
        
        # Create split sizes tensor
        last_split_size = batch_size - split_size * (n_splits - 1)
        split_sizes = tf.concat([
            tf.repeat(split_size, n_splits - 1),
            tf.expand_dims(last_split_size, 0)
        ], axis=0)
        
        # Split into virtual batches
        chunks = tf.split(x, split_sizes, axis=0)
        normalized_chunks = [self.bn(chunk, training=training) for chunk in chunks]
        
        return tf.concat(normalized_chunks, axis=0)

class GLU_Block(tf.keras.layers.Layer):
    def __init__(self, feature_dim, virtual_batch_size=None, **kwargs):
        super(GLU_Block, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        
        # Dense layer for GLU
        self.fc = tf.keras.layers.Dense(feature_dim * 2)
        
        # Batch normalization
        self.bn = GBN(
            input_dim=feature_dim * 2,
            virtual_batch_size=virtual_batch_size
        )
    
    def call(self, x, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        
        # GLU activation
        x_linear = x[:, :self.feature_dim]
        x_gated = x[:, self.feature_dim:]
        
        return x_linear * tf.nn.sigmoid(x_gated)

class FeatTransformer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, virtual_batch_size=None, **kwargs):
        super(FeatTransformer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        
        # First GLU block
        self.glu1 = GLU_Block(
            feature_dim=feature_dim,
            virtual_batch_size=virtual_batch_size
        )
        
        # Second GLU block
        self.glu2 = GLU_Block(
            feature_dim=feature_dim,
            virtual_batch_size=virtual_batch_size
        )
    
    def call(self, inputs, training=None):
        x = self.glu1(inputs, training=training)
        x = self.glu2(x, training=training)
        return x

def sparsemax(z):
    """Sparsemax activation function."""
    # Sort z
    z_sorted = tf.sort(z, axis=-1, direction='DESCENDING')
    
    # Calculate cumulative sum
    z_cumsum = tf.cumsum(z_sorted, axis=-1)
    k = tf.range(1, tf.shape(z)[-1] + 1, dtype=tf.float32)
    z_check = 1 + k * z_sorted > z_cumsum
    
    # Number of valid elements
    k_z = tf.reduce_sum(tf.cast(z_check, tf.float32), axis=-1)
    
    # Calculate threshold
    indices = tf.stack([
        tf.range(tf.shape(z)[0]),
        tf.cast(k_z - 1, tf.int32)
    ], axis=1)
    tau = tf.gather_nd(z_cumsum, indices)
    tau = (tau - 1) / k_z
    
    # Calculate p
    p = tf.maximum(tf.cast(0, z.dtype), z - tf.expand_dims(tau, -1))
    
    return p

class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, virtual_batch_size=None, **kwargs):
        super(AttentiveTransformer, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        
        # Dense layer for attention
        self.fc = tf.keras.layers.Dense(feature_dim)
        
        # Batch normalization
        self.bn = GBN(
            input_dim=feature_dim,
            virtual_batch_size=virtual_batch_size
        )
    
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        return sparsemax(x)

class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(self, feature_dim, output_dim, num_decision_steps=5,
                 relaxation_factor=1.5, sparsity_coefficient=1e-5,
                 virtual_batch_size=None, num_groups=1, epsilon=1e-5,
                 **kwargs):
        super(TabNetEncoder, self).__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.virtual_batch_size = virtual_batch_size
        self.num_groups = num_groups
        self.epsilon = epsilon
        
        # Feature transformer
        self.transform = FeatTransformer(
            feature_dim=feature_dim
        )
        
        # Attentive transformers
        self.attentive_transformers = []
        for _ in range(num_decision_steps):
            self.attentive_transformers.append(
                AttentiveTransformer(
                    feature_dim=feature_dim,
                    virtual_batch_size=virtual_batch_size
                )
            )
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, features, training=None):
        batch_size = tf.shape(features)[0]
        
        # Initialize masks
        prior_scales = tf.ones([batch_size, self.feature_dim])
        
        # Transform features
        transformed_features = self.transform(features, training=training)
        
        # Process each decision step
        step_outputs = []
        for step_idx in range(self.num_decision_steps):
            # Get attention mask
            mask = self.attentive_transformers[step_idx](transformed_features, training=training)
            
            # Update prior scales
            prior_scales = prior_scales * (self.relaxation_factor - mask)
            
            # Apply mask to features
            masked_features = transformed_features * mask
            
            # Get step output
            step_output = self.output_layer(masked_features)
            step_outputs.append(step_output)
            
            # Add sparsity loss
            batch_means = tf.reduce_mean(mask, axis=0)
            sparsity_loss = tf.reduce_mean(
                tf.reduce_sum(batch_means * tf.math.log(batch_means + self.epsilon))
            )
            self.add_loss(self.sparsity_coefficient * sparsity_loss)
        
        # Return final step output
        return step_outputs[-1]

class DynamicProjection(tf.keras.layers.Layer):
    def __init__(self, feature_dim, **kwargs):
        super(DynamicProjection, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.kernel = None
        self.bias = None
        self.built = False

    def build(self, input_shape):
        if self.built:
            return
            
        input_shape = tf.TensorShape(input_shape)
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        
        # Get the input feature dimension
        input_dim = input_shape[-1]
        if input_dim is None:
            # If input dimension is not known, defer building until call time
            return
        
        # Create kernel and bias variables with correct shapes
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_dim, self.feature_dim],
            initializer='glorot_uniform',
            dtype=dtype,
            trainable=True
        )
        self.bias = self.add_weight(
            'bias',
            shape=[self.feature_dim],
            initializer='zeros',
            dtype=dtype,
            trainable=True
        )
        
        self.built = True
        super(DynamicProjection, self).build(input_shape)

    def call(self, inputs, training=None):
        # If not built yet, build now with actual input shape
        if not self.built:
            input_shape = tf.shape(inputs)
            input_dim = input_shape[-1]
            dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
            
            # Create kernel and bias variables with correct shapes
            self.kernel = self.add_weight(
                'kernel',
                shape=[input_dim, self.feature_dim],
                initializer='glorot_uniform',
                dtype=dtype,
                trainable=True
            )
            self.bias = self.add_weight(
                'bias',
                shape=[self.feature_dim],
                initializer='zeros',
                dtype=dtype,
                trainable=True
            )
            self.built = True
        
        # Project to feature dimension using matmul and bias
        outputs = tf.matmul(inputs, self.kernel) + self.bias
        return outputs

class TabNet(tf.keras.Model):
    """
    TabNet model implementation that handles:
    1. Embedding features (pre-embedded tensors)
    2. Scalar features (1D)
    3. Regular continuous features (2D)
    """
    def __init__(self, feature_dim=64, output_dim=1, num_decision_steps=5,
                 relaxation_factor=1.5, sparsity_coefficient=1e-5,
                 virtual_batch_size=None, num_groups=1, epsilon=1e-5):
        super(TabNet, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # TabNet encoder
        self.encoder = TabNetEncoder(
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            virtual_batch_size=virtual_batch_size,
            num_groups=num_groups,
            epsilon=epsilon
        )
        
        # Input projection layer
        self.input_projection = DynamicProjection(feature_dim)
    
    def _process_input(self, inputs):
        if isinstance(inputs, dict):
            # Process dictionary inputs
            processed_features = []
            for feat_name, tensor in inputs.items():
                # Handle scalar features (1D)
                if len(tensor.shape) == 1:
                    processed = tf.expand_dims(tensor, -1)
                # Handle embedding features (already embedded) and regular features
                else:
                    processed = tensor
                processed_features.append(processed)
            # Concatenate along feature dimension
            x_processed = tf.concat(processed_features, axis=-1)
        elif isinstance(inputs, (list, tuple)):
            # Process list/tuple inputs
            processed_features = []
            for tensor in inputs:
                # Handle scalar features (1D)
                if len(tensor.shape) == 1:
                    processed = tf.expand_dims(tensor, -1)
                # Handle embedding features (already embedded) and regular features
                else:
                    processed = tensor
                processed_features.append(processed)
            # Concatenate along feature dimension
            x_processed = tf.concat(processed_features, axis=-1)
        else:
            # Single tensor input
            if len(inputs.shape) == 1:
                # Handle scalar input
                x_processed = tf.expand_dims(inputs, -1)
            else:
                x_processed = inputs
        
        # Ensure input is 2D before projection
        if len(x_processed.shape) == 1:
            x_processed = tf.expand_dims(x_processed, -1)
        
        # Project to feature dimension
        x_processed = self.input_projection(x_processed)
        return x_processed
    
    def call(self, inputs, training=None):
        x_processed = self._process_input(inputs)
        return self.encoder(x_processed, training=training)

class FeatureProcessor:
    def __init__(self, feature_config: dict):
        self.feature_config = feature_config
        self.total_dims = feature_config['total_dims']
        
    def process_features(self, feature_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Convert dictionary of features into a single concatenated tensor"""
        features_list = []
        
        for feature_name, info in self.feature_config.items():
            if feature_name != 'total_dims':
                tensor = feature_dict[feature_name]
                features_list.append(tensor)
        
        return tf.concat(features_list, axis=1) 