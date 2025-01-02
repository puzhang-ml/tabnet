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
        # Get the current batch size
        batch_size = tf.shape(x)[0]
        
        def apply_batch_norm(x_chunk):
            return self.bn(x_chunk, training=training)
            
        def process_full_batch():
            # Calculate number of splits needed
            n_splits = tf.cast(tf.math.ceil(batch_size / self.virtual_batch_size), tf.int32)
            
            # Initialize result list
            chunks_out = tf.TensorArray(dtype=x.dtype, size=n_splits)
            
            # Process each chunk
            def process_chunk(i, chunks_out):
                start_idx = i * self.virtual_batch_size
                end_idx = tf.minimum((i + 1) * self.virtual_batch_size, batch_size)
                chunk = x[start_idx:end_idx]
                chunks_out = chunks_out.write(i, apply_batch_norm(chunk))
                return i + 1, chunks_out
                
            # Process all chunks
            _, chunks_out = tf.while_loop(
                lambda i, _: i < n_splits,
                process_chunk,
                [tf.constant(0), chunks_out]
            )
            
            # Concatenate results
            return tf.concat(chunks_out.stack(), axis=0)
            
        # Use tf.cond to handle both small and large batch cases
        return tf.cond(
            batch_size <= self.virtual_batch_size,
            lambda: apply_batch_norm(x),
            process_full_batch
        )

class GLU_Block(tf.keras.layers.Layer):
    """
    Gated Linear Unit block
    """
    def __init__(self, feature_dim=64, virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.feature_dim = feature_dim
        
        # Dense layer for GLU
        self.dense = tf.keras.layers.Dense(
            feature_dim * 2,
            kernel_initializer='glorot_uniform'
        )
        
        # Batch normalization
        self.bn = GBN(
            input_dim=feature_dim * 2,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        )

    def call(self, x, training=None):
        # Apply dense layer
        x = self.dense(x)
        
        # Apply batch normalization
        x = self.bn(x, training=training)
        
        # Split for gating
        chunks = tf.split(x, num_or_size_splits=2, axis=-1)
        
        # Apply GLU activation
        return chunks[0] * tf.sigmoid(chunks[1])

class FeatTransformer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, bn_momentum=0.9, bn_epsilon=1e-5):
        super(FeatTransformer, self).__init__()
        self.feature_dim = feature_dim
        
        # First GLU block
        self.fc1 = tf.keras.layers.Dense(feature_dim * 2)
        self.bn1 = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            virtual_batch_size=None
        )
        
        # Second GLU block
        self.fc2 = tf.keras.layers.Dense(feature_dim * 2)
        self.bn2 = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            virtual_batch_size=None
        )
    
    def glu(self, x, n_units):
        """Generalized Linear Unit activation."""
        return x[:, :n_units] * tf.nn.sigmoid(x[:, n_units:])
    
    def call(self, inputs, training=None):
        # First GLU block
        x = self.fc1(inputs)
        x = self.bn1(x, training=training)
        x = self.glu(x, self.feature_dim)
        
        # Second GLU block
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.glu(x, self.feature_dim)
        
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
    def __init__(self, feature_dim, bn_momentum=0.9, bn_epsilon=1e-5, virtual_batch_size=None):
        super(AttentiveTransformer, self).__init__()
        self.feature_dim = feature_dim
        
        # Dense layer for attention
        self.fc = tf.keras.layers.Dense(feature_dim)
        
        # Batch normalization
        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=bn_momentum,
            epsilon=bn_epsilon,
            virtual_batch_size=virtual_batch_size
        )
    
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        return sparsemax(x)

class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(self, feature_dim, output_dim, n_steps=3, n_total=6, n_shared=2,
                 relaxation_factor=1.5, sparsity_coefficient=1e-5, bn_momentum=0.9,
                 bn_epsilon=1e-5):
        super(TabNetEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.n_steps = n_steps
        self.n_total = n_total
        self.n_shared = n_shared
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        
        # Initial batch norm layer
        self.initial_bn = tf.keras.layers.BatchNormalization(
            axis=-1,  # Last dimension for features
            momentum=bn_momentum,
            epsilon=bn_epsilon
        )
        
        # Feature transformer layers
        self.feature_transforms = []
        for i in range(n_total):
            self.feature_transforms.append(
                FeatTransformer(
                    feature_dim,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon
                )
            )
        
        # Attention transformers
        self.attentive_transforms = []
        for i in range(n_steps):
            self.attentive_transforms.append(
                AttentiveTransformer(
                    feature_dim,
                    bn_momentum=bn_momentum,
                    bn_epsilon=bn_epsilon
                )
            )
    
    def call(self, inputs, training=None):
        x = self.initial_bn(inputs, training=training)
        
        # Lists to store step outputs and loss
        steps_output = []
        total_entropy = 0
        
        prior = tf.ones_like(x)
        M_loss = 0
        
        for step_idx in range(self.n_steps):
            # Shared feature transform layers
            features = x
            for i in range(self.n_shared):
                features = self.feature_transforms[i](features, training=training)
            
            # Decision step feature transform layers
            decision_out = features
            for i in range(self.n_shared, self.n_total):
                decision_out = self.feature_transforms[i](decision_out, training=training)
            
            # Apply attention mechanism
            M = self.attentive_transforms[step_idx](features, training=training)
            M = M * prior
            
            # Update prior
            prior = prior * (self.relaxation_factor - M)
            
            # Compute entropy for sparsity loss
            total_entropy += tf.reduce_mean(tf.reduce_sum(
                -M * tf.math.log(M + 1e-15), axis=1
            ))
            
            # Apply feature selection to decision output
            masked_decision_out = tf.multiply(decision_out, M)
            steps_output.append(masked_decision_out)
        
        M_loss = total_entropy / self.n_steps * self.sparsity_coefficient
        steps_output = tf.stack(steps_output, axis=0)
        
        return steps_output, M_loss

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
    TabNet model implementation
    """
    def __init__(self, feature_dim=64, output_dim=1, n_steps=3, n_total=6, n_shared=2,
                 relaxation_factor=1.5, sparsity_coefficient=1e-5, bn_momentum=0.9,
                 bn_epsilon=1e-5):
        super(TabNet, self).__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # TabNet encoder
        self.tabnet = TabNetEncoder(
            feature_dim=feature_dim,
            output_dim=output_dim,
            n_steps=n_steps,
            n_total=n_total,
            n_shared=n_shared,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon
        )
        
        # Output layer
        self.output_layer = tf.keras.layers.Dense(output_dim)
        
        # Input projection layer
        self.input_projection = DynamicProjection(feature_dim)
    
    def _process_input(self, inputs):
        if isinstance(inputs, dict):
            # Concatenate dictionary inputs
            x_processed = tf.concat(list(inputs.values()), axis=1)
        elif isinstance(inputs, (list, tuple)):
            # Concatenate list/tuple inputs along feature dimension
            x_processed = tf.concat(inputs, axis=1)
        else:
            # Single tensor input
            x_processed = inputs
        
        # Project to feature dimension
        x_processed = self.input_projection(x_processed)
        return x_processed
    
    def call(self, inputs, training=None):
        # Process inputs
        x_processed = self._process_input(inputs)
        
        # Apply TabNet encoder
        steps_output, M_loss = self.tabnet(x_processed, training=training)
        
        # Get final step output and apply output layer
        final_output = steps_output[-1]  # Shape: [batch_size, feature_dim]
        output = self.output_layer(final_output)  # Shape: [batch_size, output_dim]
        
        # Add loss to the model
        self.add_loss(M_loss)
        
        return output

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