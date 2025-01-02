import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple

class GBN(tf.keras.layers.Layer):
    """Ghost Batch Normalization layer."""
    def __init__(self, feature_dim, virtual_batch_size=None, momentum=0.9, epsilon=1e-5):
        """Initialize the layer.
        
        Args:
            feature_dim: Dimension of input features
            virtual_batch_size: Size of ghost batches
            momentum: Momentum for moving average
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Initialize moving statistics
        self.beta = tf.Variable(tf.zeros([feature_dim]), trainable=True)
        self.gamma = tf.Variable(tf.ones([feature_dim]), trainable=True)
        self.moving_mean = tf.Variable(tf.zeros([feature_dim]), trainable=False)
        self.moving_variance = tf.Variable(tf.ones([feature_dim]), trainable=False)

    def _batch_norm(self, x, training):
        """Apply batch normalization to input tensor.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Normalized tensor
        """
        if training:
            mean, variance = tf.nn.moments(x, axes=[0], keepdims=True)
            # Update moving statistics
            self.moving_mean.assign(
                self.momentum * self.moving_mean + (1 - self.momentum) * tf.squeeze(mean))
            self.moving_variance.assign(
                self.momentum * self.moving_variance + (1 - self.momentum) * tf.squeeze(variance))
        else:
            mean = tf.reshape(self.moving_mean, [1, -1])
            variance = tf.reshape(self.moving_variance, [1, -1])
            
        x = (x - mean) / tf.sqrt(variance + self.epsilon)
        return x * self.gamma + self.beta
        
    def call(self, x, training=None):
        """Forward pass.
        
        Args:
            x: Input tensor
            training: Whether in training mode
            
        Returns:
            Normalized tensor
        """
        if training and self.virtual_batch_size is not None:
            # Split input into virtual batches
            batch_size = tf.shape(x)[0]
            split_size = self.virtual_batch_size
            
            if batch_size < split_size:
                return self._batch_norm(x, training)
                
            # Calculate number of full virtual batches
            n_vbs = batch_size // split_size
            
            # Split into virtual batches
            vbs = tf.split(x[:n_vbs * split_size], n_vbs)
            
            # Handle remaining samples
            remaining = x[n_vbs * split_size:]
            if tf.shape(remaining)[0] > 0:
                vbs.append(remaining)
            
            # Apply batch norm to each virtual batch
            vb_outputs = []
            for vb in vbs:
                vb_output = self._batch_norm(vb, training)
                vb_outputs.append(vb_output)
                
            # Concatenate results
            return tf.concat(vb_outputs, axis=0)
        else:
            return self._batch_norm(x, training)

class GLU_Block(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, virtual_batch_size=None, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.output_dim = output_dim
        self.fc = tf.keras.layers.Dense(output_dim * 2, input_shape=(input_dim,))
        self.bn = GBN(output_dim * 2, virtual_batch_size, momentum)

    def call(self, x, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        out = tf.multiply(x[:, :self.output_dim], tf.nn.sigmoid(x[:, self.output_dim:]))
        return out

class FeatureTransformer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, n_independent=2, n_shared=2, virtual_batch_size=None, momentum=0.02):
        super(FeatureTransformer, self).__init__()
        self.n_independent = n_independent
        self.n_shared = n_shared
        
        # Independent layers
        self.independent = [
            GLU_Block(
                feature_dim if i == 0 else feature_dim,
                feature_dim,
                virtual_batch_size,
                momentum
            ) for i in range(n_independent)
        ]
        
        # Shared layers
        self.shared = [
            GLU_Block(
                feature_dim if i == 0 else feature_dim,
                feature_dim,
                virtual_batch_size,
                momentum
            ) for i in range(n_shared)
        ]

    def call(self, x, training=None):
        # Independent GLU Blocks
        out = x
        for layer in self.independent:
            out = layer(out, training=training)
            
        # Shared GLU Blocks
        for layer in self.shared:
            out = layer(out, training=training)
        return out

class FeatTransformer(tf.keras.layers.Layer):
    def __init__(self, feature_dim, virtual_batch_size=None, **kwargs):
        super(FeatTransformer, self).__init__(**kwargs)
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

class Sparsemax(tf.keras.layers.Layer):
    """Sparsemax activation function layer"""
    def __init__(self, axis=-1, **kwargs):
        super(Sparsemax, self).__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.nn.softmax(inputs, axis=self.axis)  # Using softmax for now, will implement true sparsemax later

    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Sparsemax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class AttentiveTransformer(tf.keras.layers.Layer):
    """Attentive transformer layer for TabNet."""
    def __init__(self, feature_dim):
        """Initialize the layer.
        
        Args:
            feature_dim: Dimension of input features
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.dense = tf.keras.layers.Dense(feature_dim)
        self.gbn = GBN(feature_dim)
        self.sparsemax = Sparsemax()

    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: List containing [features, prior_scales]
            training: Whether in training mode
        
        Returns:
            Feature mask
        """
        features, prior_scales = inputs
        batch_size = tf.shape(features)[0]
        
        # Transform features
        transformed = self.dense(features)
        transformed = self.gbn(transformed, training=training)
        
        # Ensure prior_scales has the same feature dimension
        if prior_scales.shape[-1] != transformed.shape[-1]:
            # Project prior_scales to match feature dimension
            prior_scales = tf.expand_dims(prior_scales, -1)
            prior_scales = tf.tile(prior_scales, [1, self.feature_dim])
        
        # Apply mask
        mask = self.sparsemax(transformed * prior_scales)
        return mask

class TabNetEncoder(tf.keras.layers.Layer):
    """TabNet encoder layer."""
    def __init__(self, feature_dim, output_dim, num_decision_steps=5,
                 relaxation_factor=1.5, sparsity_coefficient=1e-5,
                 virtual_batch_size=None, n_independent=2, n_shared=2, momentum=0.02,
                 group_matrix=None):
        """Initialize the encoder.
        
        Args:
            feature_dim: Dimension of input features
            output_dim: Dimension of output
            num_decision_steps: Number of decision steps
            relaxation_factor: Relaxation factor for feature selection
            sparsity_coefficient: Sparsity coefficient for feature selection
            virtual_batch_size: Virtual batch size for ghost batch normalization
            n_independent: Number of independent GLU blocks
            n_shared: Number of shared GLU blocks
            momentum: Momentum for batch normalization
            group_matrix: Matrix specifying feature grouping relationships
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.sparsity_coefficient = sparsity_coefficient
        self.virtual_batch_size = virtual_batch_size
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.momentum = momentum
        
        if group_matrix is not None:
            self.group_matrix = tf.convert_to_tensor(group_matrix, dtype=tf.float32)
        else:
            self.group_matrix = None
            
        # Create layers
        self.input_dense = tf.keras.layers.Dense(feature_dim)
        self.output_dense = tf.keras.layers.Dense(output_dim)
        
    def _process_input(self, inputs):
        """Process input features to match feature_dim."""
        if isinstance(inputs, dict):
            # Process dictionary values
            processed_inputs = []
            for value in inputs.values():
                if len(value.shape) == 1:
                    value = tf.expand_dims(value, -1)
                processed_inputs.append(value)
            x = tf.concat(processed_inputs, axis=1)
        elif isinstance(inputs, list):
            # Process list of tensors
            processed_inputs = []
            for tensor in inputs:
                if len(tensor.shape) == 1:
                    tensor = tf.expand_dims(tensor, -1)
                processed_inputs.append(tensor)
            x = tf.concat(processed_inputs, axis=1)
        else:
            # Single tensor input
            if len(inputs.shape) == 1:
                x = tf.expand_dims(inputs, -1)
            else:
                x = inputs
                
        if x.shape[-1] != self.feature_dim:
            x = self.input_dense(x)
        return x
        
    def _compute_mask(self, prior, att):
        """Compute feature mask.
        
        Args:
            prior: Prior mask
            att: Attention weights
            
        Returns:
            Feature selection mask
        """
        mask = prior * att
        if self.group_matrix is not None:
            # Apply group relationships to mask
            group_att = tf.matmul(mask, self.group_matrix)  # [batch_size, feature_dim]
            # Compute group normalization factors
            group_sums = tf.reduce_sum(self.group_matrix, axis=1)  # [feature_dim]
            group_sums = tf.where(group_sums > 0, group_sums, tf.ones_like(group_sums))
            # Normalize within groups
            group_att = group_att / tf.expand_dims(group_sums + 1e-15, 0)  # [batch_size, feature_dim]
            # Map back to feature space
            mask = tf.matmul(group_att, tf.transpose(self.group_matrix))  # [batch_size, feature_dim]
        # Apply softmax to get final mask
        mask = tf.nn.softmax(mask, axis=-1)
        return mask
        
    def call(self, inputs, training=None):
        x = self._process_input(inputs)
        # Process features and compute masks
        masks = []
        total_entropy = 0
        prior = tf.ones_like(x)
        
        for step in range(self.num_decision_steps):
            mask = self._compute_mask(prior, x)
            masks.append(mask)
            
            # Update prior
            prior = prior * (self.relaxation_factor - mask)
            
            # Compute entropy for sparsity loss
            entropy = -tf.reduce_mean(tf.reduce_sum(mask * tf.math.log(mask + 1e-15), axis=1))
            total_entropy += entropy
            
        # Compute final output
        output = self.output_dense(x)
        sparsity_loss = total_entropy * self.sparsity_coefficient
        
        return output, masks, sparsity_loss

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
    """TabNet model."""
    def __init__(self, feature_dim, output_dim, num_decision_steps=5,
                 relaxation_factor=1.5, sparsity_coefficient=1e-5,
                 virtual_batch_size=None, n_independent=2, n_shared=2, momentum=0.02,
                 group_matrix=None):
        """Initialize TabNet model.
        
        Args:
            feature_dim: Dimension of input features
            output_dim: Dimension of output
            num_decision_steps: Number of decision steps
            relaxation_factor: Relaxation factor for feature selection
            sparsity_coefficient: Sparsity coefficient for feature selection
            virtual_batch_size: Virtual batch size for ghost batch normalization
            n_independent: Number of independent GLU blocks
            n_shared: Number of shared GLU blocks
            momentum: Momentum for batch normalization
            group_matrix: Matrix specifying feature grouping relationships
        """
        super().__init__()
        self.encoder = TabNetEncoder(
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            sparsity_coefficient=sparsity_coefficient,
            virtual_batch_size=virtual_batch_size,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            group_matrix=group_matrix
        )

    def call(self, inputs, training=None):
        output, masks, sparsity_loss = self.encoder(inputs, training=training)
        if training:
            self.add_loss(sparsity_loss)
            return output, masks, sparsity_loss
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