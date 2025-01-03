"""
Modified from the original TabNet implementation by DreamQuark:
https://github.com/dreamquark-ai/tabnet

This implementation adapts the PyTorch version to TensorFlow and includes
modifications for dynamic feature grouping and attention mechanisms.

MIT License

Copyright (c) 2019 DreamQuark

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple

class GBN(tf.keras.layers.Layer):
    """Ghost Batch Normalization
    
    This layer implements Ghost Batch Normalization, which enables the network to achieve
    the benefits of large batch training while using small batch sizes.
    """
    def __init__(self, feature_dim, virtual_batch_size=None, momentum=0.02, epsilon=1e-5):
        """Initialize Ghost Batch Normalization layer.
        
        Args:
            feature_dim: Dimension of the features.
            virtual_batch_size: Size of virtual batches.
            momentum: Momentum for the moving average.
            epsilon: Small constant for numerical stability.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=momentum,
            epsilon=epsilon
        )
        
    def call(self, x, training=None):
        """Forward pass.
        
        Args:
            x: Input tensor.
            training: Whether in training mode.
            
        Returns:
            Normalized tensor.
        """
        if not training or self.virtual_batch_size is None:
            return self.bn(x, training=training)
            
        # Calculate number of virtual batches
        batch_size = tf.shape(x)[0]
        split_size = self.virtual_batch_size
        
        if batch_size <= split_size:
            return self.bn(x, training=training)
            
        # Reshape input into virtual batches
        n_vbs = tf.cast(tf.math.ceil(batch_size / split_size), tf.int32)
        
        # Pad input if needed
        pad_size = n_vbs * split_size - batch_size
        if pad_size > 0:
            paddings = [[0, pad_size], [0, 0]]
            x = tf.pad(x, paddings)
            
        # Reshape into virtual batches
        x_reshaped = tf.reshape(x, [n_vbs, split_size, self.feature_dim])
        
        # Apply batch norm to each virtual batch
        def norm_fn(x_vb):
            return self.bn(x_vb, training=training)
            
        x_normalized = tf.map_fn(norm_fn, x_reshaped)
        
        # Reshape back and remove padding if needed
        x_out = tf.reshape(x_normalized, [-1, self.feature_dim])
        if pad_size > 0:
            x_out = x_out[:batch_size]
            
        return x_out

def glu(x):
    """Gated Linear Unit activation function.
    
    Args:
        x: Input tensor. The last dimension is split in half to create
           the gate and the linear transformation.
           
    Returns:
        Output tensor with half the size of the last dimension.
    """
    # Split the input into two parts along the last axis
    a, b = tf.split(x, 2, axis=-1)
    return a * tf.sigmoid(b)

class GLUBlock(tf.keras.layers.Layer):
    """Gated Linear Unit block."""

    def __init__(
        self,
        feature_dim,
        output_dim,
        momentum=0.7,
        epsilon=1e-5,
        **kwargs
    ):
        """Initialize GLU block.

        Args:
            feature_dim: The dimension of the input features.
            output_dim: The dimension of the output features.
            momentum: Batch normalization momentum.
            epsilon: Batch normalization epsilon.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim

        # Initialize layers
        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            epsilon=epsilon
        )

        # Dense layer for feature transformation
        self.dense = tf.keras.layers.Dense(
            output_dim * 2,  # Double size for gating
            use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: Input features.
            training: Whether in training mode.

        Returns:
            Transformed features.
        """
        x = self.bn(inputs, training=training)
        x = self.dense(x)

        # Split into two parts for gating
        features, gates = tf.split(x, 2, axis=-1)

        # Apply gating using sigmoid activation
        return features * tf.sigmoid(gates)

class FeatureTransformer(tf.keras.layers.Layer):
    """Feature transformer module."""

    def __init__(self, feature_dim, output_dim=None, momentum=0.98, epsilon=1e-15):
        """Initialize feature transformer.

        Args:
            feature_dim: The dimension of input features.
            output_dim: The dimension of output features. If None, use feature_dim.
            momentum: Momentum for batch normalization.
            epsilon: Epsilon for batch normalization.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim or feature_dim
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize GLU blocks
        self.glu_block_1 = GLUBlock(
            feature_dim=feature_dim,
            output_dim=self.output_dim,
            momentum=momentum,
            epsilon=epsilon
        )
        self.glu_block_2 = GLUBlock(
            feature_dim=self.output_dim,
            output_dim=self.output_dim,
            momentum=momentum,
            epsilon=epsilon
        )

    def call(self, inputs, prior_scales=None, training=None):
        """Forward pass.

        Args:
            inputs: Input features.
            prior_scales: Optional prior scales to apply.
            training: Whether in training mode.

        Returns:
            Transformed features.
        """
        # Ensure inputs have correct shape
        x = inputs
        if len(x.shape) == 1:
            x = tf.expand_dims(x, -1)
        if len(x.shape) == 2:
            x = tf.expand_dims(x, -1)

        # Apply GLU blocks
        x = self.glu_block_1(x, training=training)
        x = self.glu_block_2(x, training=training)

        # Apply prior scales if provided
        if prior_scales is not None:
            # Ensure prior_scales has correct shape
            prior_scales = tf.cast(prior_scales, x.dtype)
            if len(prior_scales.shape) == 2:
                prior_scales = tf.expand_dims(prior_scales, -1)
            x = x * prior_scales

        return x

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'momentum': self.momentum,
            'epsilon': self.epsilon
        })
        return config

class FeatTransformer(tf.keras.layers.Layer):
    """Feature transformer module."""

    def __init__(
        self,
        feature_dim,
        num_features=None,
        virtual_batch_size=None,
        momentum=0.02,
        epsilon=1e-5,
        **kwargs
    ):
        """Initialize feature transformer.
        
        Args:
            feature_dim: Dimension of features.
            num_features: Number of input features.
            virtual_batch_size: Virtual batch size for ghost batch normalization.
            momentum: Momentum for batch normalization.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional layer arguments.
        """
        self.feature_dim = feature_dim
        self.num_features = num_features
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self._epsilon = epsilon  # Store epsilon as instance variable
        super(FeatTransformer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the feature transformer.
        
        Args:
            input_shape: Input shape.
        """
        self.fc = tf.keras.layers.Dense(
            2 * self.feature_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

        if self.virtual_batch_size is None:
            self.bn = tf.keras.layers.BatchNormalization(
                momentum=self.momentum,
                epsilon=self._epsilon
            )
        else:
            self.bn = GBN(
                feature_dim=2 * self.feature_dim,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                epsilon=self._epsilon
            )

        self.built = True

    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor.
            training: Whether in training mode.
            
        Returns:
            Transformed features.
        """
        x = self.fc(inputs)
        x = self.bn(x, training=training)
        return tf.nn.glu(x)

def sparsemax(logits):
    """Sparsemax activation function."""
    # Sort logits in descending order
    sorted_logits = tf.sort(logits, direction='DESCENDING', axis=-1)
    
    # Calculate cumulative sums
    cum_sums = tf.cumsum(sorted_logits, axis=-1)
    
    # Calculate position where sum - 1 crosses sorted logits
    k = tf.range(1, tf.shape(logits)[-1] + 1, dtype=logits.dtype)
    sorted_logits_threshold = sorted_logits - (cum_sums - 1) / k
    
    # Find number of elements to keep
    max_k = tf.reduce_sum(
        tf.cast(sorted_logits_threshold > 0, tf.int32),
        axis=-1,
        keepdims=True
    )
    
    # Calculate threshold
    threshold = tf.gather(sorted_logits_threshold, max_k - 1, batch_dims=1)
    
    # Apply threshold to get sparsemax
    return tf.maximum(logits - threshold, 0.0)

def entmax15(inputs, axis=-1):
    """EntMax 1.5 activation function."""
    # Sort inputs in descending order
    input_sorted = tf.sort(inputs, direction='DESCENDING', axis=axis)
    
    # Calculate running sums
    running_sum = tf.cumsum(input_sorted, axis=axis)
    rho = tf.range(1, tf.shape(input_sorted)[axis] + 1, dtype=inputs.dtype)
    
    # Calculate threshold
    threshold = input_sorted - ((running_sum - 1) / rho)
    
    # Find last positive threshold
    is_positive = threshold > 0
    max_k = tf.reduce_sum(tf.cast(is_positive, tf.int32), axis=axis, keepdims=True)
    
    # Get threshold value
    threshold_value = tf.gather(threshold, max_k - 1, batch_dims=1)
    
    # Apply threshold
    return tf.maximum(0., inputs - threshold_value)

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
    """Attentive transformer layer."""

    def __init__(
        self,
        feature_dim,
        relaxation_factor=1.5,
        bn_epsilon=1e-5,
        bn_momentum=0.7,
        **kwargs
    ):
        """Initialize the attentive transformer.

        Args:
            feature_dim: The dimension of the features.
            relaxation_factor: Relaxation factor for feature selection.
            bn_epsilon: Batch normalization epsilon.
            bn_momentum: Batch normalization momentum.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.relaxation_factor = relaxation_factor

        # Initialize layers
        self.bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            epsilon=bn_epsilon,
            momentum=bn_momentum
        )

        # Use same feature dimension for input and output
        self.transform = tf.keras.layers.Dense(
            feature_dim,
            use_bias=False,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def call(self, features, prior_scales, training=None):
        """Forward pass.

        Args:
            features: Input features.
            prior_scales: Prior scales for feature selection.
            training: Whether in training mode.

        Returns:
            Attention mask.
        """
        x = self.bn(features, training=training)
        x = self.transform(x)
        x = tf.nn.softmax(x, axis=-1)
        return x

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'relaxation_factor': self.relaxation_factor
        })
        return config

class TabNetEncoder(tf.keras.layers.Layer):
    def __init__(
        self,
        feature_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        epsilon=1e-5,
        momentum=0.7,
        virtual_batch_size=None,
        grouped_features=None,
        mask_type="sparsemax",
        **kwargs
    ):
        """TabNet encoder.
        
        Args:
            feature_dim: Number of input features.
            output_dim: Number of output dimensions.
            n_d: Width of the decision prediction layer.
            n_a: Width of the attention embedding for each mask.
            n_steps: Number of successive steps in the network.
            gamma: Coefficient for feature reusage in masks.
            epsilon: Small constant to avoid numerical instability.
            momentum: Momentum for batch normalization.
            virtual_batch_size: Virtual batch size for ghost batch normalization.
            grouped_features: List of list of ints, groups of features that share attention.
            mask_type: Type of mask to use.
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.grouped_features = grouped_features
        self.virtual_batch_size = virtual_batch_size
        self.mask_type = mask_type

        # Initialize layers
        self.initial_bn = tf.keras.layers.BatchNormalization(
            momentum=self.momentum,
            epsilon=self.epsilon
        )

        # Initialize encoder steps
        self.encoder_steps = []
        for _ in range(n_steps):
            self.encoder_steps.append(
                FeatureTransformer(
                    feature_dim=feature_dim,
                    output_dim=n_d + n_a,
                    momentum=momentum,
                    epsilon=epsilon
                )
            )

        # Create feature transformer
        self.feature_transformer = FeatureTransformer(
            feature_dim=feature_dim,
            output_dim=n_d + n_a,
            momentum=momentum,
            epsilon=epsilon
        )

    def _compute_mask(self, features, prior_scales=None):
        """Compute feature selection mask.

        Args:
            features: Input features.
            prior_scales: Optional prior scales from previous step.

        Returns:
            Feature selection mask.
        """
        if prior_scales is None:
            prior_scales = tf.ones_like(features)

        # Transform features
        transformed = self.feature_transformer(
            features,
            prior_scales=prior_scales,
            training=True
        )

        # Compute mask
        mask = tf.keras.activations.relu(transformed)
        
        # Apply feature grouping if specified
        if self.grouped_features is not None:
            # Create a new mask tensor with same shape
            grouped_mask = tf.zeros_like(mask)
            
            # For each group, compute the mean mask value
            for group in self.grouped_features:
                # Convert group indices to tensor
                group_indices = tf.constant(group)
                
                # Gather mask values for the group
                group_mask = tf.gather(mask, group_indices, axis=1)
                
                # Compute mean mask value for the group
                mean_mask = tf.reduce_mean(group_mask, axis=1, keepdims=True)
                
                # Scatter mean mask back to all features in the group
                for idx in group:
                    grouped_mask = tf.tensor_scatter_nd_update(
                        grouped_mask,
                        tf.constant([[i, idx] for i in range(tf.shape(mask)[0])]),
                        tf.reshape(mean_mask, [-1])
                    )
            
            # Use grouped mask instead of original
            mask = grouped_mask

        # Normalize mask
        mask = mask / (tf.reduce_sum(mask, axis=1, keepdims=True) + self.epsilon)
        
        # Apply sparsemax or entmax
        if self.mask_type == "sparsemax":
            mask = sparsemax(mask)
        elif self.mask_type == "entmax":
            mask = entmax15(mask)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")
        
        # Relax mask
        if prior_scales is not None:
            mask = self.gamma * mask * prior_scales
            
        return mask

    def call(self, features, training=None):
        """Forward pass.

        Args:
            features: Input features (dict, list, or tensor).
            training: Whether in training mode.

        Returns:
            Tuple of (output, attention_masks, encoded_features).
        """
        bs = tf.shape(features)[0]
        
        # Initial batch norm
        features = self.initial_bn(features, training=training)

        # Prepare attention masks container
        attention_masks = []
        prior_scales = None
        
        # Apply encoder steps
        for step in range(self.n_steps):
            # Compute mask
            mask = self._compute_mask(features, prior_scales)
            attention_masks.append(mask)

            # Apply mask
            masked_features = features * mask
            
            # Update prior scales
            prior_scales = mask
            
            # Apply encoder
            features = self.encoder_steps[step](masked_features, training=training)

        # Reshape attention masks for output
        attention_masks = tf.stack(attention_masks, axis=1)
        
        return features, attention_masks

    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'grouped_features': self.grouped_features
        })
        return config

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

def infer_feature_groups_from_dict(inputs: Dict[str, tf.Tensor]) -> List[List[int]]:
    """Infer feature groups from input dictionary structure.
    
    Args:
        inputs: Dictionary of input tensors where keys indicate feature type
               (e.g. 'embedding_1', 'numeric_1', 'categorical_1')
               
    Returns:
        List of feature index groups where features with same prefix are grouped together
        
    Example:
        inputs = {
            'embedding_1': tensor(batch_size, 16),
            'embedding_2': tensor(batch_size, 16),
            'numeric_1': tensor(batch_size, 1),
            'numeric_2': tensor(batch_size, 1),
            'categorical_1': tensor(batch_size, 8)
        }
        
        Returns: [
            [0, 1, ..., 31],  # embedding features (32 total)
            [32, 33],         # numeric features (2 total)
            [34, ..., 41]     # categorical features (8 total)
        ]
    """
    current_idx = 0
    prefix_to_indices = {}
    
    # Sort keys to ensure consistent ordering
    for key in sorted(inputs.keys()):
        tensor = inputs[key]
        
        # Get feature dimension
        if isinstance(tensor, tf.TensorShape):
            feature_dim = tensor[-1]
        else:
            feature_dim = tensor.shape[-1]
            
        # Get prefix (e.g. 'embedding' from 'embedding_1')
        prefix = key.split('_')[0].lower()
        
        if prefix not in prefix_to_indices:
            prefix_to_indices[prefix] = []
            
        # Add all indices for this feature
        feature_indices = list(range(current_idx, current_idx + feature_dim))
        prefix_to_indices[prefix].extend(feature_indices)
        current_idx += feature_dim
    
    # Only return groups with multiple features
    return [indices for indices in prefix_to_indices.values() if len(indices) > 1]

class TabNet(tf.keras.Model):
    def __init__(
        self,
        feature_dim=None,  # Made optional since we can infer from dict
        output_dim=1,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        epsilon=1e-5,
        momentum=0.7,
        grouped_features=None,
        **kwargs
    ):
        """Initialize TabNet model.
        
        Args:
            feature_dim: Optional total dimension of features. If None, will be inferred from first input.
            output_dim: The dimension of the output features.
            n_d: Width of the decision prediction layer.
            n_a: Width of the attention embedding for each mask.
            n_steps: Number of steps in the network.
            gamma: Coefficient for feature reusage in masks.
            epsilon: Small constant for numerical stability.
            momentum: Momentum for batch normalization.
            grouped_features: Optional list of feature groups.
        """
        super().__init__(**kwargs)
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.momentum = momentum
        self.grouped_features = grouped_features
        self.encoder = None  # Will be initialized on first call
        
        # Initialize output layer
        self.output_layer = tf.keras.layers.Dense(
            output_dim,
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )

    def _infer_feature_dim(self, features):
        """Infer total feature dimension from input features."""
        if isinstance(features, dict):
            return sum(tensor.shape[-1] for tensor in features.values())
        elif isinstance(features, (list, tuple)):
            return sum(tensor.shape[-1] for tensor in features)
        else:
            return features.shape[-1]

    def _preprocess_features(self, features):
        """Preprocess input features.

        Args:
            features: Input features (dict, list, or tensor).

        Returns:
            Preprocessed features tensor.
        """
        if isinstance(features, dict):
            # Process dictionary inputs
            processed_features = []
            for key in sorted(features.keys()):  # Sort keys for consistent ordering
                feature = features[key]
                if len(tf.shape(feature)) == 1:
                    feature = tf.expand_dims(feature, -1)
                processed_features.append(feature)
            features = tf.concat(processed_features, axis=-1)
        elif isinstance(features, (list, tuple)):
            # Process list/tuple inputs
            processed_features = []
            for feature in features:
                if len(tf.shape(feature)) == 1:
                    feature = tf.expand_dims(feature, -1)
                processed_features.append(feature)
            features = tf.concat(processed_features, axis=-1)
        elif len(tf.shape(features)) == 1:
            # Handle single scalar feature
            features = tf.expand_dims(features, -1)

        return features

    def _initialize_encoder(self, feature_dim):
        """Initialize encoder with correct feature dimension."""
        self.encoder = TabNetEncoder(
            feature_dim=feature_dim,
            output_dim=feature_dim,
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            epsilon=self.epsilon,
            momentum=self.momentum,
            grouped_features=self.grouped_features
        )

    def call(self, features, training=None):
        """Forward pass.

        Args:
            features: Input features (dict, list, or tensor).
            training: Whether in training mode.

        Returns:
            Tuple of (output, attention_masks, encoded_features).
        """
        # Infer feature groups from dictionary input if not provided
        if isinstance(features, dict) and self.grouped_features is None:
            self.grouped_features = infer_feature_groups_from_dict(features)
        
        # Preprocess features
        processed_features = self._preprocess_features(features)
        
        # Initialize encoder if not done yet or if feature_dim changed
        current_feature_dim = processed_features.shape[-1]
        if self.encoder is None or (self.feature_dim is not None and current_feature_dim != self.feature_dim):
            self.feature_dim = current_feature_dim
            self._initialize_encoder(current_feature_dim)
            # Update encoder's grouped_features
            if self.grouped_features is not None:
                self.encoder.grouped_features = self.grouped_features

        # Encode features
        encoded_features, attention_masks = self.encoder(processed_features, training=training)

        # Generate output
        output = self.output_layer(encoded_features)

        return output, attention_masks, encoded_features

    def get_config(self):
        """Get model configuration."""
        config = super().get_config()
        config.update({
            'feature_dim': self.feature_dim,
            'output_dim': self.output_dim,
            'n_d': self.n_d,
            'n_a': self.n_a,
            'n_steps': self.n_steps,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'momentum': self.momentum,
            'grouped_features': self.grouped_features
        })
        return config

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

class GhostBatchNormalization(tf.keras.layers.Layer):
    """Ghost Batch Normalization layer."""

    def __init__(
        self,
        virtual_batch_size,
        momentum=0.02,
        epsilon=1e-5,
        **kwargs
    ):
        """Initialize Ghost Batch Normalization.
        
        Args:
            virtual_batch_size: Virtual batch size.
            momentum: Momentum for batch normalization.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional layer arguments.
        """
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self._epsilon = epsilon  # Store epsilon as instance variable
        super(GhostBatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        """Build the Ghost Batch Normalization layer.
        
        Args:
            input_shape: Input shape.
        """
        self.feature_dim = input_shape[-1]
        
        # Initialize moving statistics
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.feature_dim,),
            initializer='zeros',
            trainable=False
        )
        self.moving_variance = self.add_weight(
            name='moving_variance',
            shape=(self.feature_dim,),
            initializer='ones',
            trainable=False
        )
        
        # Initialize scale and offset
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.feature_dim,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.feature_dim,),
            initializer='zeros',
            trainable=True
        )

        self.built = True

    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor.
            training: Whether in training mode.
            
        Returns:
            Normalized features.
        """
        if training:
            # Compute statistics for each virtual batch
            batch_size = tf.shape(inputs)[0]
            n_vbs = tf.cast(tf.math.ceil(batch_size / self.virtual_batch_size), tf.int32)
            
            mean_vbs = []
            var_vbs = []
            
            for i in range(n_vbs):
                start_idx = i * self.virtual_batch_size
                end_idx = tf.minimum(start_idx + self.virtual_batch_size, batch_size)
                vb = inputs[start_idx:end_idx]
                
                mean_vb = tf.reduce_mean(vb, axis=0)
                var_vb = tf.reduce_variance(vb, axis=0)
                
                mean_vbs.append(mean_vb)
                var_vbs.append(var_vb)
            
            # Compute batch statistics
            batch_mean = tf.reduce_mean(mean_vbs, axis=0)
            batch_variance = tf.reduce_mean(var_vbs, axis=0)
            
            # Update moving statistics
            self.moving_mean.assign(
                self.moving_mean * (1 - self.momentum) + batch_mean * self.momentum
            )
            self.moving_variance.assign(
                self.moving_variance * (1 - self.momentum) + batch_variance * self.momentum
            )
            
            # Normalize using virtual batch statistics
            vb_stats = []
            for i in range(n_vbs):
                start_idx = i * self.virtual_batch_size
                end_idx = tf.minimum(start_idx + self.virtual_batch_size, batch_size)
                vb = inputs[start_idx:end_idx]
                
                vb_norm = (vb - mean_vbs[i]) / tf.sqrt(var_vbs[i] + self._epsilon)
                vb_stats.append(vb_norm)
            
            x = tf.concat(vb_stats, axis=0)
        else:
            # Use moving statistics for inference
            x = (inputs - self.moving_mean) / tf.sqrt(self.moving_variance + self._epsilon)
        
        # Apply scale and offset
        return x * self.gamma + self.beta