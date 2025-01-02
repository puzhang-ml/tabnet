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

class GLU_Block(tf.keras.layers.Layer):
    """Gated Linear Unit block."""

    def __init__(
        self,
        feature_dim,
        output_dim,
        virtual_batch_size=None,
        momentum=0.02,
        epsilon=1e-5,
        **kwargs
    ):
        """Initialize the GLU block.

        Args:
            feature_dim: Number of input features.
            output_dim: Dimension of output.
            virtual_batch_size: Virtual batch size for ghost batch normalization.
            momentum: Momentum for batch normalization.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize layers
        self.dense = tf.keras.layers.Dense(
            output_dim * 2,  # Double for GLU
            use_bias=False,
            kernel_initializer='glorot_uniform'
        )

        self.bn = GBN(
            feature_dim=output_dim * 2,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            epsilon=epsilon
        )

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.

        Returns:
            GLU activated features.
        """
        x = self.dense(inputs)
        x = self.bn(x, training=training)
        
        # GLU activation
        out, gate = tf.split(x, 2, axis=-1)
        return out * tf.nn.sigmoid(gate)

class FeatureTransformer(tf.keras.layers.Layer):
    """Feature transformer layer."""

    def __init__(
        self,
        feature_dim,
        output_dim,
        virtual_batch_size=None,
        momentum=0.02,
        epsilon=1e-5,
        **kwargs
    ):
        """Initialize the feature transformer.

        Args:
            feature_dim: Number of input features.
            output_dim: Dimension of output.
            virtual_batch_size: Virtual batch size for ghost batch normalization.
            momentum: Momentum for batch normalization.
            epsilon: Small constant for numerical stability.
            **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.epsilon = epsilon

        # Initialize GLU blocks
        self.glu_block = GLU_Block(
            feature_dim=feature_dim,
            output_dim=output_dim,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            epsilon=epsilon
        )

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: Input tensor.
            training: Whether in training mode.

        Returns:
            Transformed features.
        """
        return self.glu_block(inputs, training=training)

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
    """TabNet encoder module."""

    def __init__(
        self,
        feature_dim,
        output_dim,
        num_decision_steps,
        relaxation_factor,
        bn_virtual_bs=None,
        bn_momentum=0.02,
        epsilon=1e-5,
        grouped_features=None
    ):
        """Initialize TabNet encoder.
        
        Args:
            feature_dim: Dimension of the features.
            output_dim: Dimension of the output.
            num_decision_steps: Number of sequential decision steps.
            relaxation_factor: Relaxation factor that promotes the reuse of each feature.
            bn_virtual_bs: Virtual batch size for ghost batch normalization.
            bn_momentum: Momentum for batch normalization.
            epsilon: Small constant for numerical stability.
            grouped_features: List of lists of feature indices to be masked together.
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_decision_steps = num_decision_steps
        self.relaxation_factor = relaxation_factor
        self.grouped_features = grouped_features
        
        # Create group matrix if grouped_features is provided
        if grouped_features is not None:
            group_matrix = np.zeros((feature_dim, feature_dim))
            for group in grouped_features:
                for i in group:
                    for j in group:
                        group_matrix[i, j] = 1
            self.group_matrix = tf.constant(group_matrix, dtype=tf.float32)
        else:
            self.group_matrix = None
        
        # Initialize batch normalization
        self.initial_bn = GBN(
            feature_dim=feature_dim,
            virtual_batch_size=bn_virtual_bs,
            momentum=bn_momentum,
            epsilon=epsilon
        )
        
        # Initialize feature transformers
        self.feature_transformers = []
        for step in range(num_decision_steps):
            transformer = FeatureTransformer(
                feature_dim=feature_dim,
                output_dim=output_dim,
                virtual_batch_size=bn_virtual_bs,
                momentum=bn_momentum,
                epsilon=epsilon
            )
            self.feature_transformers.append(transformer)
            
    def _compute_mask(self, prior, att):
        """Compute feature mask.
        
        Args:
            prior: Prior mask from previous decision step.
            att: Attention weights.
            
        Returns:
            Feature mask.
        """
        mask = prior * att
        if self.group_matrix is not None:
            # Project mask to group space
            group_matrix = tf.cast(self.group_matrix, mask.dtype)
            group_att = tf.matmul(mask, group_matrix)
            # Compute group sums and normalize
            group_sums = tf.reduce_sum(group_matrix, axis=1)  # [feature_dim]
            group_sums = tf.expand_dims(group_sums, 0)  # [1, feature_dim]
            group_sums = tf.tile(group_sums, [tf.shape(mask)[0], 1])  # [batch_size, feature_dim]
            group_att = group_att / (group_sums + 1e-15)
            # Project back to feature space
            mask = tf.matmul(group_att, tf.transpose(group_matrix))
        mask = tf.nn.softmax(mask, axis=-1)
        return mask
        
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor or dictionary of tensors.
            training: Whether in training mode.
            
        Returns:
            Tuple of (output, masks, sparsity_loss).
        """
        if isinstance(inputs, dict):
            # Concatenate all features
            x = tf.concat([
                tf.reshape(v, [tf.shape(v)[0], -1])
                for v in inputs.values()
            ], axis=1)
        elif isinstance(inputs, (list, tuple)):
            # Concatenate list of features
            x = tf.concat([
                tf.reshape(v, [tf.shape(v)[0], -1])
                for v in inputs
            ], axis=1)
        else:
            # Handle single input tensor
            if len(inputs.shape) == 1:
                x = tf.expand_dims(inputs, -1)
            else:
                x = inputs
            
        # Initial batch normalization
        x = self.initial_bn(x, training=training)
        
        # Initialize attention and output
        prior = tf.ones_like(x)
        total_entropy = 0
        masks = []
        
        # Process through decision steps
        for step in range(self.num_decision_steps):
            # Compute attention
            att = tf.ones_like(x)
            
            # Compute mask
            mask = self._compute_mask(prior, att)
            masks.append(mask)
            
            # Apply mask and transform features
            masked_x = mask * x
            out = self.feature_transformers[step](masked_x, training=training)
            
            # Update prior
            prior = prior * (self.relaxation_factor - mask)
            
            # Compute entropy for sparsity loss
            total_entropy += tf.reduce_mean(tf.reduce_sum(
                -mask * tf.math.log(mask + 1e-15), axis=1
            ))
            
        sparsity_loss = total_entropy / self.num_decision_steps
        return out, masks, sparsity_loss

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

    def __init__(
        self,
        feature_dim,
        output_dim,
        num_decision_steps=5,
        relaxation_factor=1.5,
        sparsity_coefficient=1e-5,
        bn_virtual_bs=None,
        bn_momentum=0.02,
        epsilon=1e-5,
        grouped_features=None
    ):
        """Initialize TabNet model.
        
        Args:
            feature_dim: Dimension of the features.
            output_dim: Dimension of the output.
            num_decision_steps: Number of sequential decision steps.
            relaxation_factor: Relaxation factor that promotes the reuse of each feature.
            sparsity_coefficient: Coefficient for the sparsity regularization.
            bn_virtual_bs: Virtual batch size for ghost batch normalization.
            bn_momentum: Momentum for batch normalization.
            epsilon: Small constant for numerical stability.
            grouped_features: List of lists of feature indices to be masked together.
        """
        super().__init__()
        self.sparsity_coefficient = sparsity_coefficient
        
        # Initialize encoder
        self.encoder = TabNetEncoder(
            feature_dim=feature_dim,
            output_dim=output_dim,
            num_decision_steps=num_decision_steps,
            relaxation_factor=relaxation_factor,
            bn_virtual_bs=bn_virtual_bs,
            bn_momentum=bn_momentum,
            epsilon=epsilon,
            grouped_features=grouped_features
        )
        
    def call(self, inputs, training=None):
        """Forward pass.
        
        Args:
            inputs: Input tensor or dictionary of tensors.
            training: Whether in training mode.
            
        Returns:
            Tuple of (output, masks, sparsity_loss) in training mode,
            or just output in inference mode.
        """
        output, masks, sparsity_loss = self.encoder(inputs, training=training)
        if training:
            return output, masks, sparsity_loss * self.sparsity_coefficient
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