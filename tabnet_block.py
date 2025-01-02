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
        chunks = tf.split(x, num_or_size_splits=tf.cast(tf.math.ceil(
            tf.shape(x)[0] / self.virtual_batch_size), tf.int32), axis=0)
        res = [self.bn(x_, training=training) for x_ in chunks]
        return tf.concat(res, axis=0)

class GLU_Block(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super(GLU_Block, self).__init__()
        self.output_dim = output_dim
        self.fc = tf.keras.layers.Dense(2 * output_dim, use_bias=False)
        self.bn = GBN(2 * output_dim, virtual_batch_size, momentum)

    def call(self, x, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        x1, x2 = tf.split(x, 2, axis=-1)
        return x1 * tf.sigmoid(x2)

class FeatTransformer(tf.keras.layers.Layer):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_independent,
        n_shared,
        virtual_batch_size=128,
        momentum=0.02
    ):
        super(FeatTransformer, self).__init__()
        self.n_independent = n_independent
        self.n_shared = n_shared
        
        self.independent = [
            GLU_Block(
                input_dim if i == 0 else output_dim,
                output_dim,
                virtual_batch_size,
                momentum
            ) for i in range(n_independent)
        ]
        
        self.shared = [
            GLU_Block(
                input_dim if i == 0 else output_dim,
                output_dim,
                virtual_batch_size,
                momentum
            ) for i in range(n_shared)
        ]

    def call(self, x, training=None):
        out = x
        for layer in self.independent:
            out = layer(out, training=training)
        for layer in self.shared:
            out = layer(out, training=training)
        return out

def sparsemax(z):
    """Sparsemax activation function"""
    z = z - tf.reduce_max(z, axis=-1, keepdims=True)
    z_sorted = tf.sort(z, axis=-1, direction='DESCENDING')
    range_idx = tf.range(1, tf.shape(z)[-1] + 1, dtype=tf.float32)
    bound = 1 + range_idx * z_sorted
    cumsum_zs = tf.cumsum(z_sorted, axis=-1)
    is_gt = tf.cast(bound > cumsum_zs, tf.float32)
    k = tf.reduce_max(range_idx * is_gt, axis=-1, keepdims=True)
    threshold = (tf.reduce_sum(z_sorted * is_gt, axis=-1, keepdims=True) - 1) / k
    return tf.maximum(z - threshold, 0.0)

class TabNetEncoder(tf.keras.Model):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None
    ):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.mask_type = mask_type
        self.group_attention_matrix = group_attention_matrix
        
        self.initial_bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        
        self.encoder = [
            FeatTransformer(
                input_dim,
                n_d + n_a,
                n_independent,
                n_shared,
                virtual_batch_size,
                momentum
            ) for _ in range(n_steps)
        ]
        
        self.decoder = [
            FeatTransformer(
                input_dim,
                n_d + n_a,
                n_independent,
                n_shared,
                virtual_batch_size,
                momentum
            ) for _ in range(n_steps)
        ]

    def call(self, x, training=None):
        x = self.initial_bn(x, training=training)
        
        prior = tf.ones_like(x)
        M_loss = 0
        att = tf.ones((tf.shape(x)[0], self.input_dim))
        steps_output = []
        
        for step in range(self.n_steps):
            M = self._compute_mask(prior, att)
            M_loss += tf.reduce_mean(tf.reduce_sum(M * tf.math.log(M + 1e-10), axis=1))
            
            masked_x = M * x
            encoder_output = self.encoder[step](masked_x, training=training)
            decoder_output = self.decoder[step](masked_x, training=training)
            
            d = tf.sigmoid(decoder_output[:, :self.n_d])
            steps_output.append(d)
            
            if step < self.n_steps - 1:
                att = tf.sigmoid(encoder_output[:, self.n_d:])
                att = self.gamma * att * (1 - M)
                
        M_loss /= self.n_steps
        return steps_output, M_loss

    def _compute_mask(self, prior, att):
        mask = prior * att
        
        if self.group_attention_matrix is not None:
            group_att = tf.matmul(mask, tf.cast(self.group_attention_matrix, mask.dtype))
            mask = tf.matmul(group_att, tf.transpose(tf.cast(self.group_attention_matrix, mask.dtype)))
            
        if self.mask_type == "sparsemax":
            mask = sparsemax(mask)
        return mask

def create_feature_config(feature_dict: Dict[str, tf.Tensor]) -> dict:
    """Create feature configuration from a feature dictionary"""
    config = {}
    total_dims = 0
    
    for feature_name, tensor in feature_dict.items():
        # Always use dynamic shape
        feature_dims = tf.shape(tensor)[-1]
        config[feature_name] = {
            'start_idx': total_dims,
            'end_idx': total_dims + feature_dims,
            'dims': feature_dims
        }
        total_dims += feature_dims
    
    config['total_dims'] = total_dims
    return config

def create_group_matrix(feature_config: dict) -> tf.Tensor:
    """Create group matrix where each feature is treated as one group"""
    # Get total dimensions
    total_dims = feature_config['total_dims']
    n_features = len(feature_config) - 1  # Subtract 1 for 'total_dims' key
    
    # Create empty matrix
    group_matrix = tf.zeros((total_dims, n_features))
    current_pos = 0
    
    # Build the matrix one feature at a time
    for idx, (feature_name, info) in enumerate(feature_config.items()):
        if feature_name != 'total_dims':
            feature_dim = info['dims']
            start_idx = info['start_idx']
            end_idx = info['end_idx']
            
            # Create a mask for this feature
            feature_mask = tf.scatter_nd(
                indices=tf.reshape(tf.range(start_idx, end_idx), [-1, 1]),
                updates=tf.ones(end_idx - start_idx),
                shape=[total_dims]
            )
            
            # Update the corresponding column in the group matrix
            group_matrix = tf.tensor_scatter_nd_update(
                group_matrix,
                indices=tf.stack([tf.range(total_dims), tf.fill([total_dims], idx)], axis=1),
                updates=feature_mask
            )
    
    return group_matrix

class TabNet(tf.keras.Model):
    """TabNet model that handles dictionary inputs and preprocessor outputs"""
    def __init__(
        self,
        feature_config: dict,
        feature_dim: int = 512,
        output_dim: int = 64,
        n_steps: int = 1,
        gamma: float = 1.5,
        momentum: float = 0.98,
        virtual_batch_size: int = 128,
        n_independent: int = 2,
        n_shared: int = 2,
    ):
        super(TabNet, self).__init__()
        self.feature_config = feature_config
        self.feature_processor = FeatureProcessor(feature_config)
        
        # Initialize TabNet without group matrix
        self.tabnet = TabNetEncoder(
            input_dim=feature_config['total_dims'],
            output_dim=output_dim,
            n_d=feature_dim,
            n_a=feature_dim,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum,
            group_attention_matrix=None  # We'll handle this in call()
        )
        
        # Store feature info for group matrix creation
        self.n_features = len(feature_config) - 1  # Subtract 1 for 'total_dims'
        
    def build(self, input_shape):
        # Create group matrix once actual dimensions are known
        self.group_matrix = self.add_weight(
            name='group_matrix',
            shape=(self.feature_config['total_dims'], self.n_features),
            initializer=self.create_group_matrix_initializer(),
            trainable=False
        )
        super().build(input_shape)
    
    def create_group_matrix_initializer(self):
        def initializer(shape, dtype=None):
            matrix = tf.zeros(shape)
            for idx, (feature_name, info) in enumerate(self.feature_config.items()):
                if feature_name != 'total_dims':
                    start_idx = info['start_idx']
                    end_idx = info['end_idx']
                    matrix = matrix + tf.scatter_nd(
                        indices=[[i, idx] for i in range(start_idx, end_idx)],
                        updates=tf.ones(end_idx - start_idx),
                        shape=shape
                    )
            return matrix
        return initializer
        
    def call(self, inputs: Dict[str, tf.Tensor], training=None):
        # Process dictionary input into tensor
        x_processed = self.feature_processor.process_features(inputs)
        
        # Update TabNet's group matrix
        self.tabnet.group_attention_matrix = self.group_matrix
        
        # Get predictions
        steps_output, M_loss = self.tabnet(x_processed, training=training)
        return steps_output[-1]  # Return last step output

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