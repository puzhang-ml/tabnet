import tensorflow as tf
import numpy as np

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
        
        # Independent layers
        self.independent = [
            GLU_Block(
                input_dim if i == 0 else output_dim,
                output_dim,
                virtual_batch_size,
                momentum
            ) for i in range(n_independent)
        ]
        
        # Shared layers
        self.shared = [
            GLU_Block(
                input_dim if i == 0 else output_dim,
                output_dim,
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
        mask_type="sparsemax"
    ):
        super(TabNetEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.mask_type = mask_type
        
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
        if self.mask_type == "sparsemax":
            mask = sparsemax(mask)
        return mask 