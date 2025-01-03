import tensorflow as tf
import numpy as np

def initialize_weights(shape):
    """Initialize weights as in DreamQuark's implementation."""
    return tf.random.normal(shape) / tf.sqrt(tf.cast(shape[0], tf.float32))

class GBN(tf.keras.layers.Layer):
    """Ghost Batch Normalization."""
    def __init__(self, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.virtual_batch_size = virtual_batch_size
        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum)

    def call(self, x, training=None):
        if training and self.virtual_batch_size is not None:
            batch_size = tf.shape(x)[0]
            n_splits = batch_size // self.virtual_batch_size
            if n_splits > 0:
                chunks = tf.split(x[:n_splits * self.virtual_batch_size], n_splits)
                res = [self.bn(x_batch, training=training) for x_batch in chunks]
                # Handle remaining samples
                remainder = x[n_splits * self.virtual_batch_size:]
                if tf.shape(remainder)[0] > 0:
                    res.append(self.bn(remainder, training=training))
                return tf.concat(res, axis=0)
        return self.bn(x, training=training)

class SharedBlock(tf.keras.layers.Layer):
    """Exactly matches DreamQuark's shared block."""
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(
            output_dim,
            use_bias=False,
            kernel_initializer=lambda shape, dtype: initialize_weights(shape)
        )
        self.bn1 = GBN(virtual_batch_size, momentum)
        
        self.fc2 = tf.keras.layers.Dense(
            output_dim,
            use_bias=False,
            kernel_initializer=lambda shape, dtype: initialize_weights(shape)
        )
        self.bn2 = GBN(virtual_batch_size, momentum)

    def call(self, x, training=None):
        x = self.fc1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        x = self.bn2(x, training=training)
        return x

class GLU(tf.keras.layers.Layer):
    """Gated Linear Unit exactly as in DreamQuark."""
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.fc = tf.keras.layers.Dense(
            2 * output_dim,
            use_bias=False,
            kernel_initializer=lambda shape, dtype: initialize_weights(shape)
        )
        self.bn = GBN(virtual_batch_size, momentum)

    def call(self, x, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        out, gate = tf.split(x, 2, axis=-1)
        return out * tf.nn.sigmoid(gate)

class FeatTransformer(tf.keras.layers.Layer):
    """Feature transformer exactly as in DreamQuark."""
    def __init__(self, input_dim, output_dim, shared_block=None, n_glu=2,
                 virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.shared = shared_block
        self.n_glu = n_glu
        self.output_dim = output_dim
        
        # Specific GLU layers
        self.specifics = []
        for _ in range(n_glu):
            self.specifics.append(GLU(input_dim, output_dim, 
                                    virtual_batch_size, momentum))

    def call(self, x, training=None):
        # DreamQuark doesn't apply shared here since it's applied in forward_step
        for glu in self.specifics:
            x = glu(x, training=training)
        return x

class AttentiveTransformer(tf.keras.layers.Layer):
    """Attentive transformer exactly as in DreamQuark."""
    def __init__(self, input_dim, output_dim, virtual_batch_size=128, momentum=0.02):
        super().__init__()
        self.fc = tf.keras.layers.Dense(
            output_dim,
            use_bias=False,
            kernel_initializer=lambda shape, dtype: initialize_weights(shape)
        )
        self.bn = GBN(virtual_batch_size, momentum)

    def call(self, x, prior=None, training=None):
        x = self.fc(x)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = sparsemax(x)
        if prior is not None:
            x = x * prior
        return x

def sparsemax(logits, axis=-1):
    """Sparsemax activation function as used in DreamQuark implementation."""
    logits = tf.convert_to_tensor(logits)
    
    # Sort and calculate cumulative sum
    sorted_logits = tf.sort(logits, axis=axis, direction='DESCENDING')
    cumsum = tf.cumsum(sorted_logits, axis=axis)
    
    # Calculate position where sum - 1 crosses sorted logits
    rho = tf.range(1, tf.shape(logits)[axis] + 1, dtype=logits.dtype)
    sorted_logits_threshold = sorted_logits - (cumsum - 1) / rho
    
    # Count number of elements above threshold
    max_k = tf.reduce_sum(tf.cast(sorted_logits_threshold > 0, tf.int32), axis=axis)
    
    # Calculate threshold
    threshold = tf.gather(sorted_logits_threshold, max_k - 1, batch_dims=1)
    
    return tf.maximum(logits - threshold, 0.0)

class TabNetEncoder(tf.keras.layers.Layer):
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
        feature_groups=None
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        
        # Initial batch norm
        self.initial_bn = tf.keras.layers.BatchNormalization(momentum=momentum)
        
        # Shared across decision steps
        self.shared = SharedBlock(input_dim, n_d + n_a, virtual_batch_size, momentum)
        
        # Initial splitter (special first step)
        self.initial_splitter = FeatTransformer(
            input_dim=input_dim,
            output_dim=n_d + n_a,
            shared_block=None,
            virtual_batch_size=virtual_batch_size,
            momentum=momentum
        )
        
        # Feature transformers for subsequent steps
        self.feat_transformers = []
        self.att_transformers = []
        
        for _ in range(n_steps - 1):  # One less because of initial splitter
            self.feat_transformers.append(
                FeatTransformer(
                    input_dim=input_dim,
                    output_dim=n_d + n_a,
                    shared_block=self.shared,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum
                )
            )
            
            self.att_transformers.append(
                AttentiveTransformer(
                    input_dim=n_a,
                    output_dim=input_dim,
                    virtual_batch_size=virtual_batch_size,
                    momentum=momentum
                )
            )
            
        self.feature_groups = feature_groups
        
    def forward_step(self, x, step, prior, training=None):
        # First step is special
        if step == 0:
            # DreamQuark applies shared transform to raw input first
            x_processed = self.shared(x, training=training)
            # Then uses shared-transformed input for initial splitter
            features = self.initial_splitter(x_processed, training=training)
        else:
            # For subsequent steps, DreamQuark:
            # 1. Applies shared transform to masked input
            x_processed = self.shared(x, training=training)
            # 2. Then applies feature transformer
            features = self.feat_transformers[step-1](x_processed, training=training)
        
        # Split features
        d, a = tf.split(features, [self.n_d, self.n_a], axis=-1)
        
        # Compute mask
        if step == 0:
            mask = sparsemax(a)
        else:
            mask = self.att_transformers[step-1](a, prior, training=training)
        
        # Apply mask to raw input (not processed input)
        masked_x = x * tf.expand_dims(mask, axis=-1)
        
        return d, mask, masked_x

    def call(self, x, training=None):
        x = self.initial_bn(x, training=training)
        
        prior = None
        steps_output = []
        masks = []
        
        # Process steps
        for step_i in range(self.n_steps):
            step_out, mask, masked_x = self.forward_step(x, step_i, prior, training)
            steps_output.append(step_out)
            masks.append(mask)
            
            # Update prior (DreamQuark's way)
            if step_i == 0:
                prior = mask
            else:
                prior = self.gamma * mask
            
            # Update x for next step
            x = masked_x
            
        return tf.concat(steps_output, axis=-1), tf.stack(masks, axis=1)

class TabNet(tf.keras.Model):
    def __init__(
        self,
        feature_columns=None,  # Now optional
        output_dim=1,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        virtual_batch_size=128,
        momentum=0.02
    ):
        """TabNet model supporting dictionary inputs with dynamic feature inference.
        
        Args:
            feature_columns: Optional[Dict[str, int]], mapping feature names to dimensions.
                           If None, will be inferred from first input.
            output_dim: Output dimension
            ...
        """
        super().__init__()
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        
        # Will be set on first call if not provided
        self.feature_columns = feature_columns
        self.feature_names = None
        self.input_dim = None
        self.feature_groups = None
        self.encoder = None
        self.final = None
        
    def build(self, input_shape):
        """Build model on first call with input shape information."""
        if self.feature_columns is None:
            # Infer feature columns from input shape
            if isinstance(input_shape, dict):
                self.feature_columns = {
                    name: shape[-1] for name, shape in input_shape.items()
                }
            else:
                raise ValueError("First input must be dictionary when feature_columns not provided")
        
        if not self.built:
            self.feature_names = list(self.feature_columns.keys())
            self.input_dim = sum(self.feature_columns.values())
            self.feature_groups = self._create_feature_groups()
            
            # Create encoder
            self.encoder = TabNetEncoder(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                n_independent=self.n_independent,
                n_shared=self.n_shared,
                virtual_batch_size=self.virtual_batch_size,
                momentum=self.momentum,
                feature_groups=self.feature_groups
            )
            
            # Final prediction layer
            self.final = tf.keras.layers.Dense(self.output_dim)
            
        super().build(input_shape)
    
    def _create_feature_groups(self):
        """Create feature groups based on input dictionary."""
        feature_groups = {}
        start_idx = 0
        
        for feature_name, feature_dim in self.feature_columns.items():
            # Create indices for this feature group
            feature_indices = list(range(start_idx, start_idx + feature_dim))
            feature_groups[feature_name] = feature_indices
            start_idx += feature_dim
            
        return feature_groups
    
    def _preprocess_input(self, inputs):
        """Convert dictionary input to tensor."""
        # Concatenate features in order
        features = [inputs[name] for name in self.feature_names]
        return tf.concat(features, axis=-1)
        
    def call(self, inputs, training=None):
        # Convert dictionary input to tensor
        if isinstance(inputs, dict):
            x = self._preprocess_input(inputs)
        else:
            x = inputs
            
        features, masks = self.encoder(x, training=training)
        out = self.final(features)
        return out, masks

    def forward_masks(self, inputs):
        """Get feature masks for interpretability."""
        if isinstance(inputs, dict):
            x = self._preprocess_input(inputs)
        else:
            x = inputs
        _, masks = self.encoder(x, training=False)
        return masks
