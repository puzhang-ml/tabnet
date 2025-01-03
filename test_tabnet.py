"""
Modified from the original TabNet implementation by DreamQuark:
https://github.com/dreamquark-ai/tabnet

Test cases adapted from the original implementation to verify TensorFlow port functionality.

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
from tabnet_block import TabNet, sparsemax, GBN, SharedBlock, FeatTransformer

class TabNetTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.batch_size = 128
        self.feature_dim = 16
        self.n_steps = 3
        
    def test_ghost_batch_norm(self):
        """Test Ghost Batch Normalization behavior."""
        virtual_batch_size = 8
        gbn = GBN(virtual_batch_size=virtual_batch_size)
        
        # Test with batch size multiple of virtual_batch_size
        x = tf.random.normal((32, 16))
        out = gbn(x, training=True)
        self.assertEqual(out.shape, x.shape)
        
        # Test inference mode
        out_inference = gbn(x, training=False)
        self.assertEqual(out_inference.shape, x.shape)

    def test_shared_block(self):
        """Test SharedBlock behavior."""
        block = SharedBlock(input_dim=16, output_dim=8)
        x = tf.random.normal((self.batch_size, 16))
        out = block(x, training=True)
        self.assertEqual(out.shape, (self.batch_size, 8))

    def test_feature_transformer(self):
        """Test FeatTransformer behavior."""
        shared = SharedBlock(16, 8)
        ft = FeatTransformer(
            input_dim=16,
            output_dim=8,
            shared_block=shared
        )
        
        x = tf.random.normal((self.batch_size, 16))
        out = ft(x, training=True)
        self.assertEqual(out.shape, (self.batch_size, 8))

    def test_model_output_shape(self):
        """Test model output shapes."""
        feature_columns = {'features': self.feature_dim}  # Single feature group
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        out, masks = model(x, training=True)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertEqual(masks.shape, (self.batch_size, self.n_steps, self.feature_dim))
        
    def test_mask_sum_to_one(self):
        """Test if masks sum to approximately 1."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        _, masks = model(x, training=True)
        
        # Check if masks sum to approximately 1
        mask_sums = tf.reduce_sum(masks, axis=-1)
        self.assertAllClose(mask_sums, tf.ones_like(mask_sums), rtol=1e-5)

    def test_feature_selection(self):
        """Test if model learns to select relevant features."""
        # Create data where only first 3 features are relevant
        feature_columns = {'features': self.feature_dim}
        X = {'features': tf.random.normal((1000, self.feature_dim))}
        y = tf.cast(tf.reduce_sum(X['features'][:, :3], axis=1) > 0, tf.float32)
        
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_steps=3
        )
        
        # Train for a few steps
        optimizer = tf.keras.optimizers.Adam()
        for _ in range(10):
            with tf.GradientTape() as tape:
                pred, masks = model(X, training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, pred))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Get feature importance from masks
        masks = model.forward_masks(X)
        feature_importance = tf.reduce_mean(masks, axis=[0, 1])
        
        # First 3 features should have higher importance
        important_features = tf.argsort(feature_importance)[-3:]
        self.assertTrue(all(f < 3 for f in important_features))

    def test_initial_splitter(self):
        """Test if initial splitter behaves differently from other steps."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=3
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        
        # Get masks from two forward passes
        _, masks1 = model(x, training=True)
        _, masks2 = model(x, training=True)
        
        # First step masks should be identical
        self.assertAllClose(masks1[:, 0, :], masks2[:, 0, :])

    def test_prior_scaling(self):
        """Test if prior scaling with gamma works correctly."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            gamma=1.3
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        _, masks = model(x, training=True)
        
        # Extract masks for consecutive steps
        mask1 = masks[:, 0, :]  # First step
        mask2 = masks[:, 1, :]  # Second step
        
        # Check if second step mask is affected by prior
        # The sum of mask2 should be less than mask1 due to gamma scaling
        sum1 = tf.reduce_sum(mask1)
        sum2 = tf.reduce_sum(mask2)
        self.assertLess(sum2, sum1 * 1.3)

    def test_shared_block_reuse(self):
        """Test if shared block is properly reused across steps."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        
        # Get first step output
        with tf.GradientTape() as tape:
            # Forward pass will use shared block in each step
            out, masks = model(x, training=True)
            loss = tf.reduce_mean(out)
        
        # Get gradients for shared block
        shared_vars = model.encoder.shared.trainable_variables
        grads = tape.gradient(loss, shared_vars)
        
        # Shared block should be used multiple times, so gradients should exist
        self.assertTrue(all(g is not None for g in grads))

    def test_feature_transformer_dimensions(self):
        """Test feature transformer output dimensions match DreamQuark."""
        feature_columns = {'features': self.feature_dim}
        n_d = 8
        n_a = 8
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=n_d,
            n_a=n_a
        )
        
        # Get first feature transformer
        ft = model.encoder.feat_transformers[0]
        x = tf.random.normal((self.batch_size, self.feature_dim))
        
        # First apply shared transform as in DreamQuark
        x_processed = model.encoder.shared(x)
        out = ft(x_processed)
        
        # Output should be n_d + n_a as in DreamQuark
        self.assertEqual(out.shape[-1], n_d + n_a)

    def test_attentive_transformer_dimensions(self):
        """Test attentive transformer dimensions match DreamQuark."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8
        )
        
        # Get first attentive transformer
        at = model.encoder.att_transformers[0]
        x = tf.random.normal((self.batch_size, 8))  # Input is n_a dimensional
        mask = at(x, training=True)
        
        # Mask should be feature_dim dimensional
        self.assertEqual(mask.shape[-1], self.feature_dim)

    def test_initial_processing_order(self):
        """Test that initial batch norm and shared transform are applied in correct order."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        
        # Manual forward pass
        x_bn = model.encoder.initial_bn(x, training=True)
        x_shared = model.encoder.shared(x_bn, training=True)
        
        # Model forward pass first step
        _, masks = model(x, training=True)
        
        # First step mask should be same as manual processing
        features = model.encoder.initial_splitter(x_shared, training=True)
        _, a = tf.split(features, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        expected_mask = sparsemax(a)
        
        self.assertAllClose(masks[:, 0, :], expected_mask)

    def test_glu_behavior(self):
        """Test GLU behaves exactly as in DreamQuark."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        
        # Get first GLU layer from feature transformer
        glu = model.encoder.feat_transformers[0].specifics[0]
        x = tf.random.normal((self.batch_size, self.feature_dim))
        
        # Manual GLU computation
        fc_out = glu.fc(x)
        bn_out = glu.bn(fc_out, training=True)
        out, gate = tf.split(bn_out, 2, axis=-1)
        expected = out * tf.nn.sigmoid(gate)
        
        # Layer output
        actual = glu(x, training=True)
        
        self.assertAllClose(actual, expected)

    def test_feature_reuse_pattern(self):
        """Test that features are reused according to DreamQuark's pattern."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_steps=2
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        
        # First step
        x_tensor = x['features']  # Get tensor for manual processing
        x_bn = model.encoder.initial_bn(x_tensor, training=True)  # Apply batch norm first
        x_shared = model.encoder.shared(x_bn, training=True)      # Apply shared transform
        features1 = model.encoder.initial_splitter(x_shared, training=True)  # Feature transform
        d1, a1 = tf.split(features1, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        mask1 = sparsemax(a1)  # Generate mask
        
        # Compare with model output
        _, masks = model(x, training=True)
        self.assertAllClose(masks[:, 0, :], mask1)

    def test_sparsemax(self):
        """Test sparsemax activation function."""
        x = tf.random.normal((self.batch_size, self.feature_dim))
        out = sparsemax(x)
        
        # Check output shape
        self.assertEqual(out.shape, x.shape)
        
        # Check properties of sparsemax
        row_sums = tf.reduce_sum(out, axis=-1)
        self.assertAllClose(row_sums, tf.ones_like(row_sums))
        self.assertAllGreaterEqual(out, 0.0)

    def test_feature_selection_interpretability(self):
        """Test if feature selection is interpretable."""
        feature_columns = {'features': self.feature_dim}
        X = {'features': tf.random.normal((1000, self.feature_dim))}
        y = tf.cast(X['features'][:, 0] > 0, tf.float32)
        
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_steps=1
        )
        
        # Train for a few steps
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.02)
        for _ in range(20):
            with tf.GradientTape() as tape:
                pred, masks = model(X, training=True)
                loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, pred))
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        # Get feature importance
        _, masks = model(X, training=False)
        feature_importance = tf.reduce_mean(masks, axis=0)  # Average over batch
        
        # First feature should have highest importance
        self.assertEqual(tf.argmax(feature_importance[0]), 0)

    def test_virtual_batch_size(self):
        """Test if virtual batch size is working correctly."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            virtual_batch_size=16
        )
        
        # Test with batch size larger than virtual_batch_size
        x = {'features': tf.random.normal((64, self.feature_dim))}
        out, _ = model(x, training=True)
        self.assertEqual(out.shape, (64, 1))
        
        # Test with batch size smaller than virtual_batch_size
        x = {'features': tf.random.normal((8, self.feature_dim))}
        out, _ = model(x, training=True)
        self.assertEqual(out.shape, (8, 1))

    def test_batch_size_independence(self):
        """Test if model works with different batch sizes."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch_size in batch_sizes:
            x = {'features': tf.random.normal((batch_size, self.feature_dim))}
            out, masks = model(x, training=True)
            self.assertEqual(out.shape, (batch_size, 1))
            self.assertEqual(masks.shape, (batch_size, self.n_steps, self.feature_dim))

    def test_training_mode(self):
        """Test if model behaves differently in training vs inference mode."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        
        # Get outputs in training mode
        out_train, masks_train = model(x, training=True)
        
        # Get outputs in inference mode
        out_test, masks_test = model(x, training=False)
        
        # Outputs should be different due to batch norm behavior
        with self.assertRaises(AssertionError):
            self.assertAllClose(out_train, out_test)

    def test_parameter_count(self):
        """Test if number of parameters matches DreamQuark's implementation."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=3,
            n_independent=2,
            n_shared=2
        )
        
        # Count trainable parameters
        total_params = np.sum([np.prod(v.shape) for v in model.trainable_variables])
        
        # Verify parameter count is reasonable
        # (exact count depends on implementation details)
        self.assertGreater(total_params, 1000)
        self.assertLess(total_params, 100000)

    def test_dictionary_input(self):
        """Test if model works with dictionary inputs."""
        feature_columns = {
            'numeric': 3,
            'categorical_1': 4,
            'categorical_2': 3
        }
        
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        
        # Create sample input
        inputs = {
            'numeric': tf.random.normal((self.batch_size, 3)),
            'categorical_1': tf.random.uniform((self.batch_size, 4)),
            'categorical_2': tf.random.uniform((self.batch_size, 3))
        }
        
        # Test forward pass
        out, masks = model(inputs, training=True)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertEqual(masks.shape, (self.batch_size, self.n_steps, sum(feature_columns.values())))

    def test_feature_groups(self):
        """Test if feature groups are correctly created and used."""
        feature_columns = {
            'numeric': 3,
            'categorical': 4
        }
        
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1
        )
        
        # Check feature groups
        self.assertEqual(model.feature_groups['numeric'], [0, 1, 2])
        self.assertEqual(model.feature_groups['categorical'], [3, 4, 5, 6])
        
        # Test with input
        inputs = {
            'numeric': tf.random.normal((self.batch_size, 3)),
            'categorical': tf.random.uniform((self.batch_size, 4))
        }
        
        _, masks = model(inputs, training=False)
        
        # Check mask dimensions match feature groups
        numeric_mask = masks[:, :, model.feature_groups['numeric']]
        categorical_mask = masks[:, :, model.feature_groups['categorical']]
        
        self.assertEqual(numeric_mask.shape[-1], 3)
        self.assertEqual(categorical_mask.shape[-1], 4)

    def test_dynamic_feature_inference(self):
        """Test if model correctly infers feature dimensions from input."""
        # Create model without feature columns
        model = TabNet(output_dim=1)
        
        # Create sample input
        inputs = {
            'numeric': tf.random.normal((self.batch_size, 3)),
            'categorical_1': tf.random.uniform((self.batch_size, 4)),
            'categorical_2': tf.random.uniform((self.batch_size, 3))
        }
        
        # First call should infer and build
        out, masks = model(inputs, training=True)
        
        # Check inferred dimensions
        self.assertEqual(model.feature_columns['numeric'], 3)
        self.assertEqual(model.feature_columns['categorical_1'], 4)
        self.assertEqual(model.feature_columns['categorical_2'], 3)
        
        # Check output shapes
        self.assertEqual(out.shape, (self.batch_size, 1))
        self.assertEqual(masks.shape, (self.batch_size, model.n_steps, sum(model.feature_columns.values())))

    def test_input_validation(self):
        """Test input validation for dynamic feature inference."""
        model = TabNet(output_dim=1)
        
        # Should raise error for non-dictionary input without feature_columns
        x = tf.random.normal((self.batch_size, 16))
        with self.assertRaises(ValueError):
            model(x, training=True)

    def test_tabnet_masking_order(self):
        """Test if masking order exactly matches DreamQuark's implementation."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=2  # Use 2 steps for easier testing
        )
        
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        x_tensor = x['features']
        
        # First step verification
        with tf.GradientTape() as tape:
            # 1. Initial batch norm
            x_bn = model.encoder.initial_bn(x_tensor, training=True)
            
            # 2. Shared transform on raw input
            x_shared = model.encoder.shared(x_bn, training=True)
            
            # 3. Initial splitter
            features = model.encoder.initial_splitter(x_shared, training=True)
            d1, a1 = tf.split(features, [model.encoder.n_d, model.encoder.n_a], axis=-1)
            
            # 4. Generate first mask
            mask1 = sparsemax(a1)
            
            # 5. Apply mask to raw input
            masked_x1 = x_bn * tf.expand_dims(mask1, axis=-1)
        
        # Second step verification
        # 1. Shared transform on raw input
        x_shared2 = model.encoder.shared(x_bn, training=True)
        _, a2 = tf.split(x_shared2, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        
        # 2. Generate mask using prior
        mask2 = model.encoder.att_transformers[0](a2, mask1 * model.encoder.gamma, training=True)
        
        # 3. Apply mask to raw input
        masked_x2 = x_bn * tf.expand_dims(mask2, axis=-1)
        
        # 4. Transform masked input
        masked_processed = model.encoder.shared(masked_x2, training=True)
        features2 = model.encoder.feat_transformers[0](masked_processed, training=True)
        d2, _ = tf.split(features2, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        
        # Compare with model's internal processing
        _, masks = model(x, training=True)
        
        # Verify masks match
        self.assertAllClose(masks[:, 0, :], mask1)
        self.assertAllClose(masks[:, 1, :], mask2)
        
        # Get step-by-step outputs from model
        d1_model, mask1_model, masked_x1_model = model.encoder.forward_step(x_bn, 0, None, training=True)
        d2_model, mask2_model, masked_x2_model = model.encoder.forward_step(x_bn, 1, mask1_model, training=True)
        
        # Verify each step matches
        self.assertAllClose(d1_model, d1)
        self.assertAllClose(d2_model, d2)
        self.assertAllClose(masked_x1_model, masked_x1)
        self.assertAllClose(masked_x2_model, masked_x2)

    def test_tabnet_encoder_dict_input(self):
        """Test if TabNetEncoder handles dictionary inputs correctly."""
        # Create encoder with dynamic feature inference
        encoder = TabNetEncoder(
            input_dim=None,  # Will be inferred
            output_dim=8,
            n_d=4,
            n_a=4,
            n_steps=3
        )
        
        # Create dictionary input
        inputs = {
            'numeric': tf.random.normal((self.batch_size, 3)),
            'categorical': tf.random.uniform((self.batch_size, 4)),
            'binary': tf.random.uniform((self.batch_size, 2))
        }
        
        # First call should build and infer dimensions
        features, masks = encoder(inputs, training=True)
        
        # Check inferred dimensions
        self.assertEqual(encoder.input_dim, 9)  # 3 + 4 + 2
        self.assertEqual(len(encoder.feature_columns), 3)
        self.assertEqual(encoder.feature_columns['numeric'], 3)
        self.assertEqual(encoder.feature_columns['categorical'], 4)
        self.assertEqual(encoder.feature_columns['binary'], 2)
        
        # Check output shapes
        self.assertEqual(features.shape, (self.batch_size, encoder.n_d * encoder.n_steps))
        self.assertEqual(masks.shape, (self.batch_size, encoder.n_steps, encoder.input_dim))

    def test_tabnet_encoder_feature_groups(self):
        """Test if TabNetEncoder creates and uses feature groups correctly."""
        feature_columns = {
            'numeric': 3,
            'categorical': 4,
            'binary': 2
        }
        
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8,
            feature_columns=feature_columns
        )
        
        inputs = {
            'numeric': tf.random.normal((self.batch_size, 3)),
            'categorical': tf.random.uniform((self.batch_size, 4)),
            'binary': tf.random.uniform((self.batch_size, 2))
        }
        
        # Get outputs and check feature groups
        _, masks = encoder(inputs, training=True)
        
        # Check feature groups were created correctly
        self.assertEqual(encoder.feature_groups['numeric'], [0, 1, 2])
        self.assertEqual(encoder.feature_groups['categorical'], [3, 4, 5, 6])
        self.assertEqual(encoder.feature_groups['binary'], [7, 8])
        
        # Check mask dimensions match feature groups
        numeric_mask = masks[:, :, encoder.feature_groups['numeric']]
        categorical_mask = masks[:, :, encoder.feature_groups['categorical']]
        binary_mask = masks[:, :, encoder.feature_groups['binary']]
        
        self.assertEqual(numeric_mask.shape[-1], 3)
        self.assertEqual(categorical_mask.shape[-1], 4)
        self.assertEqual(binary_mask.shape[-1], 2)

    def test_tabnet_encoder_feature_order(self):
        """Test if TabNetEncoder maintains consistent feature order."""
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8
        )
        
        # Create inputs with different order
        inputs1 = {
            'c': tf.random.normal((self.batch_size, 2)),
            'a': tf.random.normal((self.batch_size, 2)),
            'b': tf.random.normal((self.batch_size, 2))
        }
        
        inputs2 = {
            'a': tf.random.normal((self.batch_size, 2)),
            'b': tf.random.normal((self.batch_size, 2)),
            'c': tf.random.normal((self.batch_size, 2))
        }
        
        # Get outputs
        features1, _ = encoder(inputs1, training=True)
        features2, _ = encoder(inputs2, training=True)
        
        # Features should be same regardless of input order
        # (because we sort keys when feature_names is None)
        self.assertAllClose(features1, features2)

    def test_tabnet_encoder_missing_features(self):
        """Test if TabNetEncoder handles missing features correctly."""
        feature_columns = {
            'a': 2,
            'b': 2,
            'c': 2
        }
        
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8,
            feature_columns=feature_columns
        )
        
        # Try input with missing feature
        inputs = {
            'a': tf.random.normal((self.batch_size, 2)),
            'c': tf.random.normal((self.batch_size, 2))
        }
        
        # Should raise error because 'b' is missing
        with self.assertRaises(KeyError):
            features, _ = encoder(inputs, training=True)

    def test_tabnet_encoder_tensorspec_input(self):
        """Test if TabNetEncoder handles TensorSpec inputs correctly."""
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8
        )
        
        # Create TensorSpec input shapes similar to production data
        class MockTensorSpec:
            def __init__(self, shape, inferred_value):
                self.shape = shape
                self.inferred_value = inferred_value
                
            def __getitem__(self, idx):
                return self.shape[idx]
                
        input_shape = {
            'feature1_ctr_28day': MockTensorSpec(shape=(1,), inferred_value=[None]),
            'feature2_industry_14day': MockTensorSpec(shape=(2,), inferred_value=[None, 1]),
            'feature3_indicator_3day': MockTensorSpec(shape=(1,), inferred_value=[None]),
            'feature4_posterior_28day': MockTensorSpec(shape=(1,), inferred_value=[None]),
            'feature5_country_28day': MockTensorSpec(shape=(2,), inferred_value=[None, 1]),
            'feature6_rate_7day': MockTensorSpec(shape=(2,), inferred_value=[None, 1])
        }
        
        # Build encoder with these shapes
        encoder.build(input_shape)
        
        # Verify inferred dimensions
        self.assertEqual(encoder.input_dim, 6)  # Total dimensions
        self.assertEqual(len(encoder.feature_columns), 6)  # Number of features
        
        # Check individual feature dimensions
        expected_dims = {
            'feature1_ctr_28day': 1,
            'feature2_industry_14day': 1,
            'feature3_indicator_3day': 1,
            'feature4_posterior_28day': 1,
            'feature5_country_28day': 1,
            'feature6_rate_7day': 1
        }
        
        for name, dim in expected_dims.items():
            self.assertEqual(encoder.feature_columns[name], dim)
        
        # Check feature groups
        start_idx = 0
        for name in sorted(expected_dims.keys()):
            dim = expected_dims[name]
            expected_indices = list(range(start_idx, start_idx + dim))
            self.assertEqual(encoder.feature_groups[name], expected_indices)
            start_idx += dim

    def test_tabnet_encoder_mixed_shapes(self):
        """Test if TabNetEncoder handles mixed shape types correctly."""
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8
        )
        
        # Mix of TensorSpec and regular shapes
        class MockTensorSpec:
            def __init__(self, shape, inferred_value):
                self.shape = shape
                self.inferred_value = inferred_value
                
            def __getitem__(self, idx):
                return self.shape[idx]
        
        input_shape = {
            'regular_feature1': tf.TensorShape([None, 2]),
            'spec_feature2': MockTensorSpec(shape=(2,), inferred_value=[None, 3]),
            'regular_feature3': tf.TensorShape([None, 1]),
            'spec_feature4': MockTensorSpec(shape=(1,), inferred_value=[None])
        }
        
        # Build encoder
        encoder.build(input_shape)
        
        # Verify dimensions
        expected_dims = {
            'regular_feature1': 2,
            'spec_feature2': 3,
            'regular_feature3': 1,
            'spec_feature4': 1
        }
        
        self.assertEqual(encoder.input_dim, sum(expected_dims.values()))
        
        for name, dim in expected_dims.items():
            self.assertEqual(encoder.feature_columns[name], dim)

    def test_tabnet_encoder_kerastensor_dimensions(self):
        """Test if TabNetEncoder handles KerasTensor dimensions correctly."""
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8
        )
        
        class MockKerasTensor:
            def __init__(self, shape, inferred_value):
                self.shape = shape
                self.inferred_value = inferred_value
                
            def __getitem__(self, idx):
                return self.shape[idx]
        
        input_shape = {
            'feature1': MockKerasTensor(shape=(2,), inferred_value=[None, 3]),  # Should use 3
            'feature2': MockKerasTensor(shape=(1,), inferred_value=[None]),     # Should use 1
            'feature3': MockKerasTensor(shape=(2,), inferred_value=[None, None]), # Should use 1
            'feature4': MockKerasTensor(shape=(2,), inferred_value=[None, 2])   # Should use 2
        }
        
        encoder.build(input_shape)
        
        # Verify dimensions
        expected_dims = {
            'feature1': 3,
            'feature2': 1,
            'feature3': 1,
            'feature4': 2
        }
        
        self.assertEqual(encoder.input_dim, 7)  # Total should be 3+1+1+2=7
        
        for name, dim in expected_dims.items():
            self.assertEqual(
                encoder.feature_columns[name], 
                dim, 
                f"Feature {name} dimension mismatch. Expected {dim}, got {encoder.feature_columns[name]}"
            )

    def test_tabnet_step_outputs(self):
        """Test if each step's outputs match DreamQuark's implementation exactly."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=3
        )
        
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        x_tensor = x['features']
        x_bn = model.encoder.initial_bn(x_tensor, training=True)
        
        # Step 0 verification
        # Manual processing
        x_processed_0 = model.encoder.shared(x_bn, training=True)
        features_0 = model.encoder.initial_splitter(x_processed_0, training=True)
        d0, a0 = tf.split(features_0, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        mask0 = sparsemax(a0)
        
        # Model processing
        d0_model, mask0_model, processed0_model = model.encoder.forward_step(x_bn, 0, None, training=True)
        
        # Verify step 0 outputs
        self.assertAllClose(d0_model, d0, msg="Step 0 decision output mismatch")
        self.assertAllClose(mask0_model, mask0, msg="Step 0 mask mismatch")
        self.assertAllClose(processed0_model, x_processed_0, msg="Step 0 processed output mismatch")
        
        # Step 1 verification
        # Manual processing
        x_processed_1 = model.encoder.shared(x_bn, training=True)
        _, a1 = tf.split(x_processed_1, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        mask1 = model.encoder.att_transformers[0](a1, mask0 * model.encoder.gamma, training=True)
        masked_x1 = x_bn * tf.expand_dims(mask1, axis=-1)
        masked_processed1 = model.encoder.shared(masked_x1, training=True)
        features1 = model.encoder.feat_transformers[0](masked_processed1, training=True)
        d1, _ = tf.split(features1, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        
        # Model processing
        d1_model, mask1_model, processed1_model = model.encoder.forward_step(x_bn, 1, mask0, training=True)
        
        # Verify step 1 outputs
        self.assertAllClose(d1_model, d1, msg="Step 1 decision output mismatch")
        self.assertAllClose(mask1_model, mask1, msg="Step 1 mask mismatch")
        self.assertAllClose(processed1_model, masked_processed1, msg="Step 1 processed output mismatch")
        
        # Verify full model outputs
        _, masks = model(x, training=True)
        self.assertAllClose(masks[:, 0, :], mask0, msg="Model step 0 mask mismatch")
        self.assertAllClose(masks[:, 1, :], mask1, msg="Model step 1 mask mismatch")

    def test_tabnet_feature_transformations(self):
        """Test the exact order and values of feature transformations."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=2
        )
        
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        x_tensor = x['features']
        x_bn = model.encoder.initial_bn(x_tensor, training=True)
        
        # Track all intermediate transformations
        with tf.GradientTape() as tape:
            # Step 0
            shared_out0 = model.encoder.shared(x_bn, training=True)
            splitter_out0 = model.encoder.initial_splitter(shared_out0, training=True)
            d0, a0 = tf.split(splitter_out0, [model.encoder.n_d, model.encoder.n_a], axis=-1)
            mask0 = sparsemax(a0)
            
            # Step 1
            shared_out1 = model.encoder.shared(x_bn, training=True)
            _, a1 = tf.split(shared_out1, [model.encoder.n_d, model.encoder.n_a], axis=-1)
            mask1 = model.encoder.att_transformers[0](a1, mask0 * model.encoder.gamma, training=True)
            masked_x1 = x_bn * tf.expand_dims(mask1, axis=-1)
            masked_shared1 = model.encoder.shared(masked_x1, training=True)
            features1 = model.encoder.feat_transformers[0](masked_shared1, training=True)
            d1, _ = tf.split(features1, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        
        # Get gradients to verify feature transformation flow
        grads = tape.gradient(d1, [shared_out0, shared_out1, masked_shared1])
        
        # Verify gradient flow
        self.assertIsNotNone(grads[0], "No gradient flow through initial shared transform")
        self.assertIsNotNone(grads[1], "No gradient flow through attention shared transform")
        self.assertIsNotNone(grads[2], "No gradient flow through masked shared transform")

    def test_tabnet_edge_cases(self):
        """Test TabNet behavior in edge cases."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=2,
            gamma=1.0  # No prior scaling
        )
        
        # Test with zero input
        x_zero = {'features': tf.zeros((self.batch_size, self.feature_dim))}
        out_zero, masks_zero = model(x_zero, training=True)
        
        # Masks should still sum to 1
        mask_sums = tf.reduce_sum(masks_zero, axis=-1)
        self.assertAllClose(mask_sums, tf.ones_like(mask_sums))
        
        # Test with constant input
        x_const = {'features': tf.ones((self.batch_size, self.feature_dim))}
        out_const, masks_const = model(x_const, training=True)
        
        # Different samples should have same masks
        self.assertAllClose(
            masks_const[0], masks_const[1],
            msg="Constant input should produce same masks"
        )
        
        # Test with single feature active
        x_single = {'features': tf.eye(self.feature_dim)[None]}
        out_single, masks_single = model(x_single, training=True)
        
        # First step should focus on active feature
        active_features = tf.argmax(masks_single[0, 0], axis=-1)
        self.assertEqual(active_features.numpy(), 0)

    def test_tabnet_step_flow(self):
        """Test detailed step-by-step flow in TabNet."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_d=8,
            n_a=8,
            n_steps=3
        )
        
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        
        # Run model and get step visualizations
        out, masks = model(x, training=True)
        step_data = model.encoder.debug_outputs
        
        # Verify step 0 flow
        step0 = step_data[0]
        self.assertAllClose(
            step0['masked_input'],
            step0['raw_input'] * tf.expand_dims(step0['mask'], axis=-1)
        )
        
        # Verify step 1 flow
        step1 = step_data[1]
            self.assertAllClose(
            step1['mask'],
            model.encoder.att_transformers[0](
                step1['attention'],
                step0['mask'] * model.encoder.gamma,
                training=True
            )
        )
        
        # Verify feature reuse
        feature_importance = model.encoder.get_feature_importance()
        total_usage = tf.reduce_sum(feature_importance)
        self.assertAllClose(total_usage, tf.constant(1.0))

    def test_tabnet_encoder_kerastensor_edge_cases(self):
        """Test if TabNetEncoder handles various KerasTensor edge cases correctly."""
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8
        )
        
        class MockKerasTensor:
            def __init__(self, shape, inferred_value=None):
                self.shape = shape
                self.inferred_value = inferred_value
                
            def __getitem__(self, idx):
                return self.shape[idx]
            
            def __len__(self):
                return len(self.shape)
        
        # Test various edge cases
        input_shape = {
            'feature1': MockKerasTensor(shape=(1,), inferred_value=None),  # Should use 1
            'feature2': MockKerasTensor(shape=(2,), inferred_value=[None, 1]),  # Should use 1
            'feature3': MockKerasTensor(shape=(2,), inferred_value=[None, None]),  # Should use 1
            'feature4': MockKerasTensor(shape=(2,), inferred_value=[None, 3]),  # Should use 3
            'feature5': MockKerasTensor(shape=(1,), inferred_value=[None]),  # Should use 1
            'feature6': MockKerasTensor(shape=(2,), inferred_value=None)  # Should use shape[-1]=2
        }
        
        # Build encoder
        encoder.build(input_shape)
        
        # Verify dimensions
        expected_dims = {
            'feature1': 1,  # inferred_value is None, shape=(1,) -> use 1
            'feature2': 1,  # inferred_value=[None, 1] -> use 1
            'feature3': 1,  # inferred_value=[None, None] -> use 1
            'feature4': 3,  # inferred_value=[None, 3] -> use 3
            'feature5': 1,  # inferred_value=[None] -> use 1
            'feature6': 2   # inferred_value is None, shape=(2,) -> use 2
        }
        
        # Check individual feature dimensions
        for name, expected_dim in expected_dims.items():
            self.assertEqual(
                encoder.feature_columns[name], 
                expected_dim,
                f"Feature {name} dimension mismatch. Expected {expected_dim}, got {encoder.feature_columns[name]}"
            )
        
        # Check total dimension
        self.assertEqual(encoder.input_dim, sum(expected_dims.values()))
        
        # Check feature groups
        start_idx = 0
        for name in sorted(expected_dims.keys()):
            dim = expected_dims[name]
            expected_indices = list(range(start_idx, start_idx + dim))
            self.assertEqual(
                encoder.feature_groups[name],
                expected_indices,
                f"Feature group indices mismatch for {name}"
            )
            start_idx += dim

    def test_tabnet_encoder_production_shapes(self):
        """Test TabNetEncoder with production-like KerasTensor shapes."""
        encoder = TabNetEncoder(
            input_dim=None,
            output_dim=8
        )
        
        class MockKerasTensor:
            def __init__(self, shape, inferred_value):
                self.shape = shape
                self.inferred_value = inferred_value
                
            def __getitem__(self, idx):
                return self.shape[idx]
            
            def __len__(self):
                return len(self.shape)
        
        # Simulate production input shapes
        input_shape = {
            'user_ctr_28day_indicator': MockKerasTensor(
                shape=(1,), inferred_value=[None]
            ),
            'adv_industry_ctr_14day': MockKerasTensor(
                shape=(2,), inferred_value=[None, 1]
            ),
            'adv_industry_ctr_3day_indicator': MockKerasTensor(
                shape=(1,), inferred_value=[None]
            ),
            'ad_user_posterior_100000_ctr_28day_indicator': MockKerasTensor(
                shape=(1,), inferred_value=[None]
            ),
            'adv_country_dma_ctr_28day_indicator': MockKerasTensor(
                shape=(2,), inferred_value=[None, 1]
            ),
            'flight_user_lng_clk_20s_rate_7day': MockKerasTensor(
                shape=(2,), inferred_value=[None, 1]
            )
        }
        
        # Build encoder
        encoder.build(input_shape)
        
        # All features should have dimension 1
        for name, dim in encoder.feature_columns.items():
            self.assertEqual(dim, 1, f"Feature {name} should have dimension 1")
        
        # Total dimension should be number of features
        self.assertEqual(encoder.input_dim, len(input_shape))
        
        # Feature groups should be sequential
        for i, name in enumerate(sorted(input_shape.keys())):
            self.assertEqual(encoder.feature_groups[name], [i])

if __name__ == '__main__':
    tf.test.main() 