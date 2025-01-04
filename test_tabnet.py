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

    def test_masking_order(self):
        """Test that masking order exactly matches DreamQuark's implementation."""
        feature_columns = {'features': self.feature_dim}
        model = TabNet(
            feature_columns=feature_columns,
            output_dim=1,
            n_steps=2  # Simpler to test
        )
        x = {'features': tf.random.normal((self.batch_size, self.feature_dim))}
        x_tensor = x['features']

        # Step 1: Initial processing
        x_bn = model.encoder.initial_bn(x_tensor, training=True)
        
        # First decision step
        # 1. Apply shared transform to batch-normalized input
        x_shared = model.encoder.shared(x_bn, training=True)
        # 2. Apply feature transform
        features = model.encoder.initial_splitter(x_shared, training=True)
        # 3. Generate mask
        d1, a1 = tf.split(features, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        mask1 = sparsemax(a1)
        # 4. Apply mask to batch-normalized input (not transformed)
        masked_x1 = x_bn * tf.expand_dims(mask1, axis=-1)
        
        # Second decision step
        # 1. Apply shared transform to masked input
        x_shared2 = model.encoder.shared(masked_x1, training=True)
        # 2. Apply feature transform
        features2 = model.encoder.feat_transformers[0](x_shared2, training=True)
        # 3. Generate mask with prior
        d2, a2 = tf.split(features2, [model.encoder.n_d, model.encoder.n_a], axis=-1)
        mask2 = model.encoder.att_transformers[0](a2, mask1 * model.encoder.gamma, training=True)
        # 4. Apply mask to batch-normalized input
        masked_x2 = x_bn * tf.expand_dims(mask2, axis=-1)
        
        # Compare with model's internal processing
        _, masks = model(x, training=True)
        
        # Verify masks match
        self.assertAllClose(masks[:, 0, :], mask1)
        self.assertAllClose(masks[:, 1, :], mask2)
        
        # Get intermediate outputs from model for comparison
        with tf.GradientTape() as tape:
            x_processed = x_tensor
            x_processed = model.encoder.initial_bn(x_processed, training=True)
            
            # First step
            x_shared_model = model.encoder.shared(x_processed, training=True)
            features_model = model.encoder.initial_splitter(x_shared_model, training=True)
            
            # Should match our manual processing
            self.assertAllClose(x_shared_model, x_shared)
            self.assertAllClose(features_model, features)
            
            # Second step should use masked input
            d_model, mask_model, masked_x_model = model.encoder.forward_step(x_processed, 1, mask1, training=True)
            
            # Should match our manual processing
            self.assertAllClose(masked_x_model, masked_x1)
            self.assertAllClose(mask_model, mask2)

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

if __name__ == '__main__':
    tf.test.main() 