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
from typing import List, Dict
from tabnet_block import TabNet, infer_feature_groups_from_dict

class TabNetTests(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # Common test configurations
        self.batch_size = 32
        self.feature_dim = 64
        self.output_dim = 1
        
    def create_test_features_list(self) -> List[tf.Tensor]:
        """Create test features as list of tensors"""
        return [
            tf.random.normal((self.batch_size, 10)),  # Feature 1: 10 dims
            tf.random.normal((self.batch_size, 20)),  # Feature 2: 20 dims
            tf.random.normal((self.batch_size, 30)),  # Feature 3: 30 dims
        ]
        
    def create_test_features_dict(self) -> Dict[str, tf.Tensor]:
        """Create test features as dictionary"""
        return {
            'numeric': tf.random.normal((self.batch_size, 10)),
            'embedding': tf.random.normal((self.batch_size, 20)),
            'categorical': tf.random.normal((self.batch_size, 30)),
        }
        
    def test_list_input_graph_mode(self):
        """Test TabNet with list input in Graph mode"""
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7
        )
        
        # Create tf.function for graph mode
        @tf.function
        def run_model(features):
            output, _, _ = model(features, training=True)
            return output
            
        features = self.create_test_features_list()
        output = run_model(features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_dict_input_graph_mode(self):
        """Test TabNet with dictionary input in Graph mode"""
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7
        )
        
        @tf.function
        def run_model(features):
            output, _, _ = model(features, training=True)
            return output
            
        features = self.create_test_features_dict()
        output = run_model(features)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_single_tensor_input(self):
        """Test TabNet with single tensor input"""
        input_dim = 60
        model = TabNet(
            feature_dim=input_dim,
            output_dim=self.output_dim
        )
        
        # Create single tensor input
        features = tf.random.normal((self.batch_size, input_dim))
        output, _, _ = model(features, training=True)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_large_feature_dims(self):
        """Test TabNet with large feature dimensions"""
        features = [
            tf.random.normal((self.batch_size, 100)),  # Large dim feature
            tf.random.normal((self.batch_size, 200)),  # Larger dim feature
            tf.random.normal((self.batch_size, 50))    # Normal dim feature
        ]
        
        total_dims = sum(f.shape[-1] for f in features)
        model = TabNet(
            feature_dim=total_dims,
            output_dim=self.output_dim
        )
        
        output, _, _ = model(features, training=True)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_mixed_feature_types(self):
        """Test TabNet with mixed feature types (scalar, embedding, regular)"""
        # Create feature dimensions
        scalar_dim = 1
        embedding1_dim = 16
        embedding2_dim = 32
        continuous1_dim = 10
        continuous2_dim = 20
        total_dims = scalar_dim * 2 + embedding1_dim + embedding2_dim + continuous1_dim + continuous2_dim
        
        # Define feature groups for embeddings
        current_idx = 2  # After two scalar features
        embedding1_start = current_idx
        embedding1_end = embedding1_start + embedding1_dim
        embedding2_start = embedding1_end
        embedding2_end = embedding2_start + embedding2_dim
        
        grouped_features = [
            list(range(embedding1_start, embedding1_end)),  # embedding1 group
            list(range(embedding2_start, embedding2_end))   # embedding2 group
        ]
        
        # Initialize model with feature groups
        model = TabNet(
            feature_dim=total_dims,
            output_dim=self.output_dim,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=grouped_features
        )
        
        # Create mixed input features
        features = {
            'scalar1': tf.random.normal((self.batch_size,)),  # 1D scalar feature
            'scalar2': tf.random.normal((self.batch_size,)),  # 1D scalar feature
            'embedding1': tf.random.normal((self.batch_size, embedding1_dim)),  # Pre-embedded feature
            'embedding2': tf.random.normal((self.batch_size, embedding2_dim)),  # Pre-embedded feature
            'continuous1': tf.random.normal((self.batch_size, continuous1_dim)),  # Regular 2D feature
            'continuous2': tf.random.normal((self.batch_size, continuous2_dim))   # Regular 2D feature
        }
        
        # Test in eager mode
        output, masks, _ = model(features, training=True)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Verify that embedding dimensions are masked together
        for mask in masks:
            # Check embedding1 mask values are identical within the group
            embedding1_mask = mask[:, embedding1_start:embedding1_end, :]
            first_mask = embedding1_mask[:, :1, :]
            repeated_mask = tf.repeat(first_mask, embedding1_dim, axis=1)
            self.assertAllClose(embedding1_mask, repeated_mask, rtol=1e-5)
            
            # Check embedding2 mask values are identical within the group
            embedding2_mask = mask[:, embedding2_start:embedding2_end, :]
            first_mask = embedding2_mask[:, :1, :]
            repeated_mask = tf.repeat(first_mask, embedding2_dim, axis=1)
            self.assertAllClose(embedding2_mask, repeated_mask, rtol=1e-5)
        
        # Test in graph mode
        @tf.function
        def run_model(features):
            return model(features, training=True)
            
        output, masks, _ = run_model(features)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Verify masks in graph mode
        for mask in masks:
            embedding1_mask = mask[:, embedding1_start:embedding1_end]
            self.assertTrue(tf.reduce_all(embedding1_mask == embedding1_mask[:, 0:1]))
            
            embedding2_mask = mask[:, embedding2_start:embedding2_end]
            self.assertTrue(tf.reduce_all(embedding2_mask == embedding2_mask[:, 0:1]))
        
    def test_mixed_feature_types_list(self):
        """Test TabNet with mixed feature types as list input"""
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7
        )
        
        # Create list of mixed features
        features = [
            tf.random.normal((self.batch_size,)),         # 1D scalar feature
            tf.random.normal((self.batch_size, 16)),      # Pre-embedded feature
            tf.random.normal((self.batch_size,)),         # Another 1D scalar
            tf.random.normal((self.batch_size, 32)),      # Another embedding
            tf.random.normal((self.batch_size, 10))       # Regular 2D feature
        ]
        
        # Test forward pass
        output, _, _ = model(features, training=True)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Test with @tf.function
        @tf.function
        def run_model(features):
            output, _, _ = model(features, training=True)
            return output
            
        output = run_model(features)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_single_scalar_feature(self):
        """Test TabNet with a single scalar feature"""
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7
        )
        
        # Create single scalar feature
        feature = tf.random.normal((self.batch_size,))
        
        # Test forward pass
        output, _, _ = model(feature, training=True)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_dict_input_auto_grouping(self):
        """Test TabNet with dictionary input and automatic feature grouping"""
        # Create test input with clear grouping structure
        inputs = {
            'embedding_1': tf.random.normal((self.batch_size, 16)),
            'embedding_2': tf.random.normal((self.batch_size, 16)),
            'numeric_1': tf.random.normal((self.batch_size, 1)),
            'numeric_2': tf.random.normal((self.batch_size, 1)),
            'categorical_1': tf.random.normal((self.batch_size, 8)),
            'categorical_2': tf.random.normal((self.batch_size, 8))
        }
        
        # Calculate total feature dimension
        total_dims = sum(tensor.shape[-1] for tensor in inputs.values())
        
        # Initialize model
        model = TabNet(
            feature_dim=total_dims,
            output_dim=self.output_dim,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7
        )
        
        # Test forward pass
        output, masks, _ = model(inputs, training=True)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Verify that embeddings are masked together
        for mask in masks:
            # Check embedding features (first 32 dimensions, 16+16)
            embedding_mask = mask[:, :32]
            self.assertAllClose(
                embedding_mask[:, :16],
                embedding_mask[:, 16:32],
                rtol=1e-5
            )
            
            # Check categorical features (last 16 dimensions, 8+8)
            categorical_mask = mask[:, -16:]
            self.assertAllClose(
                categorical_mask[:, :8],
                categorical_mask[:, 8:],
                rtol=1e-5
            )

    def test_feature_groups_consistency(self):
        """Test that feature groups are handled consistently with official implementation"""
        # Create test data with clear group structure
        feature_dim = 10
        batch_size = 32
        
        # Define feature groups
        grouped_features = [
            [0, 1, 2],  # First group
            [4, 5],     # Second group
            [7, 8, 9]   # Third group
        ]
        
        # Create model
        model = TabNet(
            feature_dim=feature_dim,
            output_dim=1,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=grouped_features
        )
        
        # Create input features
        features = tf.random.normal((batch_size, feature_dim))
        
        # Get output and attention masks
        output, masks, _ = model(features, training=True)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        
        # For each mask, verify that grouped features have same attention values
        for mask in masks:
            # Check first group
            group1_mask = mask[:, [0, 1, 2]]
            self.assertAllClose(
                group1_mask[:, 0],
                group1_mask[:, 1],
                rtol=1e-5
            )
            self.assertAllClose(
                group1_mask[:, 1],
                group1_mask[:, 2],
                rtol=1e-5
            )
            
            # Check second group
            group2_mask = mask[:, [4, 5]]
            self.assertAllClose(
                group2_mask[:, 0],
                group2_mask[:, 1],
                rtol=1e-5
            )
            
            # Check third group
            group3_mask = mask[:, [7, 8, 9]]
            self.assertAllClose(
                group3_mask[:, 0],
                group3_mask[:, 1],
                rtol=1e-5
            )
            self.assertAllClose(
                group3_mask[:, 1],
                group3_mask[:, 2],
                rtol=1e-5
            )

    def test_overlapping_groups_validation(self):
        """Test that overlapping feature groups raise ValueError"""
        feature_dim = 10
        
        # Create overlapping groups
        grouped_features = [
            [0, 1, 2],
            [2, 3, 4]  # Overlaps with first group
        ]
        
        # Verify that creating model with overlapping groups raises ValueError
        with self.assertRaises(ValueError):
            model = TabNet(
                feature_dim=feature_dim,
                output_dim=1,
                n_steps=5,
                gamma=1.3,
                epsilon=1e-5,
                momentum=0.7,
                grouped_features=grouped_features
            )

    def test_invalid_group_indices(self):
        """Test that invalid group indices raise ValueError"""
        feature_dim = 10
        
        # Create groups with invalid indices
        grouped_features = [
            [0, 1, 2],
            [4, 5, 10]  # 10 is invalid for feature_dim=10
        ]
        
        # Verify that creating model with invalid indices raises ValueError
        with self.assertRaises(ValueError):
            model = TabNet(
                feature_dim=feature_dim,
                output_dim=1,
                n_steps=5,
                gamma=1.3,
                epsilon=1e-5,
                momentum=0.7,
                grouped_features=grouped_features
            )

    def test_empty_feature_groups(self):
        """Test that empty feature groups are handled correctly"""
        feature_dim = 10
        batch_size = 32
        
        # Create model with empty feature groups
        model = TabNet(
            feature_dim=feature_dim,
            output_dim=1,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=[]
        )
        
        # Create input features
        features = tf.random.normal((batch_size, feature_dim))
        
        # Get output and attention masks
        output, masks, _ = model(features, training=True)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, 1))
        
        # Verify masks are not all identical (since no grouping)
        for mask in masks:
            # Take first two columns and verify they're different
            col1 = mask[:, 0]
            col2 = mask[:, 1]
            # Check that the columns are different (using mean absolute difference)
            diff = tf.reduce_mean(tf.abs(col1 - col2))
            self.assertGreater(diff, 1e-5)

    def test_single_feature_group(self):
        """Test that a single feature group works correctly"""
        feature_dim = 10
        batch_size = 32
        
        # Create model with single group containing all features
        grouped_features = [list(range(feature_dim))]
        model = TabNet(
            feature_dim=feature_dim,
            output_dim=1,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=grouped_features
        )
        
        # Create input features
        features = tf.random.normal((batch_size, feature_dim))
        
        # Get output and attention masks
        output, masks, _ = model(features, training=True)
        
        # Check that all features in each mask have identical values
        for mask in masks:
            first_col = mask[:, 0]
            for i in range(1, feature_dim):
                self.assertAllClose(mask[:, i], first_col, rtol=1e-5)

    def test_disjoint_feature_groups(self):
        """Test that disjoint feature groups work correctly"""
        feature_dim = 10
        batch_size = 32
        
        # Create disjoint groups
        grouped_features = [
            [0, 1],      # First group
            [4, 5, 6],   # Second group
            [8, 9]       # Third group
        ]
        
        model = TabNet(
            feature_dim=feature_dim,
            output_dim=1,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=grouped_features
        )
        
        features = tf.random.normal((batch_size, feature_dim))
        output, masks, _ = model(features, training=True)
        
        for mask in masks:
            # Check first group
            self.assertAllClose(mask[:, 0], mask[:, 1], rtol=1e-5)
            
            # Check second group
            self.assertAllClose(mask[:, 4], mask[:, 5], rtol=1e-5)
            self.assertAllClose(mask[:, 5], mask[:, 6], rtol=1e-5)
            
            # Check third group
            self.assertAllClose(mask[:, 8], mask[:, 9], rtol=1e-5)
            
            # Verify that groups have different values
            diff1 = tf.reduce_mean(tf.abs(mask[:, 0] - mask[:, 4]))
            diff2 = tf.reduce_mean(tf.abs(mask[:, 4] - mask[:, 8]))
            diff3 = tf.reduce_mean(tf.abs(mask[:, 0] - mask[:, 8]))
            
            self.assertGreater(diff1, 1e-5)  # Group 1 vs 2
            self.assertGreater(diff2, 1e-5)  # Group 2 vs 3
            self.assertGreater(diff3, 1e-5)  # Group 1 vs 3

    def test_feature_groups_with_single_features(self):
        """Test groups with single features mixed with grouped features"""
        feature_dim = 10
        batch_size = 32
        
        # Create groups with some single features
        grouped_features = [
            [0],        # Single feature
            [2, 3, 4],  # Group
            [6],        # Single feature
            [8, 9]      # Group
        ]
        
        model = TabNet(
            feature_dim=feature_dim,
            output_dim=1,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=grouped_features
        )
        
        features = tf.random.normal((batch_size, feature_dim))
        output, masks, _ = model(features, training=True)
        
        for mask in masks:
            # Check group with multiple features
            self.assertAllClose(mask[:, 2], mask[:, 3], rtol=1e-5)
            self.assertAllClose(mask[:, 3], mask[:, 4], rtol=1e-5)
            
            # Check last group
            self.assertAllClose(mask[:, 8], mask[:, 9], rtol=1e-5)
            
            # Verify single features have different values from groups
            diff1 = tf.reduce_mean(tf.abs(mask[:, 0] - mask[:, 2]))
            diff2 = tf.reduce_mean(tf.abs(mask[:, 6] - mask[:, 8]))
            
            self.assertGreater(diff1, 1e-5)  # Single vs group
            self.assertGreater(diff2, 1e-5)  # Single vs group

    def test_feature_groups_normalization(self):
        """Test that mask normalization works correctly with feature groups"""
        feature_dim = 6
        batch_size = 32
        
        # Create two groups of equal size
        grouped_features = [
            [0, 1, 2],  # First group
            [3, 4, 5]   # Second group
        ]
        
        model = TabNet(
            feature_dim=feature_dim,
            output_dim=1,
            n_steps=5,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7,
            grouped_features=grouped_features
        )
        
        features = tf.random.normal((batch_size, feature_dim))
        output, masks, _ = model(features, training=True)
        
        for mask in masks:
            # Check that mask sums to 1 for each sample
            mask_sums = tf.reduce_sum(mask, axis=1)
            self.assertAllClose(mask_sums, tf.ones_like(mask_sums), rtol=1e-5)
            
            # Check that group probabilities sum to approximately 0.5 each
            group1_sum = tf.reduce_sum(mask[:, :3], axis=1)
            group2_sum = tf.reduce_sum(mask[:, 3:], axis=1)
            
            # Each group should get roughly equal attention
            # (allowing for some variation due to the model's learning)
            self.assertTrue(tf.reduce_all(group1_sum >= 0.2))
            self.assertTrue(tf.reduce_all(group1_sum <= 0.8))
            self.assertTrue(tf.reduce_all(group2_sum >= 0.2))
            self.assertTrue(tf.reduce_all(group2_sum <= 0.8))

    def test_dynamic_feature_grouping(self):
        """Test dynamic feature grouping from dictionary input"""
        # Create test input with clear grouping structure
        inputs = {
            'embedding_1': tf.random.normal((self.batch_size, 16)),
            'embedding_2': tf.random.normal((self.batch_size, 16)),
            'numeric_1': tf.random.normal((self.batch_size, 1)),
            'numeric_2': tf.random.normal((self.batch_size, 1)),
            'categorical_1': tf.random.normal((self.batch_size, 8)),
            'categorical_2': tf.random.normal((self.batch_size, 8))
        }
        
        # Calculate total feature dimension
        total_dims = sum(tensor.shape[-1] for tensor in inputs.values())
        
        # Initialize model without explicit grouped_features
        model = TabNet(
            feature_dim=total_dims,
            output_dim=self.output_dim,
            n_d=8,
            n_a=8,
            n_steps=3,
            gamma=1.3,
            epsilon=1e-5,
            momentum=0.7
        )
        
        # First forward pass should trigger automatic grouping
        output, masks, _ = model(inputs, training=True)
        
        # Verify output shape
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # Verify that features are grouped correctly
        for mask in masks:
            # Check embedding features (first 32 dimensions, 16+16)
            embedding_mask = mask[:, :32]
            self.assertAllClose(
                embedding_mask[:, :16],
                embedding_mask[:, 16:32],
                rtol=1e-5
            )
            
            # Check numeric features (next 2 dimensions, 1+1)
            numeric_mask = mask[:, 32:34]
            self.assertAllClose(
                numeric_mask[:, 0],
                numeric_mask[:, 1],
                rtol=1e-5
            )
            
            # Check categorical features (last 16 dimensions, 8+8)
            categorical_mask = mask[:, -16:]
            self.assertAllClose(
                categorical_mask[:, :8],
                categorical_mask[:, 8:],
                rtol=1e-5
            )

if __name__ == '__main__':
    tf.test.main() 