import tensorflow as tf
import numpy as np
from typing import List, Dict
from tabnet_block import TabNet

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
            output_dim=self.output_dim
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
            output_dim=self.output_dim
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
        
        feature_groups = [
            list(range(embedding1_start, embedding1_end)),  # embedding1 group
            list(range(embedding2_start, embedding2_end))   # embedding2 group
        ]
        
        # Initialize model with feature groups
        model = TabNet(
            feature_dim=total_dims,
            output_dim=self.output_dim,
            feature_groups=feature_groups
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
            output_dim=self.output_dim
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
            output_dim=self.output_dim
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
            output_dim=self.output_dim
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

if __name__ == '__main__':
    tf.test.main() 