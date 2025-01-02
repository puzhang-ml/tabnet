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
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim
        )
        
        # Create single tensor input
        features = tf.random.normal((self.batch_size, 60))
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
        
        model = TabNet(
            feature_dim=self.feature_dim,
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
        
        # Create group matrix for embeddings
        group_matrix = np.zeros((self.feature_dim, self.feature_dim))
        current_idx = 2  # After two scalar features
        
        # Group embedding1 dimensions
        embedding1_start = current_idx
        embedding1_end = embedding1_start + embedding1_dim
        group_matrix[embedding1_start:embedding1_end, embedding1_start:embedding1_end] = 1
        current_idx = embedding1_end
        
        # Group embedding2 dimensions
        embedding2_start = current_idx
        embedding2_end = embedding2_start + embedding2_dim
        group_matrix[embedding2_start:embedding2_end, embedding2_start:embedding2_end] = 1
        current_idx = embedding2_end
        
        # Initialize model with group matrix
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim,
            group_matrix=group_matrix
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
            embedding1_mask = mask[:, embedding1_start:embedding1_end]
            self.assertTrue(tf.reduce_all(embedding1_mask == embedding1_mask[:, 0:1]))
            
            # Check embedding2 mask values are identical within the group
            embedding2_mask = mask[:, embedding2_start:embedding2_end]
            self.assertTrue(tf.reduce_all(embedding2_mask == embedding2_mask[:, 0:1]))
        
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

if __name__ == '__main__':
    tf.test.main() 