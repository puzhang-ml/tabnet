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
            return model(features, training=True)
            
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
            return model(features, training=True)
            
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
        output = model(features, training=True)
        
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
        
        output = model(features, training=True)
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
    def test_dynamic_shapes(self):
        """Test TabNet with dynamic input shapes"""
        model = TabNet(
            feature_dim=self.feature_dim,
            output_dim=self.output_dim
        )
        
        # Create features with unknown shapes
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, None), dtype=tf.float32),
            tf.TensorSpec(shape=(None, None), dtype=tf.float32)
        ])
        def run_model(feat1, feat2):
            return model([feat1, feat2], training=True)
            
        feat1 = tf.random.normal((self.batch_size, 10))
        feat2 = tf.random.normal((self.batch_size, 20))
        output = run_model(feat1, feat2)
        
        self.assertEqual(output.shape[0], self.batch_size)
        self.assertEqual(output.shape[1], self.output_dim)

if __name__ == '__main__':
    tf.test.main() 