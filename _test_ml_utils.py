import unittest
import numpy as np
from ml_utils import NeuralNetwork, AdamOptimizer, Layer, sigmoid

class TestNeuralNetwork(unittest.TestCase):

    def setUp(self):
        self.nn = NeuralNetwork([2, 100, 25, 1], dropout_rate=0.2, reg_lambda=0.01)             # 2 input, 1 output, 2 hidden layers
        self.optimizer = AdamOptimizer(learning_rate=0.01)                                      # Adam optimizer
        np.random.seed(42) 
        self.X = np.random.randn(100, 2)                                                        # Example data
        self.y = np.random.randint(0, 2, (100, 1))                                              # Example labels
        self.nn.train(self.X, self.y, self.X, self.y, self.optimizer, epochs=1, batch_size=32)  # Train the model

    def test_forward(self):
        outputs = self.nn.forward(self.X)                               # Forward pass
        self.assertEqual(outputs.shape, (100, 1))                       # Check output, shape
        self.assertTrue(np.all(outputs >= 0) and np.all(outputs <= 1))  # Check output, range

    def test_backward(self):
        self.nn.forward(self.X)                                 # Forward pass
        self.nn.backward(self.X, self.y)                        # Backward pass
        for layer in self.nn.layers:                            # Check gradients                
            dW, db = layer.gradients                                # Get gradients
            self.assertEqual(dW.shape, layer.weights.shape)         # Check shape, weights
            self.assertEqual(db.shape, layer.biases.shape)          # Check shape, biases

    def test_train(self):
        initial_loss = self.nn.calculate_loss(self.X, self.y)                                       # Initial loss
        self.nn.train(self.X, self.y, self.X, self.y, self.optimizer, epochs=10, batch_size=32)     # Train the model
        final_loss = self.nn.calculate_loss(self.X, self.y)                                         # Final loss
        self.assertLess(final_loss, initial_loss)                                                   # Check loss, decrease

    def test_evaluate(self):
        accuracy, predicted = self.nn.evaluate(self.X, self.y)      # Evaluate the model
        self.assertTrue(0 <= accuracy <= 1)                         # Check accuracy, range
        self.assertEqual(predicted.shape, self.y.shape)             # Check predicted, shape

    def test_calculate_loss(self):
        loss = self.nn.calculate_loss(self.X, self.y)               # Calculate loss
        self.assertGreater(loss, 0)                                 # Check loss, positive

if __name__ == '__main__':
    unittest.main()