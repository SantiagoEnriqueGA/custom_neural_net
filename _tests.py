import unittest
import numpy as np
from ml_utils import train, ConfigurableNN, DataLoader, EarlyStopping, AdamOptimizer

# class TestTrain(unittest.TestCase):

#     def test_train(self):
#         # Create a simple dataset
#         np.random.seed(42)
#         dataset = np.random.randn(100, 11)  # 100 samples, 10 features + 1 label
#         dataset[:, -1] = (dataset[:, -1] > 0).astype(int)  # Binary labels

#         # Create data loaders
#         train_loader = DataLoader(dataset[:80], batch_size=16)  # 80% for training
#         test_loader = DataLoader(dataset[80:], batch_size=16)   # 20% for testing

#         # Initialize the model
#         model = ConfigurableNN(input_dim=10, hidden_dims=[5, 5], dropout_rate=0.5)

#         # Train the model
#         train(model, train_loader, test_loader, save_dir=None, patience=2, epochs=5)

#         # Check if the model has been trained
#         self.assertGreater(len(model.parameters()), 0, "Model parameters should not be empty after training.")  

class TestConfigurableNN(unittest.TestCase):

    def setUp(self):
        self.input_dim = 10
        self.hidden_dims = [5, 3]
        self.model = ConfigurableNN(self.input_dim, self.hidden_dims)

    def test_initialization(self):
        self.assertEqual(len(self.model.weights), 3)
        self.assertEqual(len(self.model.biases), 3)
        self.assertEqual(self.model.weights[0].shape, (self.input_dim, self.hidden_dims[0]))
        self.assertEqual(self.model.weights[1].shape, (self.hidden_dims[0], self.hidden_dims[1]))
        self.assertEqual(self.model.weights[2].shape, (self.hidden_dims[1], 1))
        self.assertEqual(self.model.biases[0].shape, (self.hidden_dims[0],))
        self.assertEqual(self.model.biases[1].shape, (self.hidden_dims[1],))
        self.assertEqual(self.model.biases[2].shape, (1,))

    def test_forward(self):
        x = np.random.randn(2, self.input_dim)
        output = self.model.forward(x)
        self.assertEqual(output.shape, (2,))

    def test_relu(self):
        x = np.array([-1, 0, 1])
        output = self.model.relu(x)
        np.testing.assert_array_equal(output, [0, 0, 1])

    def test_parameters(self):
        params = self.model.parameters()
        self.assertEqual(len(params), 6)
        for i in range(3):
            np.testing.assert_array_equal(params[2*i], self.model.weights[i])
            np.testing.assert_array_equal(params[2*i + 1], self.model.biases[i])

    def test_train_eval_mode(self):
        self.model.train()
        self.assertEqual(self.model.dropout_rate, 0.5)
        self.model.eval()
        self.assertEqual(self.model.dropout_rate, 0.0)

class TestEarlyStopping(unittest.TestCase):

    def test_early_stopping_initialization(self):
        early_stopping = EarlyStopping(patience=3, delta=0.01)
        self.assertEqual(early_stopping.patience, 3)
        self.assertEqual(early_stopping.delta, 0.01)
        self.assertEqual(early_stopping.counter, 0)
        self.assertEqual(early_stopping.best_loss, float('inf'))
        self.assertFalse(early_stopping.early_stop)

    def test_early_stopping_improvement(self):
        early_stopping = EarlyStopping(patience=3, delta=0.01)
        model = None  # Placeholder for model
        early_stopping(0.5, model)
        self.assertEqual(early_stopping.best_loss, 0.5)
        self.assertEqual(early_stopping.counter, 0)
        self.assertFalse(early_stopping.early_stop)

    def test_early_stopping_no_improvement(self):
        early_stopping = EarlyStopping(patience=3, delta=0.01)
        model = None  # Placeholder for model
        early_stopping(0.5, model)
        early_stopping(0.6, model)
        self.assertEqual(early_stopping.counter, 1)
        self.assertFalse(early_stopping.early_stop)

    def test_early_stopping_trigger(self):
        early_stopping = EarlyStopping(patience=3, delta=0.01)
        model = None  # Placeholder for model
        early_stopping(0.5, model)
        early_stopping(0.6, model)
        early_stopping(0.6, model)
        early_stopping(0.6, model)
        self.assertEqual(early_stopping.counter, 3)
        self.assertTrue(early_stopping.early_stop)

    def test_early_stopping_reset_on_improvement(self):
        early_stopping = EarlyStopping(patience=3, delta=0.01)
        model = None  # Placeholder for model
        early_stopping(0.5, model)
        early_stopping(0.6, model)
        early_stopping(0.6, model)
        early_stopping(0.4, model)
        self.assertEqual(early_stopping.best_loss, 0.4)
        self.assertEqual(early_stopping.counter, 0)
        self.assertFalse(early_stopping.early_stop)
        
        
class TestAdamOptimizer(unittest.TestCase):

    def setUp(self):
        # Initialize parameters
        self.params = [np.array([1.0, 2.0]), np.array([3.0, 4.0])]
        self.optimizer = AdamOptimizer(self.params, learning_rate=0.001)

    def test_initialization(self):
        # Test if the optimizer is initialized correctly
        self.assertEqual(self.optimizer.learning_rate, 0.001)
        self.assertEqual(self.optimizer.beta1, 0.9)
        self.assertEqual(self.optimizer.beta2, 0.999)
        self.assertEqual(self.optimizer.epsilon, 1e-8)
        self.assertEqual(self.optimizer.t, 0)
        self.assertEqual(len(self.optimizer.m), len(self.params))
        self.assertEqual(len(self.optimizer.v), len(self.params))

    def test_step(self):
        # Mock gradients
        for param in self.params:
            param = np.array([0.1, 0.1])

        # Perform a step
        self.optimizer.step()

        # Check if the parameters are updated correctly
        for param in self.params:
            self.assertTrue(np.all(param != np.array([1.0, 2.0])))

    def test_zero_grad(self):
        # Mock gradients
        for param in self.params:
            param = np.array([0.1, 0.1])

        # Zero the gradients
        self.optimizer.zero_grad()

        # Check if the gradients are zeroed
        for param in self.params:
            self.assertTrue(np.all(param.grad == np.zeros_like(param)))


if __name__ == '__main__':
    unittest.main()