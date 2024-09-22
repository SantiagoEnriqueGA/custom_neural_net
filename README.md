# Custom Neural Network Project

## Goal
This project implements a fully customizable neural network using just Python and NumPy. It supports various features like multiple layers, different activation functions, various optimization functions, cross-entropy loss, binary classification, dropout, and L2 regularization.

The project aims to demonstrate how neural networks can be built and trained using only core Python libraries, without relying on high-level frameworks like TensorFlow or PyTorch. By doing this, you gain deeper insight into the underlying mechanisms of backpropagation, gradient descent, and optimization algorithms.

---

## ml_utils.py

This module contains custom implementations of the KMeans and DBSCAN clustering algorithms, along with additional functionalities for evaluating clustering performance and visualizing the clustering results.

### Classes

- **AdamOptimizer:** Implements the Adam optimization algorithm for training neural networks. It includes parameters such as learning rate, beta values for moment estimates, and a regularization parameter.

- **CrossEntropyLoss:** Custom implementation of the cross-entropy loss for multi-class classification, calculated using the formula:
  \[
  \text{Loss} = -\frac{1}{m} \sum (y \cdot \log(p) + (1 - y) \cdot \log(1 - p))
  \]
  where `m` is the number of samples.

- **BCEWithLogitsLoss:** Custom binary cross-entropy loss implementation with logits. It calculates loss using the formula:
  \[
  \text{Loss} = -\text{mean}(y \cdot \log(p) + (1 - y) \cdot \log(1 - p))
  \]
  This class applies the sigmoid function to logits to obtain probabilities.

- **NeuralNetwork:** A class for training and evaluating a custom neural network model. Key features include:
  - Supports multiple layers with customizable sizes and activation functions.
  - Implements forward and backward propagation.
  - Supports dropout regularization and L2 regularization.
  - Includes a method for training with mini-batch gradient descent, along with early stopping.
  - Provides functionality for hyperparameter tuning via grid search.
  - Evaluates model performance on training and test data.

- **Layer:** Represents a single layer in the neural network. Contains attributes for `weights`, `biases`, `activation_function`, and `gradients`.
  - **Methods:**
    - `activate(Z)`: Applies the specified activation function.
    - `activation_derivative(Z)`: Returns the derivative of the activation function for backpropagation.

- **Activation:** Contains static methods for various activation functions and their derivatives.
  - **Methods:**
    - `relu(z)`, `relu_derivative(z)`: ReLU activation and its derivative.
    - `leaky_relu(z, alpha)`, `leaky_relu_derivative(z, alpha)`: Leaky ReLU activation and its derivative.
    - `tanh(z)`, `tanh_derivative(z)`: Tanh activation and its derivative.
    - `sigmoid(z)`, `sigmoid_derivative(z)`: Sigmoid activation and its derivative.
    - `softmax(z)`: Softmax activation function.

### Testing Functions

- **test_breast_cancer():** Tests the neural network on the Breast Cancer dataset.
  - Loads the dataset, splits it into training and test sets, and standardizes the features.
  - Initializes the neural network and optimizer, trains the model, and evaluates accuracy.
  - Prints a classification report for performance metrics.

- **test_iris():** Tests the neural network on the Iris dataset.
  - Similar to the breast cancer testing function, it loads, splits, and standardizes the Iris dataset.
  - Initializes the neural network and optimizer, trains the model, and evaluates accuracy.
  - Outputs a classification report for performance metrics.
