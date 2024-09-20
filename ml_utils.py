import numpy as np

class AdamOptimizer:
    """
    Adam optimizer class for training neural networks.
    Formula derived from: https://arxiv.org/abs/1412.6980
    Args:
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
        reg_lambda (float, optional): The regularization parameter. Defaults to 0.01.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01):
        self.learning_rate = learning_rate      # Learning rate, alpha 
        self.beta1 = beta1                      # Exponential decay rate for the first moment estimates
        self.beta2 = beta2                      # Exponential decay rate for the second moment estimates
        self.epsilon = epsilon                  # Small value to prevent division by zero
        self.reg_lambda = reg_lambda            # Regularization parameter, large lambda means more regularization
        
        self.m = []                             # First moment estimates
        self.v = []                             # Second moment estimates
        self.t = 0                              # Time step

    def initialize(self, layers):
        """
        Initializes the first and second moment estimates for each layer's weights.
        Args: layers (list): List of layers in the neural network.
        Returns: None
        """
        for layer in layers:                                # For each layer in the neural network..
            self.m.append(np.zeros_like(layer.weights))     # Initialize first moment estimates
            self.v.append(np.zeros_like(layer.weights))     # Initialize second moment estimates

    def update(self, layer, dW, db, index):
        """
        Updates the weights and biases of a layer using the Adam optimization algorithm.
        Args:
            layer (Layer): The layer to update.
            dW (ndarray): The gradient of the weights.
            db (ndarray): The gradient of the biases.
            index (int): The index of the layer.
        Returns: None
        """
        self.t += 1                                                                     # Increment time step
        self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * dW              # Update first moment estimate, beta1 * m + (1 - beta1) * dW
        self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * np.square(dW)   # Update second moment estimate, beta2 * v + (1 - beta2) * dW^2

        m_hat = self.m[index] / (1 - self.beta1 ** self.t)                              # Bias-corrected first moment estimate, m / (1 - beta1^t)
        v_hat = self.v[index] / (1 - self.beta2 ** self.t)                              # Bias-corrected second moment estimate, v / (1 - beta2^t)

        layer.weights -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.reg_lambda * layer.weights)   # Update weights, alpha * m_hat / (sqrt(v_hat) + epsilon) + lambda * weights
        layer.biases -= self.learning_rate * db                                                                             # Update biases, alpha * db

class CrossEntropyLoss:
    """
    Custom cross entropy loss implementation using numpy for multi-class classification.
    Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
    Methods:
        __call__(self, logits, targets): Calculate the cross entropy loss.
    """
    def __call__(self, logits, targets):
        """
        Calculate the cross entropy loss.
        Args:
            logits (np.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
            targets (np.ndarray): The target labels of shape (num_samples, num_classes).
        Returns:
            float: The cross entropy loss.
        """      
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # Exponential of logits, subtract max to prevent overflow
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)      # Probabilities from logits
        loss = -np.sum(targets * np.log(probs + 1e-15)) / logits.shape[0]   # Cross-entropy loss
        
        return loss

class BCEWithLogitsLoss:
    """
    Custom binary cross entropy loss with logits implementation using numpy.
    Formula: -mean(y * log(p) + (1 - y) * log(1 - p))
    Methods:
        __call__(self, logits, targets): Calculate the binary cross entropy loss.
    """
    def __call__(self, logits, targets):
        """
        Calculate the binary cross entropy loss.
        Args:
            logits (np.ndarray): The logits (predicted values) of shape (num_samples,).
            targets (np.ndarray): The target labels of shape (num_samples,).
        Returns:
            float: The binary cross entropy loss.
        """
        probs = 1 / (1 + np.exp(-logits))                                                               # Apply sigmoid to logits to get probabilities
        loss = -np.mean(targets * np.log(probs + 1e-15) + (1 - targets) * np.log(1 - probs + 1e-15))    # Binary cross-entropy loss
        
        return loss

class NeuralNetwork:
    """
    Neural network class for training and evaluating a custom neural network model.
    Parameters:
        - layer_sizes (list): A list of integers representing the sizes of each layer in the neural network.
        - dropout_rate (float): The dropout rate to be applied during training. Default is 0.2.
        - reg_lambda (float): The regularization lambda value. Default is 0.01.
    """
    
    def __init__(self, layer_sizes, dropout_rate=0.2, reg_lambda=0.01):
        self.layers = []                                                        # List to store the layers of the neural network
        self.dropout_rate = dropout_rate                                        # Dropout rate
        self.reg_lambda = reg_lambda                                            # Regularization lambda
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i]))         # Create layers with input and output sizes

    def forward(self, X):
        """
        Performs forward propagation through the neural network.
        Args: X (ndarray): Input data of shape (batch_size, input_size).
        Returns: ndarray: Output predictions of shape (batch_size, output_size).
        """
        A = X                                                               # Input layer
        self.activations = [X]                                              # Store activations for backpropagation
        for layer in self.layers[:-1]:                                      # Loop through all layers except the last one
            Z = np.dot(A, layer.weights) + layer.biases                     # Linear transformation
            A = np.maximum(0, Z)                                            # ReLU activation
            if self.dropout_rate > 0:
                A = self.apply_dropout(A)                                   # Apply dropout regularization, if applicable
            self.activations.append(A)                                      # Store activations for backpropagation
        Z = np.dot(A, self.layers[-1].weights) + self.layers[-1].biases     # Output layer linear transformation
        outputs = sigmoid(Z)                                                # Sigmoid activation for classification
        self.activations.append(outputs)                                    # Store activations for backpropagation
        
        return outputs

    def apply_dropout(self, A):
        """
        Applies dropout regularization to the input array.
        Parameters: A: numpy.ndarray: Input array to apply dropout regularization to.
        Returns: numpy.ndarray: Array with dropout regularization applied.
        """
        if self.dropout_rate > 0:                           # If dropout rate is greater than 0
            keep_prob = 1 - self.dropout_rate               # Calculate keep probability
            mask = np.random.rand(*A.shape) < keep_prob     # Create a mask for the dropout
            A = np.multiply(A, mask)                        # Apply the mask to the input array
            A /= keep_prob                                  # Scale the output of the dropout layer
        return A

    def backward(self, y):
        """
        Performs backward propagation to calculate the gradients of the weights and biases in the neural network.
        Parameters: y (numpy.ndarray): Target labels of shape (m, 1), where m is the number of samples.
        Returns: None
        """
        m = y.shape[0]          # Number of samples
        y = y.reshape(-1, 1)    # Reshape y to ensure it is a column vector
        
        outputs = self.activations[-1]                                      # Output predictions
        dA = -(y / (outputs + 1e-15) - (1 - y) / (1 - outputs + 1e-15))     # Gradient of the loss function with respect to the output

        for i in reversed(range(len(self.layers))):
            dZ = dA * sigmoid_derivative(self.activations[i + 1])           # Gradient of the loss function with respect to the linear output
            
            dW = np.dot(self.activations[i].T, dZ) / m + self.reg_lambda * self.layers[i].weights   # Gradient of the loss function
            db = np.sum(dZ, axis=0, keepdims=True) / m                                              # Bias of the loss function 
            
            dA = np.dot(dZ, self.layers[i].weights.T)                                               # dA for the next iteration

            self.layers[i].gradients = (dW, db)                                                     # Store the gradients

    def train(self, X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32, early_stopping_threshold=5):
        """
        Trains the neural network model.
        Parameters:
            - X_train (numpy.ndarray): Training data features.
            - y_train (numpy.ndarray): Training data labels.
            - X_test (numpy.ndarray): Test data features.
            - y_test (numpy.ndarray): Test data labels.
            - optimizer (Optimizer): The optimizer used for updating the model parameters.
            - epochs (int): Number of training epochs (default: 100).
            - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
            - early_stopping_threshold (int): Number of epochs to wait for improvement in training loss before early stopping (default: 5).
        Returns: None
        """
        optimizer.initialize(self.layers)   # Initialize the optimizer
        best_loss = float('inf')            # Initialize best loss to infinity
        patience = 0                        # Initialize patience for early stopping

        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])                   # Create indices for shuffling
            np.random.shuffle(indices)                              # Shuffle indices
            X_train, y_train = X_train[indices], y_train[indices]   # Shuffle the training data

            for i in range(0, X_train.shape[0], batch_size):        # Loop through the training data in batches
                X_batch = X_train[i:i+batch_size]                   # X of the current batch
                y_batch = y_train[i:i+batch_size]                   # y of the current batch

                outputs = self.forward(X_batch)                     # Perform forward pass, get predictions
                self.backward(y_batch)                              # Perform backward pass, calculate gradients

                for idx, layer in enumerate(self.layers):   # For each layer in the neural network..
                    dW, db = layer.gradients                # Get the gradients, weights and biases
                    optimizer.update(layer, dW, db, idx)    # Update the weights and biases using the optimizer

            # Calculate training loss and accuracy
            train_loss = self.calculate_loss(X_train, y_train)
            train_accuracy, train_pred = self.evaluate(X_train, y_train)
            
            # Calculate test loss and accuracy
            test_loss = self.calculate_loss(X_test, y_test)
            test_accuracy, test_pred = self.evaluate(X_test, y_test)
            
            # Print the training progress
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            # Implement early stopping
            if train_loss < best_loss:      # If training loss improves
                best_loss = train_loss      # Update best loss
                patience = 0                # Reset patience
            else:                           # If training loss does not improve
                patience += 1               # Increment patience
            if patience >= early_stopping_threshold:    # If patience exceeds the threshold, stop training
                print("Early stopping triggered")       
                break

    def calculate_loss(self, X, y, class_weights=None):
        """
        Calculates the loss of the neural network model.
        Formula: loss = loss_fn(outputs, y) + reg_lambda * sum([sum(layer.weights**2) for layer in self.layers])
        Parameters:
            - X (numpy.ndarray): Input data of shape (num_samples, num_features).
            - y (numpy.ndarray): Target labels of shape (num_samples,).
            - class_weights (numpy.ndarray, optional): Weights for each class. Default is None.
        Returns: loss (float): The calculated loss value.
        """
        outputs = self.forward(X)               # Perform forward pass to get predictions
        if class_weights is None:
            class_weights = np.ones_like(y)     # If class weights are not provided, set them to 1
        
        if self.layers[-1].weights.shape[1] > 1:        # Multi-class classification, cross-entropy loss
            loss_fn = CrossEntropyLoss()
            loss = loss_fn(outputs, y.reshape(-1, 1)) 
        else:                                           # Binary classification, binary cross-entropy loss
            loss_fn = BCEWithLogitsLoss()
            loss = loss_fn(outputs, y)
        
        loss += self.reg_lambda * np.sum([np.sum(layer.weights**2) for layer in self.layers])   # Loss with L2 regularization
        
        return loss

    def evaluate(self, X, y):
        """
        Evaluates the performance of the neural network model on the given input data.
        Parameters:
            - X (numpy.ndarray): The input data for evaluation.
            - y (numpy.ndarray): The target labels for evaluation.
        Returns:
            - accuracy (float): The accuracy of the model's predictions.
            - predicted (numpy.ndarray): The labels predicted by the model.
        """
        y_hat = self.forward(X)                             # Perform forward pass to get predictions
        
        if self.layers[-1].weights.shape[1] > 1:            # Multi-class classification
            predicted = np.argmax(y_hat, axis=1)            # Get the class with the highest probability
            accuracy = np.mean(predicted == np.argmax(y))   # Calculate accuracy
        
        else:                                               # Binary classification
            predicted = (y_hat > 0.5).astype(int)           # Convert probabilities to binary predictions
            accuracy = np.mean(predicted == y)              # Calculate accuracy
        
        return accuracy, predicted                          # Return accuracy and predicted labels

class Layer:
    """
    Initializes a Layer object.
    Args:
        input_size (int): The size of the input to the layer.
        output_size (int): The size of the output from the layer.
    """
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01  # Initialize weights with small random values
        self.biases = np.zeros((1, output_size))                        # Initialize biases with zeros
        self.gradients = None                                           # Initialize gradients to None

def sigmoid(z):
    # Sigmoid function with clipping to prevent overflow
    return 1 / (1 + np.exp(-np.clip(z, -500, 500))) 

def sigmoid_derivative(z):
    # Derivative of the sigmoid function
    sig = sigmoid(z)
    return sig * (1 - sig)

def test_breast_cancer():
    """Test the neural network model on the Breast Cancer dataset."""
    import random
    np.random.seed(42)
    random.seed(42)
    
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target.reshape(-1, 1)

    print(f"\nTesting on Breast Cancer dataset:")
    print(f"--------------------------------------------------------------------------")
    print(f"X shape: {X.shape}, Y shape: {y.shape}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the neural network and optimizer
    nn = NeuralNetwork([X_train.shape[1], 100, 25, 1], dropout_rate=0.2, reg_lambda=0.0)
    optimizer = AdamOptimizer(learning_rate=0.001)

    # Train the neural network
    nn.train(X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32)

    # Evaluate the neural network
    accuracy, predicted = nn.evaluate(X_test, y_test)
    print(f"Final Accuracy: {accuracy}")

    # Print classification report
    from sklearn.metrics import classification_report
    print("Classification Report:")
    print(classification_report(y_test, predicted, zero_division=0))
    
def test_iris():
    """"Test the neural network model on the Iris dataset."""
    import random
    np.random.seed(42)
    random.seed(42)
    
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load the dataset
    data = load_iris()
    X = data.data
    y = data.target

    print(f"\nTesting on Iris dataset:")
    print(f"--------------------------------------------------------------------------")
    print(f"X shape: {X.shape}, Y shape: {y.shape}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the dataset
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Initialize the neural network and optimizer
    nn = NeuralNetwork([X_train.shape[1], 100, 25, 3], dropout_rate=0.1, reg_lambda=0.0)
    optimizer = AdamOptimizer(learning_rate=0.001)

    # Train the neural network
    nn.train(X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32, early_stopping_threshold=10)

    # Evaluate the neural network
    accuracy, predicted = nn.evaluate(X_test, y_test)
    print(f"Final Accuracy: {accuracy}")

    # Print classification report
    from sklearn.metrics import classification_report
    print("Classification Report:")
    print(classification_report(y_test, predicted, zero_division=0))


if __name__ == "__main__":
    test_breast_cancer()
    test_iris()