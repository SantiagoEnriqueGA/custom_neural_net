import numpy as np

class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.m = []
        self.v = []
        self.t = 0

    def initialize(self, layers):
        for layer in layers:
            self.m.append(np.zeros_like(layer.weights))
            self.v.append(np.zeros_like(layer.weights))

    def update(self, layer, dW, db, index):
        self.t += 1
        self.m[index] = self.beta1 * self.m[index] + (1 - self.beta1) * dW
        self.v[index] = self.beta2 * self.v[index] + (1 - self.beta2) * np.square(dW)

        m_hat = self.m[index] / (1 - self.beta1 ** self.t)
        v_hat = self.v[index] / (1 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * (m_hat / (np.sqrt(v_hat) + self.epsilon) + self.reg_lambda * layer.weights)
        layer.biases -= self.learning_rate * db

class NeuralNetwork:
    def __init__(self, layer_sizes, dropout_rate=0.2, reg_lambda=0.01):
        self.layers = []
        self.dropout_rate = dropout_rate
        self.reg_lambda = reg_lambda
        for i in range(1, len(layer_sizes)):
            self.layers.append(Layer(layer_sizes[i-1], layer_sizes[i]))

    def forward(self, X):
        A = X
        self.activations = [X]
        for layer in self.layers[:-1]:
            Z = np.dot(A, layer.weights) + layer.biases
            A = np.maximum(0, Z)  # ReLU activation
            if self.dropout_rate > 0:
                A = self.apply_dropout(A)
            self.activations.append(A)
        # Last layer (output layer with sigmoid)
        Z = np.dot(A, self.layers[-1].weights) + self.layers[-1].biases
        outputs = sigmoid(Z)
        self.activations.append(outputs)
        return outputs

    def apply_dropout(self, A):
        drop_mask = np.random.rand(*A.shape) > self.dropout_rate
        return A * drop_mask / (1 - self.dropout_rate)

    def backward(self, X, y):
        m = y.shape[0]
        outputs = self.activations[-1]
        dA = -(y / outputs - (1 - y) / (1 - outputs))  # Gradient of the loss function with respect to the output

        for i in reversed(range(len(self.layers))):
            # Calculate dZ using the derivative of the activation function
            dZ = dA * sigmoid_derivative(self.activations[i + 1])
            # Calculate gradients for weights and biases
            dW = np.dot(self.activations[i].T, dZ) / m + self.reg_lambda * self.layers[i].weights
            db = np.sum(dZ, axis=0) / m
            # Update dA for the next layer
            dA = np.dot(dZ, self.layers[i].weights.T)

            # Store gradients in the layer
            self.layers[i].gradients = (dW, db)

    def train(self, X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32, early_stopping_threshold=5):
        optimizer.initialize(self.layers)
        best_loss = float('inf')
        patience = 0

        for epoch in range(epochs):
            # Shuffle data for each epoch
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]

                outputs = self.forward(X_batch)
                self.backward(X_batch, y_batch)

                for idx, layer in enumerate(self.layers):
                    dW, db = layer.gradients
                    optimizer.update(layer, dW, db, idx)

            # Calculate training loss and accuracy
            train_loss = self.calculate_loss(X_train, y_train)
            train_accuracy, _ = self.evaluate(X_train, y_train)
            
            # Calculate test loss and accuracy
            test_loss = self.calculate_loss(X_test, y_test)
            test_accuracy, _ = self.evaluate(X_test, y_test)
            
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
            
            if train_loss < best_loss:
                best_loss = train_loss
                patience = 0
            else:
                patience += 1
            if patience >= early_stopping_threshold:
                print("Early stopping triggered")
                break

    def calculate_loss(self, X, y, class_weights=None):
        outputs = self.forward(X)
        if class_weights is None:
            class_weights = np.ones_like(y)
        loss = -np.mean(class_weights * (y * np.log(outputs) + (1 - y) * np.log(1 - outputs)))
        # Add L2 regularization to the loss
        loss += self.reg_lambda * np.sum([np.sum(layer.weights**2) for layer in self.layers])
        return loss

    def evaluate(self, X, y):
        # Perform forward pass to get predictions
        y_hat = self.forward(X)
        # Convert predicted probabilities to binary labels (0 or 1)
        predicted = (y_hat > 0.5).astype(int)
        # Compute accuracy
        accuracy = np.mean(predicted == y)
        # Return both accuracy and predictions
        return accuracy, predicted


class Layer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.gradients = None

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)


# Example usage:
# nn = NeuralNetwork([2, 100, 25, 1], dropout_rate=0.2, reg_lambda=0.01)
# optimizer = AdamOptimizer(learning_rate=0.01)

# X = np.random.randn(100, 2)  # Example data
# y = np.random.randint(0, 2, (100, 1))  # Example labels

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# nn.train(X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32)
# accuracy, predicted = nn.evaluate(X, y)
# print(f"Final Accuracy: {accuracy}")

# print(predicted)
