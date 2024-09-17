import re
from scipy import optimize
from sklearn.metrics import classification_report
from tqdm import tqdm
import datetime
import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1/(1 + np.exp(-z))

class DataLoader:
    """
    Custom data loader implementation using numpy.

    Args:
        dataset (np.ndarray): The dataset to load.
        batch_size (int): The batch size.

    Attributes:
        dataset (np.ndarray): The dataset to load.
        batch_size (int): The batch size.
        num_samples (int): The number of samples in the dataset.
        num_batches (int): The number of batches in the dataset.
        current_batch (int): The index of the current batch.

    Methods:
        __iter__(self): Return an iterator object.
        __next__(self): Get the next batch of data.
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)
        self.num_batches = (self.num_samples + self.batch_size - 1) // self.batch_size
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.num_samples)
        batch = self.dataset[start_idx:end_idx]
        self.current_batch += 1

        return batch
        
class ConfigurableNN:
    """
    A configurable neural network module.

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dims (list): A list of integers specifying the number of units in each hidden layer.
        dropout_rate (float, optional): The dropout rate to be applied to the hidden layers. Defaults to 0.5.
    """

    def __init__(self, input_dim, hidden_dims, dropout_rate=0.5):   # Initialize the module
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.weights = []
        self.biases = []
        self.activations = []

        # Initialize the weights and biases for each layer
        for i in range(len(hidden_dims)):                                               # Iterate over the number of hidden layers
            if i == 0:
                self.weights.append(np.random.randn(input_dim, hidden_dims[i]))         # Initialize the weight matrix for the first layer
            else:
                self.weights.append(np.random.randn(hidden_dims[i-1], hidden_dims[i]))  # Initialize the weight matrix for the subsequent layers
            
            self.biases.append(np.zeros(hidden_dims[i]))                                # Initialize the bias vector for each layer

        self.weights.append(np.random.randn(hidden_dims[-1], 1))                        # Initialize the weight matrix for the output layer
        self.biases.append(np.zeros(1))                                                 # Initialize the bias vector for the output layer

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array.
        """
        self.activations = []
        self.activations.append(x)

        for i in range(len(self.weights)):                                      # Iterate over the number of layers
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]   # Compute the linear transformation, z = wx + b, z: [batch_size, hidden_dim]
            a = self.relu(z)                                                    # Apply the activation function, a: [batch_size, hidden_dim]
            self.activations.append(a)                                          # Append the activation to the list

        return self.activations[-1].squeeze()  

    def relu(self, x):
        """
        ReLU activation function.

        Args:
            x (np.ndarray): The input array.

        Returns:
            np.ndarray: The output array.
        """
        return np.maximum(0, x)

    def parameters(self):
        """
        Get the parameters of the neural network.

        Returns:
            list: The list of parameters.
        """
        params = []
        for i in range(len(self.weights)):
            params.append(self.weights[i])
            params.append(self.biases[i])
        return params

    def train(self):
        """
        Set the model to training mode.
        """
        self.dropout_rate = 0.5

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.dropout_rate = 0.0
    
class EarlyStopping:
    """
    Class for implementing early stopping during model training.

    Args:
        patience (int): Number of epochs to wait for improvement.
        delta (float): Minimum change in validation loss to be considered as improvement.

    Attributes:
        patience (int): Number of epochs to wait for improvement.
        delta (float): Minimum change in validation loss to be considered as improvement.
        counter (int): Counter to keep track of epochs without improvement.
        best_loss (float): The best loss achieved so far.
        early_stop (bool): Flag to indicate if early stopping condition is met.

    Methods:
        __call__(self, val_loss, model, save_dir=None): Check if validation loss improved and perform early stopping if necessary.
    """
    def __init__(self, patience=5, delta=0): # Initialize the early stopping object
        self.patience = patience        # Number of epochs to wait for improvement
        self.delta = delta              # Minimum change in validation loss to be considered as improvement
        self.counter = 0                # Counter to keep track of epochs without improvement
        self.best_loss = float('inf')   # Initialize the best loss with infinity
        self.early_stop = False         # Flag to indicate if early stopping condition is met

    def __call__(self, val_loss, model, save_dir=None): # Call the early stopping object
        """
        Check if validation loss improved and perform early stopping if necessary.

        Args:
            val_loss (float): The validation loss.
            model: The model being trained.
            save_dir (str, optional): The directory to save the best model. Defaults to None.
        """
        if val_loss < self.best_loss - self.delta:  # Check if validation loss improved
            self.best_loss = val_loss               # Update the best loss
            self.counter = 0                        # Reset the counter
            if save_dir:    
                # TODO: Save the model
                # print('TBD: Model improved and saved.')
                i=1
        else:
            self.counter += 1                   # Increment the counter
            if self.counter >= self.patience:   # Check if early stopping condition is met
                self.early_stop = True          # Set the early stop flag
                print(f"Early stopping triggered after {self.counter} epochs.")
             
class Parameter:
    """
    Custom parameter class with gradient attribute.

    Args:
        data (np.ndarray): The parameter data.

    Attributes:
        data (np.ndarray): The parameter data.
        grad (np.ndarray): The gradient of the parameter.

    """
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

class AdamOptimizer:
    """
    Custom Adam optimizer implementation using numpy.

    Args:
        parameters (list): List of parameters to optimize.
        learning_rate (float, optional): The learning rate. Defaults to 0.001.
        beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
        beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
        epsilon (float, optional): A small constant for numerical stability. Defaults to 1e-8.

    Methods:
        step(self): Perform a single optimization step.
        zero_grad(self): Zero the gradients of all parameters.
        compute_gradients(self, model, inputs, labels, criterion): Compute gradients for all parameters.
        update_gradients(self, grads): Update the gradients of all parameters.
    """
    def __init__(self, parameters, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = [Parameter(param) for param in parameters]
        self.learning_rate = learning_rate 
        self.beta1 = beta1                  # Exponential decay rate for the first moment estimates
        self.beta2 = beta2                  # Exponential decay rate for the second moment estimates
        self.epsilon = epsilon              # Small constant for numerical stability
        self.m = [np.zeros_like(param.data) for param in self.parameters]  # First moment estimate
        self.v = [np.zeros_like(param.data) for param in self.parameters]  # Second moment estimate
        self.t = 0                          # Time step

    def step(self):
        self.t += 1                                         # Increment the time step
        for i, param in enumerate(self.parameters):         # Iterate over the parameters
            gradient = param.grad                           # Get the gradient
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradient        # Update the first moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * gradient**2     # Update the second moment estimate
            m_hat = self.m[i] / (1 - self.beta1**self.t)                            # Bias-corrected first moment estimate
            v_hat = self.v[i] / (1 - self.beta2**self.t)                            # Bias-corrected second moment estimate
            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)   # Update the parameters

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.data)

    def compute_gradients(self, model, inputs, labels, criterion):
        """
        Compute gradients for all parameters.

        Args:
            model: The model being trained.
            inputs (np.ndarray): The input data.
            labels (np.ndarray): The target labels.
            criterion: The loss function.

        Returns:
            list: A list of gradients for each parameter.
        """
        grads = []
        for param in model.parameters():
            grad = np.zeros_like(param)
            for i in range(param.size):
                original_value = param.flat[i]
                param.flat[i] = original_value + 1e-5
                loss1 = criterion(model.forward(inputs), labels)
                param.flat[i] = original_value - 1e-5
                loss2 = criterion(model.forward(inputs), labels)
                param.flat[i] = original_value
                grad.flat[i] = (loss1 - loss2) / (2 * 1e-5)
            grads.append(grad)
        return grads

    def update_gradients(self, grads):
        """
        Update the gradients of all parameters.

        Args:
            grads (list): A list of gradients for each parameter.
        """
        for param, grad in zip(self.parameters, grads):
            param.grad = grad
            
class Scaler:
    """
    Custom scaler class for scaling the gradients during backpropagation.

    Methods:
        scale(self, loss): Scale the loss.
        step(self, optimizer): Update the weights.
        update(self): Update the scaler.
    """
    def __init__(self):
        self.scale_factor = 1.0         # Initialize the scale factor

    def scale(self, loss):
        """Scale the loss."""
        return loss * self.scale_factor  # Scale the loss

    def step(self, optimizer):
        """Update the weights."""
        optimizer.step()

    def update(self):
        """Update the scaler."""
        self.scale_factor *= 0.99
        
class BCEWithLogitsLoss:
    """
    Custom binary cross entropy with logits loss implementation using numpy.

    Methods:
        __call__(self, logits, targets): Calculate the binary cross entropy with logits loss.
    """
    def __call__(self, logits, targets):
        """
        Calculate the binary cross entropy with logits loss.

        Args:
            logits (np.ndarray): The logits.
            targets (np.ndarray): The target labels.

        Returns:
            float: The binary cross entropy with logits loss.
        """
        loss = np.maximum(logits, 0) - logits * targets + np.log(1 + np.exp(-np.abs(logits)))   # Calculate the loss element-wise
        return np.mean(loss)

def train(model, train_loader, test_loader, save_dir, patience=5, epochs=10):
    """
    Trains a given model using the specified data loaders, criterion, and scheduler.

    Args:
        model: The model to be trained.
        train_loader: The data loader for the training set.
        test_loader: The data loader for the test set.
        save_dir (str): The directory to save the trained model.
        patience (int, optional): The number of epochs to wait for improvement before early stopping. Defaults to 5.
        epochs (int, optional): The number of epochs to train the model. Defaults to 10.
    """
    early_stopping = EarlyStopping(patience)    # Initialize the early stopping object
    
    scaler = Scaler()                               # Initialize the scaler
    optimizer = AdamOptimizer(model.parameters())   # Initialize the optimizer
    criterion = BCEWithLogitsLoss()                 # Initialize the loss function
    
    for epoch in range(epochs): # Iterate over the specified number of epochs
        model.train()           # Set the model to training mode
        running_loss = 0.0      # Initialize the running loss
        correct = 0             # Initialize the number of correct predictions
        total = 0               # Initialize the total number of predictions
        
        # Iterate over the training data
        for data in train_loader:                       # Iterate over the training data
            inputs, labels = data[:, :-1], data[:, -1]
            
            optimizer.zero_grad()                     # Zero the gradients
                        
            outputs = model.forward(inputs)             # Forward pass
            loss = criterion(outputs, labels)           # Calculate the loss

            grads = optimizer.compute_gradients(model, inputs, labels, criterion)
            optimizer.update_gradients(grads)
                            
            scaler.step(optimizer)          # Update the weights
            scaler.update()                 # Update the GradScaler
            
            running_loss += loss * inputs.size              # Accumulate the running loss            
            predicted = np.round(sigmoid(outputs))          # Get the predicted labels
            total += labels.size                         # Increment the total count
            correct += (predicted == labels).sum().item()   # Increment the correct count
            
            # print(f"Optimizer.parameters.grad: {optimizer.parameters[0].grad}"
            #       f"Optimizer.parameters.data: {optimizer.parameters[0].data}"
            #       f"Graident: {grads[0]}")
        
        epoch_loss = running_loss / len(train_loader.dataset)   # Calculate the average loss per sample
        epoch_accuracy = correct / total                        # Calculate the accuracy
        
        # test_loss, test_accuracy = evaluate(model, test_loader, criterion) # Evaluate the model on the test set
        
        # Print metrics
        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, "
            #   f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}"
            )
        
        # TODO: create custom scheduler
        # scheduler.step(test_loss)                   # Step the scheduler with the test loss
        
        # early_stopping(test_loss, model, save_dir)  # Call the early stopping function with the test loss and model
        
        if early_stopping.early_stop:   # Check if early stopping condition is met
            break 
            
    get_classification_report(model, test_loader) # After training, get classification report

def evaluate(model, test_loader, criterion):
    """
    Evaluate the performance of a model on the test data.

    Args:
        model: The model to evaluate.
        test_loader: The test data loader.
        criterion: The loss function.

    Returns:
        tuple: A tuple containing the test loss and accuracy.
    """
    model.eval()    # Set the model to evaluation mode
    test_loss = 0.0 # Initialize the test loss
    correct = 0     # Initialize the number of correct predictions
    total = 0       # Initialize the total number of predictions

    # for inputs, labels in tqdm(test_loader, desc="Evaluating"): # Iterate over the test data
    for data in test_loader: # Iterate over the training data
        inputs, labels = data[:, :-1], data[:, -1]
        outputs = model.forward(inputs)                      # Forward pass
        loss = criterion(outputs, labels)                    # Calculate the loss
        test_loss += loss * inputs.shape[0]                   # Accumulate the test loss

        predicted = np.round(sigmoid(outputs))                # Get the predicted labels
        total += labels.shape[0]                              # Increment the total count
        correct += np.sum(predicted == labels)                # Increment the correct count

    accuracy = correct / total              # Calculate the accuracy
    test_loss /= len(test_loader.dataset)   # Calculate the average test loss per sample

    return test_loss, accuracy

def get_classification_report(model, test_loader):
    """
    Generate a classification report for a given model using the test data.

    Args:
        model: The trained model.
        test_loader: The data loader for the test data.

    Returns:
        dict: A dictionary containing the classification report metrics.
    """
    y_true = []     # Initialize the list to store true labels
    y_pred = []     # Initialize the list to store predicted labels
    
    for data in test_loader: # Iterate over the training data
        inputs, labels = data[:, :-1], data[:, -1]
        outputs = model.forward(inputs)                                 # Forward pass
        predicted = np.round(sigmoid(outputs))                           # Get the predicted labels
        y_true.extend(labels)                                            # Append true labels to the list
        y_pred.extend(predicted)                                         # Append predicted labels to the list
            
    y_true = np.array(y_true)   # Convert true labels to numpy array
    y_pred = np.array(y_pred)   # Convert predicted labels to numpy array
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    return classification_report(y_true, y_pred, output_dict=True)
