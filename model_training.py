import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ml_utils import NeuralNetwork, AdamOptimizer
from sklearn.metrics import classification_report

# Function to load and preprocess Pima Indians Diabetes dataset
def load_pima_diabetes_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('y', axis=1).to_numpy()
    y = df['y'].to_numpy().reshape(-1, 1)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X, y, X_train, X_test, y_train, y_test

# Function to load and preprocess Wisconsin Breast Prognostic dataset
def load_breast_prognostic_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('diagnosis', axis=1).to_numpy()
    y = df['diagnosis'].to_numpy().reshape(-1, 1)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X, y, X_train, X_test, y_train, y_test

# Function to train and evaluate the Neural Network
def train_and_evaluate_model(X_train, X_test, y_train, y_test, 
                             layers, output_size, lr, dropout, reg_lambda, hidden_activation='relu', output_activation='softmax',
                             epochs=100, batch_size=32):
    input_size = X_train.shape[1]
    
    activations = [hidden_activation] * len(layers) + [output_activation]

    # Initialize Neural Network
    nn = NeuralNetwork([input_size] + layers + [output_size], dropout_rate=dropout, reg_lambda=reg_lambda, activations=activations)
    optimizer = AdamOptimizer(learning_rate=lr)

    nn.train(X_train, y_train, X_test, y_test, optimizer=optimizer, epochs=epochs, batch_size=batch_size, early_stopping_threshold=10)

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    import random
    np.random.seed(41)
    random.seed(41)
    
    # Define parameter grid and tuning ranges
    dropout = 0.1
    reg_lambda=  0.0
    lr = 0.0001
    layers = [100, 50, 25] 
    output_size = 1

    # Train and evaluate on Pima Indians Diabetes dataset
    print("\n--- Training on Pima Indians Diabetes Dataset ---")
    
    X, y, X_train, X_test, y_train, y_test = load_pima_diabetes_data("data/pima-indians-diabetes_prepared.csv")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, layers,output_size, lr, dropout, reg_lambda, hidden_activation='tanh', output_activation='softmax',
                             epochs=1000, batch_size=32)

    # Train and evaluate on Wisconsin Breast Prognostic dataset
    print("\n--- Training on Wisconsin Breast Prognostic Dataset ---")
    
    X, y, X_train, X_test, y_train, y_test = load_breast_prognostic_data("data/Wisconsin_breast_prognostic.csv")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, layers,output_size, lr, dropout, reg_lambda, hidden_activation='relu', output_activation='softmax',
                             epochs=1000, batch_size=32)

if __name__ == "__main__":
    main()
