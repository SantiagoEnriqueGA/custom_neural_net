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
    
    return X_train, X_test, y_train, y_test

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
    
    return X_train, X_test, y_train, y_test

# Function to train and evaluate the Neural Network
def train_and_evaluate_model(X_train, X_test, y_train, y_test, param_grid, num_layers_range, layer_size_range, lr_range, epochs=100, batch_size=32):
    input_size = X_train.shape[1]
    output_size = 1

    # Initialize Neural Network
    nn = NeuralNetwork([input_size] + [100, 50, 25] + [output_size], dropout_rate=0.5, reg_lambda=0.0)
    optimizer = AdamOptimizer(learning_rate=0.0001)

    # Hyperparameter tuning with Adam optimizer
    best_params, best_accuracy = nn.tune_hyperparameters(
        param_grid,
        num_layers_range,
        layer_size_range,
        X_train,
        y_train,
        X_test,
        y_test,
        optimizer_type='Adam',
        lr_range=lr_range,
        epochs=epochs,
        batch_size=batch_size
    )

    print(f"Best parameters: {best_params} with accuracy: {best_accuracy:.4f}")

    # Train the final model with best parameters
    nn = NeuralNetwork([input_size] + [best_params['layer_size']] * best_params['num_layers'] + [output_size], 
                       dropout_rate=best_params['dropout_rate'], 
                       reg_lambda=best_params['reg_lambda'])
    nn.train(X_train, y_train, X_test, y_test, optimizer, epochs=epochs, batch_size=batch_size)

    # Evaluate the Model
    test_accuracy, y_pred = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

def main():
    # Define parameter grid and tuning ranges
    param_grid = {
        'dropout_rate': [0.1, 0.2],
        'reg_lambda': [0.0, 0.01]
    }
    num_layers_range = (2, 5, 1)  # min, max, step
    layer_size_range = (25, 100, 25)  # min, max, step
    lr_range = (1e-5, 0.01, 5)  # (min_lr, max_lr, num_steps)

    # Train and evaluate on Pima Indians Diabetes dataset
    print("\n--- Training on Pima Indians Diabetes Dataset ---")
    X_train, X_test, y_train, y_test = load_pima_diabetes_data("data/pima-indians-diabetes_prepared.csv")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, param_grid, num_layers_range, layer_size_range, lr_range)

    # Train and evaluate on Wisconsin Breast Prognostic dataset
    print("\n--- Training on Wisconsin Breast Prognostic Dataset ---")
    X_train, X_test, y_train, y_test = load_breast_prognostic_data("data/Wisconsin_breast_prognostic.csv")
    train_and_evaluate_model(X_train, X_test, y_train, y_test, param_grid, num_layers_range, layer_size_range, lr_range)

if __name__ == "__main__":
    main()
