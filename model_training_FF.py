import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_utils import NeuralNetwork, AdamOptimizer

# file_orig = "data/pima-indians-diabetes_prepared.csv"
# df = pd.read_csv(file_orig)
# X = df.drop('y', axis=1).to_numpy()
# y = df['y'].to_numpy().reshape(-1, 1)

file_orig = "data/Wisconsin_breast_prognostic.csv"
df = pd.read_csv(file_orig)
X = df.drop('diagnosis', axis=1).to_numpy()
y = df['diagnosis'].to_numpy().reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = X_train.shape[1]
hidden_layers = [100, 50, 25]
output_size = 1

nn = NeuralNetwork([input_size] + hidden_layers + [output_size], dropout_rate=0.5, reg_lambda=0.0)
optimizer = AdamOptimizer(learning_rate=0.0001)

nn.train(X_train, y_train, X_test, y_test, optimizer, epochs=100, batch_size=32, early_stopping_threshold=10)

# 4. Evaluate the Model
test_accuracy, y_pred = nn.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

