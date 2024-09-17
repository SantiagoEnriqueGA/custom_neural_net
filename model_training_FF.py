
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ml_utils

file_orig = "data/pima-indians-diabetes_prepared.csv"
df = pd.read_csv(file_orig)
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = pd.concat([X_train, y_train], axis=1).to_numpy()
test_data = pd.concat([X_test, y_test], axis=1).to_numpy()

train_loader = ml_utils.DataLoader(train_data, batch_size=36)
test_loader = ml_utils.DataLoader(test_data, batch_size=36)

# Train the model
model = ml_utils.ConfigurableNN(X_train.shape[1], hidden_dims=[128,64,16], dropout_rate=.5)
ml_utils.train(model, train_loader, test_loader, epochs=10, patience=5, save_dir='models/ff_model.pth')

