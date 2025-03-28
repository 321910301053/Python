import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from models.model import create_model, train_model, evaluate_model, save_model
from preprocessing.data_preprocessing import load_data, preprocess_data
from utils.helper_functions import plot_metrics
import pickle

# Load data
X, Y = load_data()
X, Y = preprocess_data(X, Y)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# One-hot encode labels
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# Create and train model
model = create_model()
history = train_model(model, X_train, y_train, X_val, y_val)

# Save the trained model
save_model(model)

# Evaluate model
evaluate_model(model, X_test, y_test)

# Plot accuracy and loss
plot_metrics(history)
