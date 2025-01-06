#Iris Classifier 


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import tkinter as tk
from tkinter import filedialog

# Configure API keys
os.environ["KAGGLE_USERNAME"] = "kylebirch"
os.environ["KAGGLE_KEY"] = "c1a2ecd4c8651eb69498103dc7fd144d"

# Function to handle zipped datasets
def handle_zipped_dataset(zip_path):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"The file '{zip_path}' does not exist.")
    extracted_folder = "extracted_data"
    os.makedirs(extracted_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)
    csv_files = [f for f in os.listdir(extracted_folder) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the extracted dataset.")
    return os.path.join(extracted_folder, csv_files[0])

# Function to browse for a file
def browse_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv"), ("Zip files", "*.zip")]
    )
    if not file_path:
        raise ValueError("No file selected.")
    return file_path

# Load Iris dataset
def load_iris_data():
    data = load_iris()
    X, y = data.data, data.target
    y = to_categorical(y, num_classes=3)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test

# Train neural network
def train_neural_network(x_train, y_train, x_test, y_test, input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)
    return model, history

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Main function
def main():
    x_train, y_train, x_test, y_test = load_iris_data()
    model, history = train_neural_network(x_train, y_train, x_test, y_test, input_shape=(4,), num_classes=3)
    
    # Plot training results
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
