import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import zipfile
import tkinter as tk
from tkinter import filedialog
import kagglehub

# Configure Kaggle API keys
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

# Function to load the Iris dataset from Kaggle
def load_kaggle_iris():
    print("Downloading the Iris dataset from Kaggle...")
    path = kagglehub.dataset_download("uciml/iris")
    print("Path to dataset files:", path)
    csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError("No CSV files found in the Kaggle dataset.")
    return os.path.join(path, csv_files[0])

# Function to load and preprocess data
def load_data(option):
    if option == "1":
        file_path = browse_file()
        if file_path.endswith('.zip'):
            file_path = handle_zipped_dataset(file_path)
        data = pd.read_csv(file_path)
    elif option == "2":
        file_path = load_kaggle_iris()
        data = pd.read_csv(file_path)
    else:
        raise ValueError("Invalid option selected.")

    print("Dataset loaded successfully!")
    print("Columns:", list(data.columns))

    # Automatically detect the target column if "Species" exists
    if "Species" in data.columns:
        target_col = "Species"
        print(f"Automatically detected target column: {target_col}")
    else:
        target_col = input("Enter the target column name: ").strip()

    X = data.drop(columns=[target_col])

    # Data quality checks
    print("Performing data quality checks...")
    if X.isnull().sum().sum() > 0:
        print("Missing values detected. Filling with column means...")
        X.fillna(X.mean(), inplace=True)

    if X.dtypes.any() == 'object':
        print("Non-numeric data detected. Encoding categorical features...")
        X = pd.get_dummies(X)

    y = data[target_col]
    
    # Encode target if categorical
    if y.dtype == 'O' or len(set(y)) < len(y):
        y = to_categorical(pd.factorize(y)[0])
    return train_test_split(X.values, y, test_size=0.2, random_state=42)

# Train neural network
def train_neural_network(x_train, y_train, x_test, y_test, input_shape, num_classes, epochs):
    model = Sequential([
        Input(shape=input_shape),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )
    history = model.fit(
        x_train, y_train, 
        validation_data=(x_test, y_test), 
        epochs=epochs, 
        batch_size=16, 
        callbacks=[early_stopping]
    )
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
    print("Choose an option to load your dataset:")
    print("1. Upload your own dataset")
    print("2. Use the Kaggle Iris dataset")
    option = input("Enter your choice (1 or 2): ").strip()

    x_train, x_test, y_train, y_test = load_data(option)
    input_shape = x_train.shape[1:]
    num_classes = y_train.shape[1]

    epochs = int(input("Enter the number of epochs for training: ").strip())

    model, history = train_neural_network(x_train, y_train, x_test, y_test, input_shape, num_classes, epochs)

    # Plot training results
    plt.figure(figsize=(12, 6))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

    evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
