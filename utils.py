import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore

# Function to load trained model
def load_trained_model(model_path="models/shape_classification_model.h5"):
    """Loads the trained model from the given file path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: Model file '{model_path}' not found.")
    
    return load_model(model_path)

# Function to preprocess an image for the model
def preprocess_image(image_path, target_size=(256, 256)):
    """Loads an image, resizes it, and normalizes pixel values."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Error: Image file '{image_path}' not found.")
    
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values
    return np.expand_dims(image, axis=0)  # Add batch dimension
