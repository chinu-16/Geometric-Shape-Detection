import argparse
import numpy as np
from utils import load_trained_model, preprocess_image

# Define shape classes
shape_classes = ["circle", "square", "triangle", "pentagon", "ellipse"]

def predict(image_path):
    """Loads the model, preprocesses an image, and predicts the shape."""
    print("Loading model...")
    model = load_trained_model()

    print(f"Processing image: {image_path}")
    image = preprocess_image(image_path)

    print("Making prediction...")
    predictions = model.predict(image)
    
    # Get class label
    predicted_class = np.argmax(predictions)
    predicted_shape = shape_classes[predicted_class]
    
    print(f"Predicted Shape: {predicted_shape}")
    return predicted_shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict shape from an image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    args = parser.parse_args()

    predict(args.image)
