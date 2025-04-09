import cv2
import numpy as np
from utils import load_trained_model, preprocess_image
import tensorflow as tf

# Load trained model
model = load_trained_model()

# Class labels
shape_classes = ["Circle", "Square", "Triangle", "Pentagon", "Ellipse"]

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess frame
    processed_frame = cv2.resize(frame, (256, 256))
    processed_frame = processed_frame / 255.0
    processed_frame = np.expand_dims(processed_frame, axis=0)

    # Make prediction
    predictions = model.predict(processed_frame)
    shape_idx = np.argmax(predictions[0])
    predicted_shape = shape_classes[shape_idx]

    # Draw prediction text on frame
    cv2.putText(frame, f"Detected: {predicted_shape}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Real-Time Shape Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
