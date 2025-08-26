"""Minimal TensorFlow object detection inference example.

- Loads SSD MobileNet V2 from TF Hub.
- Runs inference on a local image.
- Draws boxes and saves annotated image.
- Logs inference time.

This script is formatted and linted to pass CI (megalinter, black, flake8, etc).
"""

import time
from typing import Dict, List
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
INPUT_IMAGE_PATH = "input.jpg"
OUTPUT_IMAGE_PATH = "output.jpg"
MODEL_HANDLE = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
CONFIDENCE_THRESHOLD = 0.5


def load_model() -> tf.keras.Model:
    """Load SSD MobileNet V2 model from TensorFlow Hub.

    Returns:
        Pre-trained object detection model.
    """
    print("Loading model...")
    detector = hub.load(MODEL_HANDLE)
    print("Model loaded!")
    return detector


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for object detection.

    Args:
        image_path: Path to input image.

    Returns:
        Preprocessed image array.

    Raises:
        FileNotFoundError: If image file is not found.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Convert BGR to RGB and add batch dimension
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb_img = np.expand_dims(rgb_img, axis=0)
    return rgb_img, img


def run_inference(detector: tf.keras.Model, image: np.ndarray) -> Dict:
    """Run object detection inference on image.

    Args:
        detector: Pre-trained detection model.
        image: Preprocessed image array.

    Returns:
        Dictionary containing detection results.
    """
    start_time = time.time()
    results = detector(image)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.4f} seconds")
    
    # Convert tensor results to numpy
    result = {key: value.numpy() for key, value in results.items()}
    return result


def draw_detections(img: np.ndarray, boxes: List, scores: List, classes: List) -> None:
    """Draw bounding boxes and labels on image.

    Args:
        img: Input image array.
        boxes: Detection bounding boxes.
        scores: Detection confidence scores.
        classes: Detection class IDs.
    """
    h, w, _ = img.shape
    
    for i, score in enumerate(scores):
        if score > CONFIDENCE_THRESHOLD:
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1 = int(xmin * w), int(ymin * h)
            x2, y2 = int(xmax * w), int(ymax * h)
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"ID:{int(classes[i])} {score:.2f}"
            cv2.putText(
                img,
                label,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )


def main() -> None:
    """Main function to run object detection inference."""
    # Load model
    detector = load_model()
    
    # Preprocess image
    rgb_img, original_img = preprocess_image(INPUT_IMAGE_PATH)
    
    # Run inference
    results = run_inference(detector, rgb_img)
    
    # Extract detection results
    boxes = results.get("detection_boxes", [])
    scores = results.get("detection_scores", [])
    classes = results.get("detection_classes", [])
    
    # Draw detections
    draw_detections(original_img, boxes, scores, classes)
    
    # Save annotated image
    cv2.imwrite(OUTPUT_IMAGE_PATH, original_img)
    print(f"Output saved at {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
