"""Minimal TensorFlow object detection inference script with fixed model.

This script uses TensorFlow Hub's EfficientDet model for fast object detection.
It accepts an image path via CLI, performs inference with a fixed confidence
threshold, and saves an annotated output image with bounding boxes.
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageDraw, ImageFont


# -----------------------------------------------------------------------------
# Fixed configuration: model and confidence threshold
# -----------------------------------------------------------------------------
FIXED_MODEL_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"
FIXED_THRESHOLD = 0.5

# COCO class names (80 classes for object detection)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def load_model() -> tf.keras.Model:
    """Load EfficientDet model from TensorFlow Hub.

    Returns:
        A pre-trained EfficientDet model for object detection.
    """
    model = hub.load(FIXED_MODEL_URL)
    return model


def preprocess_image(image_path: str) -> Tuple[np.ndarray, Image.Image]:
    """Load and preprocess image for EfficientDet model.

    Args:
        image_path: Path to the input image.

    Returns:
        Tuple of (preprocessed_tensor, original_pil_image).
    """
    # Load image with PIL
    pil_image = Image.open(image_path).convert('RGB')
    
    # Convert to numpy array and normalize to [0, 1]
    image_np = np.array(pil_image) / 255.0
    
    # Convert to tensor and add batch dimension
    image_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, 0)
    
    return image_tensor, pil_image


def run_inference(image_path: str, output_path: str) -> str:
    """Run object detection on a single image and save visualization.

    Args:
        image_path: Path to input image file.
        output_path: Where to save the output image with drawn boxes.

    Returns:
        The path to the saved result image.
    """
    # Load model
    model = load_model()
    
    # Preprocess image
    image_tensor, pil_image = preprocess_image(image_path)
    
    # Run inference
    detections = model(image_tensor)
    
    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()  # [N, 4] in normalized coords
    scores = detections['detection_scores'][0].numpy()  # [N]
    classes = detections['detection_classes'][0].numpy().astype(int)  # [N]
    
    # Filter by confidence threshold
    keep_indices = scores >= FIXED_THRESHOLD
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    classes = classes[keep_indices]
    
    # Draw bounding boxes on image
    draw = ImageDraw.Draw(pil_image)
    img_width, img_height = pil_image.size
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    
    for box, score, class_id in zip(boxes, scores, classes):
        # Convert normalized coordinates to pixel coordinates
        y1, x1, y2, x2 = box
        x1 = int(x1 * img_width)
        y1 = int(y1 * img_height)
        x2 = int(x2 * img_width)
        y2 = int(y2 * img_height)
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
        
        # Get class name
        class_name = (
            COCO_CLASSES[class_id - 1] 
            if 1 <= class_id <= len(COCO_CLASSES) 
            else f"class_{class_id}"
        )
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw label background
        draw.rectangle(
            [(x1, y1 - text_height - 2), (x1 + text_width + 2, y1)],
            fill=(255, 0, 0)
        )
        draw.text((x1 + 1, y1 - text_height - 1), label, fill=(255, 255, 255), font=font)
    
    # Save result
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    pil_image.save(output_path)
    
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: image path and optional output path."""
    parser = argparse.ArgumentParser(
        description="Minimal TensorFlow object detection inference (EfficientDet)"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--output", default="tf_output.jpg", help="Path to save visualization"
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint: run inference with fixed model and threshold."""
    args = parse_args()
    result_path = run_inference(image_path=args.image, output_path=args.output)
    print(f"Model: EfficientDet-D0, Threshold: {FIXED_THRESHOLD}")
    print(f"Saved result to: {result_path}")


if __name__ == "__main__":
    main()
