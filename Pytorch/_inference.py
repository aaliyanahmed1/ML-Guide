import argparse
import os
from typing import Dict, List

import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F


# -----------------------------------------------------------------------------
# Minimal COCO label list (indices match torchvision COCO-trained models)
# -----------------------------------------------------------------------------
COCO_CLASSES: List[str] = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'N/A', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# -----------------------------------------------------------------------------
# Fixed configuration: model and confidence threshold
# -----------------------------------------------------------------------------
FIXED_MODEL_NAME = 'fasterrcnn'
FIXED_THRESHOLD = 0.5


def load_model() -> torch.nn.Module:
    """Load a COCO-pretrained torchvision object detection model.

    The model is hardcoded to Faster R-CNN ResNet50 FPN as requested.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.eval()
    return model


def run_inference(image_path: str, output_path: str) -> str:
    """Run object detection on a single image and save a visualization.

    Args:
        image_path: Path to input image file.
        output_path: Where to save the output image with drawn boxes.

    Returns:
        The path to the saved result image.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model and move to device
    model = load_model().to(device)

    # Load and transform image to tensor expected by torchvision models
    image = Image.open(image_path).convert('RGB')
    image_tensor = F.to_tensor(image).to(device).unsqueeze(0)  # shape: [1, 3, H, W]

    # Forward pass with no gradients
    with torch.no_grad():
        outputs: List[Dict[str, torch.Tensor]] = model(image_tensor)

    predictions = outputs[0]

    # Filter by fixed confidence threshold
    keep = predictions['scores'] >= FIXED_THRESHOLD
    boxes = predictions['boxes'][keep].cpu().tolist()
    labels = predictions['labels'][keep].cpu().tolist()
    scores = predictions['scores'][keep].cpu().tolist()

    # Draw boxes on a copy of the original image
    drawer = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        drawer.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)

        class_name = (
            COCO_CLASSES[label] if 0 <= label < len(COCO_CLASSES) else f"class_{label}"
        )
        caption = f"{class_name}: {score:.2f}"

        # Text background for readability
        text_w, text_h = drawer.textbbox((0, 0), caption, font=font)[2:]
        drawer.rectangle(
            [(x1, y1 - text_h - 2), (x1 + text_w + 2, y1)], fill=(255, 0, 0)
        )
        drawer.text((x1 + 1, y1 - text_h - 1), caption, fill=(255, 255, 255), font=font)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    image.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments: only the image path and optional output path."""
    parser = argparse.ArgumentParser(
        description="Minimal torchvision COCO inference (hardcoded model)"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--output", default="output.jpg", help="Path to save visualization"
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint: run inference with hardcoded model and threshold."""
    args = parse_args()
    result_path = run_inference(image_path=args.image, output_path=args.output)
    print(f"Model: {FIXED_MODEL_NAME}, Threshold: {FIXED_THRESHOLD}")
    print(f"Saved result to: {result_path}")


if __name__ == "__main__":
    main()
