"""RF-DETR Large (Roboflow Inference) - minimal example with visualization.

Inputs:
- image_path (str): local image file path

Outputs:
- Visualization with bounding boxes, class ids, and confidence scores

`torch` is imported to print available device; Roboflow SDK inference itself
does not require transferring tensors to that device in this script.
"""

import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from inference import get_model

# Device status (informational)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load RF-DETR Large model
model = get_model("rfdetr-large/1")

# Input image
image_path = "test_image.jpg"
image = Image.open(image_path).convert("RGB")

# Inference (SDK handles preprocessing)
outputs = model.infer(image_path, confidence=0.5)

# Normalize predictions to a list
preds = outputs[0].get("predictions", outputs[0]) if isinstance(outputs, list) else outputs
if isinstance(preds, dict):
    preds = [preds]

# Convert to corner-format boxes with labels and scores
boxes, labels, scores = [], [], []
for p in preds:
    conf = float(p.get("confidence", p.get("score", 0.0)))
    x_c = float(p.get("x", 0.0))
    y_c = float(p.get("y", 0.0))
    w = float(p.get("width", p.get("w", 0.0)))
    h = float(p.get("height", p.get("h", 0.0)))
    x1, y1 = x_c - w / 2.0, y_c - h / 2.0
    x2, y2 = x_c + w / 2.0, y_c + h / 2.0
    boxes.append([x1, y1, x2, y2])
    labels.append(int(p.get("class_id", 0)))
    scores.append(conf)

# Visualize
plt.figure(figsize=(12, 8))
plt.imshow(image)
for (x1, y1, x2, y2), label, score in zip(boxes, labels, scores):
    rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    plt.text(x1, y1 - 5, f"Class {label}: {score:.2f}", color='red', fontsize=10, backgroundcolor='white')

plt.axis('off')
plt.show()
