"""Minimal RF-DETR inference using Roboflow Inference SDK with timing and output.

- Loads model via `inference.get_model` (Roboflow SDK)
- Reads a local image path, runs inference, prints timings
- Converts predictions to tensors: boxes [N,4], labels [N], scores [N]
- Saves an annotated image with bounding boxes and scores
"""

import time
from typing import Any, Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont
from inference import get_model  # Roboflow Inference SDK

_LOAD_SOURCE = "roboflow-inference"


# -----------------------------------------------------------------------------
# USER SETTINGS (edit these paths/values)
# -----------------------------------------------------------------------------
IMAGE_PATH = "path/to/your/image.jpg"  # e.g., "./example.jpg"
MODEL_ID = "rfdetr-base/1"  # Roboflow RF-DETR base model id
CONFIDENCE_THRESHOLD = 0.5  # keep detections with score >= threshold
OUTPUT_IMAGE = "rfdetr_result.jpg"  # annotated output file path


def load_model(model_id: str):
    """Load RF-DETR via Roboflow Inference SDK and report timing."""
    t0 = time.perf_counter()
    model = get_model(model_id)
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Load] Source={_LOAD_SOURCE}, Model={model_id}, Time={load_ms:.2f} ms")
    return model


def read_image(image_path: str) -> Image.Image:
    """Open image with PIL, report timing and size, and return the image."""
    t0 = time.perf_counter()
    image = Image.open(image_path).convert("RGB")
    prep_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Read] Image={image_path}, Size={image.size}, Time={prep_ms:.2f} ms")
    return image


def run_inference(model: Any, image_path: str) -> Dict[str, Any]:
    """Call Roboflow model.infer() on a local image and return raw response."""
    t0 = time.perf_counter()
    outputs = model.infer(image_path, confidence=CONFIDENCE_THRESHOLD)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Inference] Time={infer_ms:.2f} ms")
    return outputs


def to_tensors(outputs: Any, threshold: float) -> Dict[str, torch.Tensor]:
    """Convert Roboflow predictions to tensors: boxes [N,4], labels [N], scores [N]."""
    t0 = time.perf_counter()

    # Normalize output structure to a list of predictions dicts
    if isinstance(outputs, list):
        data = outputs[0]
    else:
        data = outputs

    preds = data.get("predictions", data)
    if isinstance(preds, dict):
        preds = [preds]

    boxes: List[List[float]] = []
    labels: List[int] = []
    scores: List[float] = []

    for p in preds:
        conf = float(p.get("confidence", p.get("score", 0.0)))
        if conf < threshold:
            continue
        # Roboflow returns x,y,w,h where (x,y) is the box center
        x_c = float(p.get("x", 0.0))
        y_c = float(p.get("y", 0.0))
        w = float(p.get("width", p.get("w", 0.0)))
        h = float(p.get("height", p.get("h", 0.0)))
        x1 = x_c - w / 2.0
        y1 = y_c - h / 2.0
        x2 = x_c + w / 2.0
        y2 = y_c + h / 2.0
        boxes.append([x1, y1, x2, y2])
        labels.append(int(p.get("class_id", 0)))
        scores.append(conf)

    result = {
        "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32) if scores else torch.zeros((0,), dtype=torch.float32),
    }

    post_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[Postprocess] Kept={len(boxes)}, Time={post_ms:.2f} ms")
    return result


def annotate_and_save(image: Image.Image, detections, output_path: str) -> None:
    """Draw bounding boxes with labels/scores on an image and save it."""
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    boxes = detections["boxes"].cpu().tolist()
    labels = detections["labels"].cpu().tolist()
    scores = detections["scores"].cpu().tolist()

    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
        caption = f"id={label} | conf={score:.2f}"
        tw, th = draw.textbbox((0, 0), caption, font=font)[2:]
        draw.rectangle([(x1, y1 - th - 2), (x1 + tw + 2, y1)], fill=(255, 0, 0))
        draw.text((x1 + 1, y1 - th - 1), caption, fill=(255, 255, 255), font=font)

    image.save(output_path)
    print(f"[Save] Annotated image saved to: {output_path}")


def main() -> None:
    """End-to-end minimal inference with timing and prints using Roboflow SDK."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using: {device}")

    total_t0 = time.perf_counter()

    # 1) Load model
    model = load_model(MODEL_ID)

    # 2) Read image
    pil_image = read_image(IMAGE_PATH)

    # 3) Inference (Roboflow SDK handles preprocessing/resize internally)
    outputs = run_inference(model, IMAGE_PATH)

    # 4) Convert predictions to tensors and filter
    detections = to_tensors(outputs, threshold=CONFIDENCE_THRESHOLD)

    # 5) Print detections in a readable form
    boxes = detections["boxes"].cpu().tolist()
    labels = detections["labels"].cpu().tolist()
    scores = detections["scores"].cpu().tolist()

    print("[Detections] count=", len(boxes))
    for i, (box, lab, score) in enumerate(zip(boxes, labels, scores), 1):
        x1, y1, x2, y2 = [round(v, 2) for v in box]
        print(f"  #{i:02d} box=({x1}, {y1}, {x2}, {y2}) label_id={lab} score={score:.3f}")

    # 6) Save annotated output image
    annotate_and_save(pil_image.copy(), detections, OUTPUT_IMAGE)

    total_ms = (time.perf_counter() - total_t0) * 1000.0
    print(f"[Total] End-to-end time: {total_ms:.2f} ms")


if __name__ == "__main__":
    main()
