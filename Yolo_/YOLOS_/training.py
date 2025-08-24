"""
Road Sign Detection Model Training using YOLOv12n
--------------------------------------------------
This script fine-tunes a pre-trained YOLOv12n model on a custom road sign detection dataset.
It includes configuration for training, validation monitoring, and post-training evaluation.
"""

from ultralytics import YOLO
import os

# Define paths and hyperparameters
MODEL_WEIGHTS = 'yolo12n.pt'  # Pre-trained YOLOv12n weights
DATA_YAML = 'processed_yolo_dataset/data.yaml'  # Dataset configuration in YOLO format
IMG_SIZE = 640
EPOCHS = 30
BATCH_SIZE = 16
LR = 0.001
OPTIMIZER = 'SGD'  # Options: ['SGD', 'Adam', 'AdamW']
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
NUM_WORKERS = 2
RESULTS_DIR = 'runs/train/road_sign_yolo12n'

# Create the YOLO model instance
model = YOLO(MODEL_WEIGHTS)

# -------------------------------
# Training the model
# -------------------------------
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    lr0=LR,
    optimizer=OPTIMIZER,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    warmup_epochs=WARMUP_EPOCHS,
    workers=NUM_WORKERS,
    project='runs/train',
    name='road_sign_yolo12n',
    exist_ok=True  # Overwrite existing run if exists
)

# -------------------------------
# Validation on validation set
# -------------------------------
# This runs automatically after training, but can be re-run for analysis
val_metrics = model.val(
    data=DATA_YAML,
    imgsz=IMG_SIZE,
    split='val'  # Use validation set for evaluation
)

# Print key metrics
print("\nValidation Metrics:")
print(val_metrics)  # Contains precision, recall, mAP50, mAP50-95, etc.

# -------------------------------
# Inference on a test image (sample test)
# -------------------------------
TEST_IMAGE = 'test_images/sample_road_sign.jpg'  # Replace with your actual test image
results = model.predict(
    source=TEST_IMAGE,
    imgsz=IMG_SIZE,
    conf=0.25,
    save=True  # Saves output image with boxes
)

print(f"\nInference completed on: {TEST_IMAGE}")
print("Detected Objects:", results[0].names)
