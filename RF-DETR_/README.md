# RF-DETR Object Detection

Simple implementation of RF-DETR (Recursive Feature Detection Transformer) for object detection tasks.

## Models Available

1. **RF-DETR Nano** (`rfdetr-nano.py`)
   - Lightweight version for edge devices
   - Fast inference speed

2. **RF-DETR Small** (`rfdetr-small.py`)
   - Balanced performance
   - Good for most applications

3. **RF-DETR Base** (`rfdetrbase.py`)
   - Standard model
   - Best accuracy/speed trade-off

4. **RF-DETR Large** (`rfdetrlarge.py`)
   - Highest accuracy
   - For complex detection tasks

## Quick Start

1. **Install Requirements**
```bash
pip install -r requirements.txt
```

2. **Training**
```bash
python train_rfdetr.py
```

3. **Inference**
```python
# Choose a model version (nano/small/base/large)
from rfdetrbase import detect_objects

# Run detection
results = detect_objects("image.jpg")

## Usage

### Training
```python
# Configure training
config = {
    'data_root': 'path/to/images',
    'train_annotations': 'path/to/train.json',
    'val_annotations': 'path/to/val.json',
    'num_epochs': 20,
    'batch_size': 4
}

# Start training
python train_rfdetr.py

## Model Selection Guide

- **Nano**: Mobile devices, edge computing
- **Small**: Desktop applications, real-time needs
- **Base**: General purpose, good balance
- **Large**: High accuracy requirements

## Requirements

- PyTorch >= 2.0.0
- Transformers >= 4.31.0
- Torchvision >= 0.15.0

## Requirements

- PyTorch >= 2.0.0
- Torchvision >= 0.15.0
- Transformers >= 4.31.0
- TIMM >= 0.9.2
- Additional dependencies in requirements.txt

## Model Features

RF-DETR improves upon traditional DETR by:
- Using recursive feature refinement
- Improved attention mechanisms
- Better convergence speed
- Enhanced small object detection

## Example Usage

1. Training on custom dataset:
```python
# Update dataset paths
config = {
    'data_root': 'path/to/images',
    'train_annotations': 'path/to/train.json',
    'val_annotations': 'path/to/val.json'
}
```

2. Running inference:
```python
# Place images in test_images directory
# Results will be saved in predictions directory
python inference_rfdetr.py
```

## Output Format

Detection results include:
- Bounding box coordinates
- Class predictions
- Confidence scores
- Visualizations with labeled boxes