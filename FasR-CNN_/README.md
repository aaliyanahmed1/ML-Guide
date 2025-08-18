# Faster R-CNN Object Detection

Simple implementation of Faster R-CNN with multiple backbone options for object detection tasks.

## Available Models

1. **ResNet50 Backbone** (`fastrcnn_resnet50.py`)
   - Best balance of speed and accuracy
   - Recommended for most use cases

2. **ResNet101 Backbone** (`fastrcnn_resnet101.py`)
   - Higher accuracy
   - Better for complex scenes

3. **MobileNetV3 Backbone** (`fastrcnn_mobile.py`)
   - Fast inference
   - Suitable for mobile/edge devices

## Quick Start

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run inference:
```python
# Using ResNet50
from fastrcnn_resnet50 import FasterRCNN_ResNet50
detector = FasterRCNN_ResNet50()
detector.detect("image.jpg")
```

## Training Custom Dataset

```python
# Update paths in train_fastrcnn.py
config = {
    'data_dir': 'path/to/images',
    'train_annotations': 'path/to/train.json',
    'model_type': 'resnet50'  # or 'resnet101' or 'mobile'
}

# Start training
python train_fastrcnn.py
```

## Features

- Multiple backbone options
- COCO format dataset support
- Automatic GPU/CPU detection
- Progress visualization
- Model checkpointing
- Easy-to-use interface

## Requirements

- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- PIL, matplotlib, tqdm

## Model Comparison

| Model | Speed | Accuracy | Memory |
|-------|--------|----------|---------|
| ResNet50 | ★★★☆☆ | ★★★★☆ | ★★★☆☆ |
| ResNet101 | ★★☆☆☆ | ★★★★★ | ★★★★☆ |
| MobileNet | ★★★★★ | ★★★☆☆ | ★★☆☆☆ |
