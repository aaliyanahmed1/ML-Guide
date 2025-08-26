# TensorFlow Object Detection & Examples

This folder contains minimal, well-documented examples of TensorFlow for object detection and core functionality. Each file demonstrates specific TensorFlow capabilities with clean, CI-compliant code.

## Main Object Detection Scripts

- **`_inference.py`** - Minimal TensorFlow object detection inference
  - Uses EfficientDet-D0 from TensorFlow Hub for fast detection
  - CLI: `--image` (required), `--output` (optional, default: tf_output.jpg)
  - Hardcoded confidence threshold: 0.5
  - Saves annotated image with bounding boxes and labels

- **`training.py`** - Minimal TensorFlow object detection training
  - Uses MobileNetV2 backbone with transfer learning on COCO dataset
  - Hardcoded hyperparameters: epochs=5, batch_size=32, lr=0.001
  - Automatically downloads COCO subset (1000 train, 100 val samples)
  - Saves trained model to `tf_object_detection_model.h5`

## Library Introduction Scripts

- **`tensorflow_core.py`** - Core TensorFlow operations and model building
- **`tf_keras.py`** - High-level Keras API for neural networks
- **`tf_data.py`** - Efficient data loading and preprocessing pipelines
- **`tf_image.py`** - Image processing utilities (resize, flip, brightness)
- **`tf_audio.py`** - Audio processing functions and signal generation
- **`tf_text.py`** - Natural language processing tools and tokenization
- **`tensorflow_hub_.py`** - Pre-trained models from TensorFlow Hub

## Clone & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aaliyanahmed1/ML-Guide
   cd ML-Guide/tensorflow
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run object detection inference:**
   ```bash
   python _inference.py --image path/to/your/image.jpg --output result.jpg
   ```

4. **Run object detection training:**
   ```bash
   python training.py
   ```

5. **Run library examples:**
   ```bash
   python tensorflow_core.py
   python tf_keras.py
   python tf_data.py
   python tf_image.py
   python tf_audio.py
   python tf_text.py
   python tensorflow_hub_.py
   ```

## Key Features

- **Minimal Code**: Clean, readable implementations with detailed comments
- **CI Compliant**: All files follow MegaLinter standards for Python
- **Hardcoded Config**: No complex CLI arguments - just edit constants in code
- **Fast Models**: EfficientDet for inference, MobileNetV2 for training
- **COCO Format**: Standard object detection dataset format
- **Transfer Learning**: Pre-trained models with custom heads

Each script is self-contained and includes comprehensive comments explaining the workflow and TensorFlow concepts.


