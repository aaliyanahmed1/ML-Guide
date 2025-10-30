# Machine Learning Hands-On Guide

Hi, in this guide we are going to deeply learn about machine learning's fundamental concepts to core advanced frameworks, model selection, benchmarking, fine-tuning and custom training and so on. From prototyping to production level real-time deployments, from computer systems/servers to edge devices cross-platform deployments. This guide will cover all the topics that are essentially needed to become zero-to-hero in Machine learning. In this documentation we will be focusing mainly on the object detection branch that belongs to Deep learning.

AI is simply the ability to think and act like humans. There are its branches:
Machine Learning > Deep Learning > Computer Vision.

**Machine Learning**: A branch of AI in which machines learn from labeled data patterns instead of fixed rules-based systems.
Three main types of Machine Learning:

**1: Supervised Learning 2: Unsupervised Learning 3: Reinforcement Learning.**

**Deep Learning**: Its branch of machine learning that is based on the special type of neural networks that learns complex patterns from data. It mainly falls under supervised learning where models are trained on labeled datasets to learn the mapping between inputs and outputs.

One of the special branches of deep learning is Computer Vision that uses Convolutional Neural Networks to learn complex patterns from data and perform predictions in different environments efficiently in real-time deployments as well. In computer vision, images are processed to extract meaningful features, which can then be used for various tasks such as classification, segmentation, and one of the most important applications—object detection, where models not only recognize objects but also locate them within the image. In this guide we will deeply learn about object detection.

---

## Prerequisites

**What you need before starting:** Basic requirements to work with object detection models and training pipelines. Without these you might face issues during setup or training phase.

Before starting with this guide, you should have:
- Basic Python programming knowledge
- Understanding of basic math (linear algebra, calculus)
- Familiarity with command line operations
- Python 3.8+ installed
- GPU with CUDA support (recommended for training)
- At least 8GB RAM (16GB+ recommended for training)

---

## Environment Setup

**Setting up development environment:** Create isolated Python environment to avoid dependency conflicts and install all necessary packages for object detection workflow.

First, set up your Python environment:

**File: `setup_environment.sh`**
```bash
# Create virtual environment to isolate dependencies
python -m venv ml_env

# Activate environment
# Windows:
ml_env\Scripts\activate
# Linux/Mac:
source ml_env/bin/activate

# Install PyTorch with CUDA support for GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install TensorFlow for alternative framework option
pip install tensorflow

# Install OpenCV for image processing operations
pip install opencv-python

# Install numerical and visualization libraries
pip install numpy pandas matplotlib

# Install ONNX for model export and deployment
pip install onnx onnxruntime

# Install MLflow for experiment tracking
pip install mlflow
```

---

## Object Detection

**Understanding object detection:** Object detection is a specific branch of computer vision that reads the images and detects the desired objects inside of the image along with their areas (x, y coordinates). Like if you input an image to the object detection model (YOLO, RF-DETR etc), it won't only tell that there is an object like cat/dog in the image, but also tells where they are by drawing bounding boxes on the areas and mentioning the class (name of object).

image input > model > inference > detection > output

![Object Detection](/images/onnxpy.png)

Like on this image we can see bounding boxes and the names of the objects like person and vehicle (bus) etc. That's the whole thing that is called object detection. It finds objects and labels objects within an image.

This was the verbal introduction of the object detection.
Now move towards all the technical details and requirements needed to develop, select, fine-tune and train the object detection model. From dataset collection to preprocessing.

---

## Object Detection Model Selection

**Choosing the right model:** When selecting the model for the object detection tasks first, we must thoroughly do research according to requirements and use case like where we want to deploy it and what we want to detect from it so this is the must check before moving forward towards data-preprocessing and fine-tuning. For example, you are developing a real-time security surveillance application that needs to detect events from the images under 50ms. Then model's speed shouldn't be compromised at any cost.

On the other hand if you are developing medical imaging system to analyze medical reports like X-rays, MRIs and CT-scans etc. to detect diseases then accuracy must be the top check. No any minor compromise on the accuracy no matter how long model takes to detect because here the use case isn't real-time so we can leverage the speed but not accuracy. But balance must be there in all cases.

---

### Model Performance Evaluation

**Evaluating model performance:** Now let's see what are the evaluation steps to compare the speeds and accuracies of different models and select the best one according to the requirements. These metrics help you decide which model fits your deployment scenario.

---

#### Speed Metrics

**Measuring inference speed:** These metrics tell you how fast your model can process images. Critical for real-time applications like surveillance or autonomous vehicles.

**Latency**: How long the model takes to process one image/frame (milliseconds). Lower latency means faster processing.

**FPS (Frames Per Second)**: How many frames model can process in 1 second. Valid for the real-time applications.

**Throughput**: Number of images processed per second when running in batch mode. Important for server deployments.

**Model Size**: Models' variants like nano, medium and large and the difference in speed and accuracy, the tradeoffs among them and what fits in the required use case.

---

#### Accuracy Metrics

**Measuring detection accuracy:** These metrics tell you how accurately your model detects and localizes objects. Critical for applications where missing an object or false detection can have serious consequences.

---

**1: Mean Average Precision (mAP)**

**What is mAP:** It evaluates how precisely model detects objects among crowded frames. It measures the accuracy of the model in identifying and localizing objects within an image. It combines precision (the proportion of correctly identified objects among all predicted objects) and recall (the proportion of correctly identified objects among all the actual objects). It gives a single score that shows how well model finds objects and how well it avoids false positives. A higher mAP means the model is more reliable and consistent. It helps to compare different models and select according to the use case and requirement.

**File: `metrics/map_calculation.py`**
```python
"""
Mean Average Precision (mAP) Calculation
-----------------------------------------
This script calculates mAP metric for object detection models.
mAP is the standard metric used to evaluate object detection models.
"""

import numpy as np


def calculate_mAP(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision for object detection.

    mAP measures the quality of object detection by combining precision
    and recall across all classes. Higher mAP = better model.

    Args:
        predictions: List of predicted boxes with confidence scores
                    Format: [{'box': [x1,y1,x2,y2], 'class': int, 'confidence': float}]
        ground_truths: List of actual boxes
                      Format: [{'box': [x1,y1,x2,y2], 'class': int}]
        iou_threshold: IoU threshold to consider prediction as correct (default: 0.5)
                      Lower threshold = more lenient, Higher = more strict

    Returns:
        mAP score (float between 0 and 1, higher is better)
    """
    aps = []  # Store Average Precision for each class
    num_classes = max([p['class'] for p in predictions]) + 1

    # Calculate AP for each class separately
    for class_id in range(num_classes):
        # Get predictions and ground truths for this specific class only
        class_preds = [p for p in predictions if p['class'] == class_id]
        class_gts = [gt for gt in ground_truths if gt['class'] == class_id]

        # Skip if no ground truths exist for this class
        if len(class_gts) == 0:
            continue

        # Sort predictions by confidence score (highest first)
        # This ensures we process most confident predictions first
        class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)

        # Calculate precision and recall at each threshold
        tp = 0  # True Positives: correct detections
        fp = 0  # False Positives: incorrect detections
        precisions = []
        recalls = []

        for pred in class_preds:
            # Check if this prediction matches any ground truth box
            matched = False
            for gt in class_gts:
                # Calculate IoU between predicted box and ground truth
                if calculate_iou(pred['box'], gt['box']) >= iou_threshold:
                    tp += 1  # Correct detection
                    matched = True
                    break

            if not matched:
                fp += 1  # Incorrect detection

            # Calculate precision: what % of predictions are correct
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            # Calculate recall: what % of ground truths are detected
            recall = tp / len(class_gts)

            precisions.append(precision)
            recalls.append(recall)

        # Calculate Average Precision for this class
        ap = calculate_ap(precisions, recalls)
        aps.append(ap)

    # Return mean of all Average Precisions
    return sum(aps) / len(aps) if len(aps) > 0 else 0.0


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (0 to 1)
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def calculate_ap(precisions, recalls):
    """
    Calculate Average Precision from precision-recall curve.

    Args:
        precisions: List of precision values
        recalls: List of recall values

    Returns:
        Average Precision score
    """
    # Add sentinel values at the end
    precisions = [0] + precisions + [0]
    recalls = [0] + recalls + [1]

    # Calculate area under precision-recall curve
    ap = 0
    for i in range(len(precisions) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap
```

---

**2: Recall**

**What is Recall:** The ratio of correctly predicted positive detections to all actual objects present. Out of all real objects present, how many did the model successfully detect. High recall = fewer false negatives. It evaluates the model's ability to capture every possible object without missing them. Recall is calculated by dividing true positives by the sum of true positives and false negatives. It ensures that even subtle and partially visible objects are not overlooked. Higher recall makes the model reliable for the scenarios where missing object is critical like medical imaging and security surveillance deployments.

**File: `metrics/recall_calculation.py`**
```python
"""
Recall Metric Calculation
--------------------------
Recall measures the model's ability to find all objects.
Critical for applications where missing an object is costly.
"""


def calculate_recall(true_positives, false_negatives):
    """
    Calculate recall metric for object detection.

    Recall = TP / (TP + FN)

    High recall means model misses fewer objects (low false negatives).
    Important for: security systems, medical imaging, autonomous vehicles.

    Args:
        true_positives: Number of correctly detected objects
        false_negatives: Number of missed objects (present but not detected)

    Returns:
        Recall score (0 to 1, higher is better)

    Example:
        If there are 100 cars in images and model detects 90:
        TP = 90, FN = 10
        Recall = 90 / (90 + 10) = 0.9 or 90%
    """
    # Avoid division by zero
    if (true_positives + false_negatives) == 0:
        return 0.0

    return true_positives / (true_positives + false_negatives)


# Example usage
if __name__ == "__main__":
    # Example: Detection results
    tp = 85  # Correctly detected 85 objects
    fn = 15  # Missed 15 objects

    recall = calculate_recall(tp, fn)
    print(f"Recall: {recall:.2%}")  # Output: Recall: 85.00%

    # Low recall example (many missed objects)
    recall_low = calculate_recall(50, 50)
    print(f"Low Recall: {recall_low:.2%}")  # Output: Low Recall: 50.00%

    # High recall example (few missed objects)
    recall_high = calculate_recall(95, 5)
    print(f"High Recall: {recall_high:.2%}")  # Output: High Recall: 95.00%
```

---

**3: Precision**

**What is Precision:** It measures the accuracy of a model's positive predictions, indicating the proportion of items predicted as positive that were actually correct. It is calculated as True Positives / (True Positives + False Positives). High precision means the model has few false positives. This metric helps you understand how trustworthy your model is when it predicts a specific outcome.

**File: `metrics/precision_calculation.py`**
```python
"""
Precision Metric Calculation
-----------------------------
Precision measures how accurate the model's predictions are.
Critical for applications where false alarms are costly.
"""


def calculate_precision(true_positives, false_positives):
    """
    Calculate precision metric for object detection.

    Precision = TP / (TP + FP)

    High precision means model makes fewer false alarms (low false positives).
    Important for: reducing false alarms in security, medical diagnostics.

    Args:
        true_positives: Number of correctly detected objects
        false_positives: Number of incorrect detections (false alarms)

    Returns:
        Precision score (0 to 1, higher is better)

    Example:
        If model makes 100 predictions and 90 are correct:
        TP = 90, FP = 10
        Precision = 90 / (90 + 10) = 0.9 or 90%
    """
    # Avoid division by zero
    if (true_positives + false_positives) == 0:
        return 0.0

    return true_positives / (true_positives + false_positives)


# Example usage
if __name__ == "__main__":
    # Example: Detection results
    tp = 90  # 90 correct predictions
    fp = 10  # 10 false alarms

    precision = calculate_precision(tp, fp)
    print(f"Precision: {precision:.2%}")  # Output: Precision: 90.00%

    # Low precision example (many false alarms)
    precision_low = calculate_precision(50, 50)
    print(f"Low Precision: {precision_low:.2%}")  # Output: Low Precision: 50.00%

    # High precision example (few false alarms)
    precision_high = calculate_precision(95, 5)
    print(f"High Precision: {precision_high:.2%}")  # Output: High Precision: 95.00%
```

---

**4: IoU (Intersection over Union)**

**What is IoU:** It measures how much the predicted object bounding box overlaps with the real (ground truth) box. It's the ratio between overlap area/total combined area. Higher IoU = better prediction accuracy by the model. It compares the difference between ground truths and predictions, making visible the accuracy of the model. As visible in the image below green box is ground truth actual object area and the red box is predicted area by model so we can see slightly difference in overlapping of the boxes this visibly shows the accuracy of the model.

![IOU](/images/IOU__.png)

Certain threshold is set for predicting the class with accuracy.

![IOU_threshold](/images/IOU_THR.png)

**File: `metrics/iou_calculation.py`**
```python
"""
IoU (Intersection over Union) Calculation
------------------------------------------
IoU measures how well predicted boxes align with ground truth boxes.
Standard metric for evaluating bounding box accuracy.
"""

import numpy as np


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.

    IoU = Area of Overlap / Area of Union

    IoU tells you how well the predicted box matches the actual object location.
    - IoU = 1.0: Perfect match
    - IoU = 0.5: Decent detection (common threshold)
    - IoU = 0.0: No overlap at all

    Args:
        box1: [x1, y1, x2, y2] first box coordinates
              (x1, y1) = top-left corner
              (x2, y2) = bottom-right corner
        box2: [x1, y1, x2, y2] second box coordinates

    Returns:
        IoU score between 0 and 1 (higher is better)

    Example:
        Ground truth box: [100, 100, 200, 200]
        Predicted box:    [110, 110, 210, 210]
        IoU ≈ 0.68 (good detection)
    """
    # Calculate intersection rectangle coordinates
    # Take maximum of x1, y1 (top-left) and minimum of x2, y2 (bottom-right)
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    # max(0, ...) ensures no negative area if boxes don't overlap
    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)
    intersection = intersection_width * intersection_height

    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union area
    # Union = Total area covered by both boxes
    # Union = Area1 + Area2 - Intersection (subtract overlap once)
    union = box1_area + box2_area - intersection

    # Calculate IoU
    # Avoid division by zero
    iou = intersection / union if union > 0 else 0

    return iou


def batch_iou(boxes1, boxes2):
    """
    Calculate IoU for multiple box pairs efficiently.

    Args:
        boxes1: numpy array of shape [N, 4]
        boxes2: numpy array of shape [N, 4]

    Returns:
        IoU scores for each pair [N]
    """
    # Calculate intersection
    x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate areas
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Calculate union
    union = boxes1_area + boxes2_area - intersection

    # Calculate IoU
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero

    return iou


# Example usage
if __name__ == "__main__":
    # Example 1: Perfect match
    ground_truth = [100, 100, 200, 200]
    prediction = [100, 100, 200, 200]
    iou = calculate_iou(ground_truth, prediction)
    print(f"Perfect match IoU: {iou:.2f}")  # Output: 1.00

    # Example 2: Good detection
    ground_truth = [100, 100, 200, 200]
    prediction = [110, 110, 210, 210]
    iou = calculate_iou(ground_truth, prediction)
    print(f"Good detection IoU: {iou:.2f}")  # Output: ~0.68

    # Example 3: No overlap
    ground_truth = [100, 100, 200, 200]
    prediction = [300, 300, 400, 400]
    iou = calculate_iou(ground_truth, prediction)
    print(f"No overlap IoU: {iou:.2f}")  # Output: 0.00
```

---

**5: F1 Score**

**What is F1 Score:** F1 score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. Useful when you need to balance between false positives and false negatives.

**File: `metrics/f1_score_calculation.py`**
```python
"""
F1 Score Calculation
--------------------
F1 Score balances precision and recall into a single metric.
Useful when you need equal importance for both metrics.
"""


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    F1 score is the harmonic mean of precision and recall.
    - F1 = 1.0: Perfect precision and recall
    - F1 = 0.5: Moderate performance
    - F1 = 0.0: Poor performance

    Use F1 when both precision and recall are equally important.

    Args:
        precision: Precision score (0 to 1)
        recall: Recall score (0 to 1)

    Returns:
        F1 score (0 to 1, higher is better)

    Example:
        Precision = 0.8, Recall = 0.9
        F1 = 2 * (0.8 * 0.9) / (0.8 + 0.9) = 0.847
    """
    # Avoid division by zero
    if (precision + recall) == 0:
        return 0.0

    # Calculate harmonic mean
    # Harmonic mean penalizes extreme values (unlike arithmetic mean)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


# Example usage
if __name__ == "__main__":
    # Example 1: Balanced precision and recall
    precision = 0.85
    recall = 0.85
    f1 = calculate_f1_score(precision, recall)
    print(f"Balanced F1: {f1:.3f}")  # Output: 0.850

    # Example 2: High precision, low recall
    precision = 0.95
    recall = 0.60
    f1 = calculate_f1_score(precision, recall)
    print(f"Imbalanced F1: {f1:.3f}")  # Output: 0.737

    # Example 3: Perfect scores
    precision = 1.0
    recall = 1.0
    f1 = calculate_f1_score(precision, recall)
    print(f"Perfect F1: {f1:.3f}")  # Output: 1.000
```

---

**6: Confusion Matrix**

**What is Confusion Matrix:** Confusion matrix visualizes the performance of the model by showing true positives, false positives, true negatives and false negatives for each class. It helps identify which classes the model confuses with each other.

**File: `metrics/confusion_matrix.py`**
```python
"""
Confusion Matrix Visualization
-------------------------------
Confusion matrix helps identify which classes the model confuses.
Essential for understanding model weaknesses.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(predictions, ground_truths, class_names):
    """
    Plot confusion matrix for object detection.

    Confusion matrix shows:
    - Diagonal: Correct predictions
    - Off-diagonal: Confusion between classes

    Helps identify:
    - Which classes are hard to detect
    - Which classes are confused with each other
    - Overall model performance per class

    Args:
        predictions: List of predicted class labels [N]
        ground_truths: List of actual class labels [N]
        class_names: List of class names ['car', 'person', 'bike', ...]

    Example:
        If model confuses 'dog' with 'cat' frequently,
        you'll see high values in off-diagonal cells.
    """
    num_classes = len(class_names)

    # Initialize confusion matrix with zeros
    # Rows = actual classes, Columns = predicted classes
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Fill confusion matrix
    for pred, gt in zip(predictions, ground_truths):
        # Increment count at position [actual_class, predicted_class]
        matrix[gt][pred] += 1

    # Create heatmap visualization
    plt.figure(figsize=(10, 8))

    # Plot heatmap with annotations
    sns.heatmap(
        matrix,
        annot=True,          # Show numbers in cells
        fmt='d',             # Integer format
        cmap='Blues',        # Color scheme
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Predictions'}
    )

    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix - Model Performance by Class')
    plt.tight_layout()

    # Save plot
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print analysis
    print("\nConfusion Matrix Analysis:")
    print("-" * 50)

    # Calculate accuracy per class
    for i, class_name in enumerate(class_names):
        total = matrix[i].sum()
        correct = matrix[i][i]
        accuracy = (correct / total * 100) if total > 0 else 0
        print(f"{class_name}: {accuracy:.1f}% accuracy ({correct}/{total})")


# Example usage
if __name__ == "__main__":
    # Example detection results
    predictions = [0, 0, 1, 1, 2, 0, 1, 2, 2, 1]  # Predicted classes
    ground_truths = [0, 0, 1, 2, 2, 1, 1, 2, 2, 1]  # Actual classes
    class_names = ['car', 'person', 'bike']

    plot_confusion_matrix(predictions, ground_truths, class_names)
```

These are the steps needed to consider and critically evaluate before selecting and finalizing the model for the system.

---

## Data Preprocessing

**Why data preprocessing matters:** This is the most important and critical part of whole Machine Learning system. Whole performance of the model depends upon the dataset on which it's been trained. There is a famous saying (garbage in garbage out). Dataset must be cleaned, well balanced and must cover all the features that are required, and systems need to detect.

---

### Steps for Data Preprocessing

**Dataset splitting strategy:** Dataset is split into 3 folders: Train/Validation/Test. This splitting ensures model learns from training data, tunes hyperparameters on validation data, and evaluates final performance on unseen test data.

**Train**: This folder contains the larger amount of the dataset and models learn patterns and features from the images and annotations. Typically 70-80% of total data.

**Validation**: This folder is used during training to tune hyperparameters and monitor model performance. It helps prevent overfitting. Typically 10-15% of total data.

**Test**: This contains the final set of images that model hasn't seen before. It used after training to check how accurate the model has performed in real-time scenarios, simply testing the model on new images that were not included in the train dataset on which labels were drawn. Typically 10-15% of total data.

**File: `preprocessing/split_dataset.py`**
```python
"""
Dataset Splitting Script
------------------------
Split dataset into train, validation and test sets.
Essential for proper model evaluation and preventing overfitting.
"""

import os
import shutil
import random
from pathlib import Path


def split_dataset(source_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation and test sets.

    Why split data:
    - Train: Model learns patterns from this data
    - Validation: Tune hyperparameters, monitor overfitting
    - Test: Final evaluation on completely unseen data

    Args:
        source_folder: Path to folder containing all images and labels
        train_ratio: Proportion for training set (default: 0.7 = 70%)
        val_ratio: Proportion for validation set (default: 0.15 = 15%)
        test_ratio: Proportion for test set (default: 0.15 = 15%)

    Example:
        If you have 1000 images:
        - Train: 700 images (70%)
        - Val: 150 images (15%)
        - Test: 150 images (15%)
    """
    # Validate ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Ratios must sum to 1.0"

    # Get all image files from source folder
    # Support common image formats
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(source_folder)
              if f.lower().endswith(image_extensions)]

    # Shuffle images for random splitting
    # Set seed for reproducibility
    random.seed(42)
    random.shuffle(images)

    # Calculate split indices
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split files into three sets
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Create directories and move files
    for split_name, files in [('train', train_files),
                               ('val', val_files),
                               ('test', test_files)]:

        # Create split directory
        split_dir = os.path.join(source_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)

        print(f"\nProcessing {split_name} set ({len(files)} files)...")

        for file in files:
            # Move image file
            src_image = os.path.join(source_folder, file)
            dst_image = os.path.join(split_dir, file)
            shutil.move(src_image, dst_image)

            # Also move corresponding label file if exists
            # Support multiple label formats
            label_extensions = ['.txt', '.json', '.xml']
            base_name = os.path.splitext(file)[0]

            for ext in label_extensions:
                label_file = base_name + ext
                src_label = os.path.join(source_folder, label_file)

                if os.path.exists(src_label):
                    dst_label = os.path.join(split_dir, label_file)
                    shutil.move(src_label, dst_label)

    # Print summary
    print("\n" + "="*50)
    print("Dataset Split Summary:")
    print("="*50)
    print(f"Train: {len(train_files)} images ({train_ratio*100:.1f}%)")
    print(f"Validation: {len(val_files)} images ({val_ratio*100:.1f}%)")
    print(f"Test: {len(test_files)} images ({test_ratio*100:.1f}%)")
    print(f"Total: {total} images")
    print("="*50)


# Example usage
if __name__ == "__main__":
    # Path to your dataset folder
    dataset_path = "path/to/your/dataset"

    # Split dataset
    split_dataset(
        source_folder=dataset_path,
        train_ratio=0.7,   # 70% for training
        val_ratio=0.15,    # 15% for validation
        test_ratio=0.15    # 15% for testing
    )
```

---

#### Data Cleaning

**Removing bad data:** Removing duplicates to remove the wrong information, incorrect annotations and irrelevant images that are not required because model training requires resources like GPU and memory for data-saving so be sure to manage all the resources efficiently.

**File: `preprocessing/data_cleaning.py`**
```python
"""
Data Cleaning Script
--------------------
Remove duplicates, validate annotations, and clean dataset.
Essential for preventing model from learning from corrupted data.
"""

import hashlib
import os
from PIL import Image


def remove_duplicate_images(folder_path):
    """
    Remove duplicate images from dataset based on image hash.

    Why remove duplicates:
    - Duplicates waste training time and memory
    - Can cause overfitting to duplicated samples
    - Inflate dataset size artificially

    How it works:
    - Calculate MD5 hash of each image
    - If hash already seen, it's a duplicate
    - Delete duplicate and keep first occurrence

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of removed duplicate filenames
    """
    seen_hashes = set()  # Store hashes of images we've seen
    duplicates = []       # Track removed duplicates

    print(f"Scanning for duplicates in {folder_path}...")

    # Process each file in folder
    for filename in os.listdir(folder_path):
        # Only process image files
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue

        filepath = os.path.join(folder_path, filename)

        # Calculate image hash (MD5)
        # Hash is unique identifier for image content
        with open(filepath, 'rb') as f:
            # Read file in chunks for memory efficiency
            img_data = f.read()
            img_hash = hashlib.md5(img_data).hexdigest()

        # Check if we've seen this image before
        if img_hash in seen_hashes:
            # Duplicate found - remove it
            duplicates.append(filename)
            os.remove(filepath)
            print(f"Removed duplicate: {filename}")

            # Also remove corresponding label file if exists
            label_file = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(folder_path, label_file)
            if os.path.exists(label_path):
                os.remove(label_path)
        else:
            # New image - remember its hash
            seen_hashes.add(img_hash)

    print(f"\nTotal duplicates removed: {len(duplicates)}")
    return duplicates


def validate_annotations(images_folder, labels_folder):
    """
    Validate that all images have corresponding annotations and vice versa.

    Why validate:
    - Images without labels can't be used for training
    - Labels without images indicate missing data
    - Prevents crashes during training

    Args:
        images_folder: Path to images folder
        labels_folder: Path to labels folder

    Returns:
        Tuple of (images_without_labels, labels_without_images)
    """
    print("Validating annotations...")

    # Get base filenames (without extension)
    # Extract just the name part, ignoring .jpg, .txt extensions
    images = set([os.path.splitext(f)[0] for f in os.listdir(images_folder)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    labels = set([os.path.splitext(f)[0] for f in os.listdir(labels_folder)
                  if f.lower().endswith('.txt')])

    # Find mismatches
    images_without_labels = images - labels  # Images that have no label file
    labels_without_images = labels - images  # Labels that have no image file

    # Report findings
    if images_without_labels:
        print(f"\nWARNING: {len(images_without_labels)} images without labels:")
        for img in list(images_without_labels)[:5]:  # Show first 5
            print(f"  - {img}")
        if len(images_without_labels) > 5:
            print(f"  ... and {len(images_without_labels)-5} more")

    if labels_without_images:
        print(f"\nWARNING: {len(labels_without_images)} labels without images:")
        for lbl in list(labels_without_images)[:5]:  # Show first 5
            print(f"  - {lbl}")
        if len(labels_without_images) > 5:
            print(f"  ... and {len(labels_without_images)-5} more")

    if not images_without_labels and not labels_without_images:
        print("✓ All images have corresponding labels!")

    return images_without_labels, labels_without_images


def check_corrupted_images(folder_path):
    """
    Check for corrupted or unreadable images.

    Why check:
    - Corrupted images crash training
    - Unreadable images waste time
    - Better to find issues before training starts

    Args:
        folder_path: Path to folder containing images

    Returns:
        List of corrupted image filenames
    """
    corrupted = []

    print("Checking for corrupted images...")

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
            continue

        filepath = os.path.join(folder_path, filename)

        try:
            # Try to open and verify image
            img = Image.open(filepath)
            img.verify()  # Verify it's actually an image
        except Exception as e:
            # Image is corrupted or unreadable
            corrupted.append(filename)
            print(f"Corrupted image: {filename} - {str(e)}")

    print(f"\nTotal corrupted images: {len(corrupted)}")
    return corrupted


# Example usage
if __name__ == "__main__":
    # Path to your dataset
    dataset_path = "path/to/dataset"
    images_path = os.path.join(dataset_path, "images")
    labels_path = os.path.join(dataset_path, "labels")

    # Remove duplicates
    remove_duplicate_images(images_path)

    # Validate annotations
    validate_annotations(images_path, labels_path)

    # Check for corrupted images
    check_corrupted_images(images_path)
```

---

#### Image Resizing

**Standardizing input size:** Resizing all the images to a uniform size that is required by the selected model for the input. Different models require different input sizes (e.g., YOLO uses 640x640, some models use 512x512).

**File: `preprocessing/image_resizing.py`**
```python
"""
Image Resizing Script
---------------------
Resize all images to uniform size required by the model.
Essential because models expect fixed input dimensions.
"""

from PIL import Image
import os
from pathlib import Path


def resize_images(input_folder, output_folder, target_size=(640, 640),
                 maintain_aspect_ratio=False):
    """
    Resize all images in folder to target size.

    Why resize:
    - Models require fixed input size (e.g., 640x640 for YOLO)
    - Consistent size = consistent batch processing
    - Smaller size = faster training/inference

    Args:
        input_folder: Source folder containing original images
        output_folder: Destination folder for resized images
        target_size: Tuple of (width, height) for target size
                    Common sizes: (640, 640), (512, 512), (416, 416)
        maintain_aspect_ratio: If True, pad image to maintain aspect ratio
                              If False, stretch/squeeze to exact size

    Example:
        Original: 1920x1080
        Target: 640x640
        - Without aspect ratio: Squeezed to 640x640
        - With aspect ratio: Resized to 640x360, padded to 640x640
    """
    # Create output folder if doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(image_extensions)]

    print(f"Resizing {len(image_files)} images to {target_size}...")

    for idx, filename in enumerate(image_files):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open image
        img = Image.open(input_path)
        original_size = img.size

        if maintain_aspect_ratio:
            # Resize maintaining aspect ratio
            img = resize_with_padding(img, target_size)
        else:
            # Direct resize (may distort image)
            img = img.resize(target_size, Image.LANCZOS)

        # Save resized image
        # Use same format as original
        img.save(output_path, quality=95)

        # Print progress every 100 images
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")

    print(f"\n✓ Resized all images to {target_size}")
    print(f"Saved to: {output_folder}")


def resize_with_padding(img, target_size):
    """
    Resize image maintaining aspect ratio and pad to target size.

    This prevents image distortion by adding padding.

    Args:
        img: PIL Image object
        target_size: (width, height) target size

    Returns:
        Resized and padded image
    """
    target_width, target_height = target_size

    # Calculate scaling factor to fit within target size
    # Use minimum scale to ensure image fits
    width, height = img.size
    scale = min(target_width / width, target_height / height)

    # Calculate new size
    new_width = int(width * scale)
    new_height = int(height * scale)

    # Resize image
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Create new image with target size (black background)
    new_img = Image.new('RGB', target_size, (0, 0, 0))

    # Calculate paste position (center the resized image)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image onto padded background
    new_img.paste(img, (paste_x, paste_y))

    return new_img


# Example usage
if __name__ == "__main__":
    # Resize images for YOLO (640x640)
    resize_images(
        input_folder="dataset/original_images",
        output_folder="dataset/resized_images",
        target_size=(640, 640),
        maintain_aspect_ratio=True  # Prevents distortion
    )

    # Resize for different model (512x512)
    resize_images(
        input_folder="dataset/original_images",
        output_folder="dataset/resized_512",
        target_size=(512, 512),
        maintain_aspect_ratio=False  # Allow stretching
    )
```

---

#### Data Augmentation

**Increasing dataset diversity:** This step of data-preprocessing plays vital role in model performance and generalization. This step includes rotation, flipping (horizontal/vertical) to increase diversity, scaling by zooming in/out to simulate objects at different distances, and brightness adjustment to simulate different lighting conditions like sunlight or cloudy weather. By training model on images with varying brightness, it learns to recognize objects accurately regardless of environmental changes. This helps reduce false positives and improves the model's performance for every condition.

**File: `preprocessing/data_augmentation.py`**
```python
"""
Data Augmentation Script
-------------------------
Apply various transformations to increase dataset diversity.
Critical for improving model generalization and robustness.
"""

import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import random


def augment_dataset(input_folder, output_folder, augmentations_per_image=5):
    """
    Apply data augmentation to increase dataset size and diversity.

    Why augment:
    - Helps model generalize better to new data
    - Prevents overfitting to training samples
    - Simulates real-world variations (lighting, angles, etc.)
    - Increases effective dataset size without collecting new images

    Augmentation techniques applied:
    - Horizontal/Vertical flipping
    - Rotation (random angles)
    - Brightness adjustment
    - Scaling (zoom in/out)
    - Noise addition

    Args:
        input_folder: Source folder containing original images
        output_folder: Destination folder for augmented images
        augmentations_per_image: Number of augmented versions per image

    Example:
        If you have 100 images and augmentations_per_image=5:
        Output = 100 original + 500 augmented = 600 total images
    """
    os.makedirs(output_folder, exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    print(f"Augmenting {len(image_files)} images...")

    for idx, filename in enumerate(image_files):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        # Read original image
        img = cv2.imread(input_path)

        # Save original image to output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)

        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            # Apply random augmentations
            augmented_img = apply_random_augmentation(img.copy())

            # Save augmented image
            aug_filename = f"{base_name}_aug_{aug_idx}{extension}"
            aug_path = os.path.join(output_folder, aug_filename)
            cv2.imwrite(aug_path, augmented_img)

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(image_files)} images")

    total_images = len(image_files) * (1 + augmentations_per_image)
    print(f"\n✓ Augmentation complete!")
    print(f"Original images: {len(image_files)}")
    print(f"Total images (including augmented): {total_images}")


def apply_random_augmentation(img):
    """
    Apply random augmentation techniques to an image.

    Args:
        img: Input image (numpy array from OpenCV)

    Returns:
        Augmented image
    """
    # Randomly choose which augmentations to apply
    augmentations = [
        flip_horizontal,
        flip_vertical,
        rotate_random,
        adjust_brightness,
        scale_image,
        add_noise
    ]

    # Apply 1-3 random augmentations
    num_augs = random.randint(1, 3)
    selected_augs = random.sample(augmentations, num_augs)

    for aug_func in selected_augs:
        img = aug_func(img)

    return img


def flip_horizontal(img):
    """Flip image horizontally (left-right)."""
    return cv2.flip(img, 1)


def flip_vertical(img):
    """Flip image vertically (top-bottom)."""
    return cv2.flip(img, 0)


def rotate_random(img):
    """Rotate image by random angle (-30 to 30 degrees)."""
    angle = random.randint(-30, 30)
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform rotation
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height),
                             borderMode=cv2.BORDER_REFLECT)
    return rotated


def adjust_brightness(img):
    """Adjust image brightness randomly (0.5x to 1.5x)."""
    # Convert to PIL for brightness adjustment
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Random brightness factor
    factor = random.uniform(0.5, 1.5)
    enhancer = ImageEnhance.Brightness(pil_img)
    brightened = enhancer.enhance(factor)

    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(brightened), cv2.COLOR_RGB2BGR)


def scale_image(img):
    """Scale (zoom) image randomly (0.8x to 1.2x)."""
    scale = random.uniform(0.8, 1.2)
    height, width = img.shape[:2]

    # Calculate new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)

    # Resize image
    scaled = cv2.resize(img, (new_width, new_height))

    # Crop or pad to original size
    if scale > 1.0:
        # Crop from center
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        scaled = scaled[start_y:start_y+height, start_x:start_x+width]
    else:
        # Pad to original size
        pad_y = (height - new_height) // 2
        pad_x = (width - new_width) // 2
        scaled = cv2.copyMakeBorder(scaled, pad_y, height-new_height-pad_y,
                                   pad_x, width-new_width-pad_x,
                                   cv2.BORDER_REFLECT)

    return scaled


def add_noise(img):
    """Add random Gaussian noise to image."""
    # Generate Gaussian noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)

    # Add noise to image
    noisy = cv2.add(img, noise)

    return noisy


# Example usage
if __name__ == "__main__":
    augment_dataset(
        input_folder="dataset/train/images",
        output_folder="dataset/train/augmented",
        augmentations_per_image=5  # Creates 5 augmented versions per image
    )
```

---

#### Format Checking/Conversion

**Ensuring correct annotation format:** Every model has its own specific annotation format for reading labels. Annotations must be in the correct format.

**YOLO Format**: YOLO models use TXT file as label file containing (class + bounding boxes x, y coordinates) for each image file and names of both files image and label file should be exact same.
Example: "Image-1.jpg = image-1.txt"

![Yolo format labels file](/images/yololabel.png)

**Detectron2, Faster R-CNN, Mask R-CNN, RF-DETR**: These models take JSON files as annotations. This file contains the metadata of the dataset including the information of each image (filename, size), the objects in each image (bounding boxes, categories of the object like class) and also the list of all objects/classes. Annotations are linked to the corresponding images using ids.

![json file structure](/images/examplee.png)

**File: `preprocessing/format_conversion.py`**
```python
"""
Annotation Format Conversion Script
------------------------------------
Convert between different annotation formats (YOLO, COCO, Pascal VOC).
Essential for using datasets with different models.
"""

import json
import os
from pathlib import Path


def yolo_to_coco(yolo_labels_folder, images_folder, output_json, class_names):
    """
    Convert YOLO format annotations to COCO JSON format.

    YOLO format (per image, .txt file):
        class_id center_x center_y width height (normalized 0-1)

    COCO format (single .json file):
        {
            "images": [...],
            "annotations": [...],
            "categories": [...]
        }

    Args:
        yolo_labels_folder: Folder containing YOLO .txt label files
        images_folder: Folder containing corresponding images
        output_json: Path to output COCO JSON file
        class_names: List of class names ['car', 'person', ...]
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories
    for idx, class_name in enumerate(class_names):
        coco_data["categories"].append({
            "id": idx,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1
    image_id = 1

    # Process each image
    for image_file in os.listdir(images_folder):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Get image dimensions
        from PIL import Image
        img_path = os.path.join(images_folder, image_file)
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Add image info
        coco_data["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": img_width,
            "height": img_height
        })

        # Read YOLO labels
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(yolo_labels_folder, label_file)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # Parse YOLO format: class_id center_x center_y width height
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    center_x = float(parts[1]) * img_width
                    center_y = float(parts[2]) * img_height
                    bbox_width = float(parts[3]) * img_width
                    bbox_height = float(parts[4]) * img_height

                    # Convert to COCO format (x, y, width, height)
                    # where (x, y) is top-left corner
                    x = center_x - (bbox_width / 2)
                    y = center_y - (bbox_height / 2)

                    # Add annotation
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

        image_id += 1

    # Save COCO JSON
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"✓ Converted YOLO to COCO format")
    print(f"Output: {output_json}")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")


def coco_to_yolo(coco_json, output_folder):
    """
    Convert COCO JSON format to YOLO format.

    Args:
        coco_json: Path to COCO JSON file
        output_folder: Folder to save YOLO .txt label files
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load COCO data
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)

    # Create image_id to filename mapping
    image_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Convert each image's annotations
    for image_id, annotations in annotations_by_image.items():
        img = image_info[image_id]
        img_width = img['width']
        img_height = img['height']

        # Create YOLO label file
        label_filename = os.path.splitext(img['file_name'])[0] + '.txt'
        label_path = os.path.join(output_folder, label_filename)

        with open(label_path, 'w') as f:
            for ann in annotations:
                # Get COCO bbox (x, y, width, height)
                x, y, width, height = ann['bbox']
                category_id = ann['category_id']

                # Convert to YOLO format (normalized center coordinates)
                center_x = (x + width / 2) / img_width
                center_y = (y + height / 2) / img_height
                norm_width = width / img_width
                norm_height = height / img_height

                # Write YOLO format
                f.write(f"{category_id} {center_x} {center_y} {norm_width} {norm_height}\n")

    print(f"✓ Converted COCO to YOLO format")
    print(f"Output folder: {output_folder}")
    print(f"Created {len(annotations_by_image)} label files")


# Example usage
if __name__ == "__main__":
    # Convert YOLO to COCO
    yolo_to_coco(
        yolo_labels_folder="dataset/labels",
        images_folder="dataset/images",
        output_json="dataset/annotations.json",
        class_names=['car', 'person', 'bike']
    )

    # Convert COCO to YOLO
    coco_to_yolo(
        coco_json="dataset/annotations.json",
        output_folder="dataset/yolo_labels"
    )
```

---

### Data Preprocessing Platform

**[Roboflow](https://roboflow.com/)**

It's a web-based tool that has functionality to organize the data, preprocess including augmentation and format conversion among all the different models. It allows users to upload their dataset and annotate it, augment it, select the model and it will generate well balanced dataset including the Train & Test. It allows you to train model in it as well on some free credits, and then you can choose the "paid version".

![](/images/roboflow.png)
![](/images/code_snipper.png)

It guides you through all the necessary steps you need to complete and then generates an API key for direct dataset integration via API or as a downloadable .zip file.

Here you can select the model format.
![](/images/format.png)

---

### Dataset Collection

**Finding datasets:** Now comes the main part that where to find and get the dataset to train model, so the universal platform where multiple datasets are available is [Kaggle](https://www.kaggle.com/). It's widely used and most of the general datasets are available on it for free (check for the license of usage for each).

**Video Frames**

In case if dataset isn't available on this platform then we have second option: fetching frames from videos and real-time recordings and then defining the objects names. Let's assume you are collecting dataset for any company that has some products and they want to count all of them on the last stage of conveyor belt to count the production of units. So object can be anything special and not available, so then it comes that way of collecting dataset from the video recordings, just taking the video of those products where they are (e.g., on the conveyor belt) and then extracting frames from it using program as well (PYTHON).

This is the simple Python code snippet that can be used to extract frames from the videos to collect the dataset.

**File: `dataset_collection/extract_frames.py`**
```python
"""
Video Frame Extraction Script
------------------------------
Extract frames from video files to collect dataset images.
Useful when you don't have existing dataset.
"""

import cv2
import os


def extract_frames_from_video(video_path, num_frames_to_extract, output_folder):
    """
    Extract frames from video file.

    Use this when:
    - No existing dataset available
    - Need to collect custom object images
    - Have surveillance footage or recorded videos

    Args:
        video_path: Path to input video file
        num_frames_to_extract: Number of frames to extract
        output_folder: Output directory for extracted frames

    Example:
        extract_frames_from_video(
            "conveyor_belt_recording.mp4",
            1000,
            "dataset/raw_frames"
        )
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video info:")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {total_frames/fps:.2f} seconds")

    # Calculate frame interval to get evenly distributed frames
    interval = total_frames // num_frames_to_extract

    count = 0
    extracted = 0

    while extracted < num_frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at intervals
        if count % interval == 0:
            frame_filename = f"{output_folder}/frame_{extracted:05d}.jpg"
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {extracted + 1}/{num_frames_to_extract}")
            extracted += 1

        count += 1

    cap.release()
    print(f"\n✓ Extracted {extracted} frames to {output_folder}")


# Example usage
if __name__ == "__main__":
    # Set your variables here
    video_path = "input_video.mp4"
    num_frames_to_extract = 500
    output_folder = "dataset/frames"

    extract_frames_from_video(video_path, num_frames_to_extract, output_folder)
```

After you have collected your dataset images, you can use Roboflow to annotate them and convert the dataset into the format required by your model.

[Tutorial](https://youtu.be/Dk-6MCQ9j-c?si=dIzQyNsWWxoysQLV) - Complete guide how to prepare the dataset and getting it ready for the training.

---

## Training

**Understanding the training process:** This step is similar to training any human being for doing a specific task. So like that AI models are also trained on the specific dataset to perform predictions based on the patterns learned from the dataset. To start the training firstly we prepare dataset (discussed earlier), then selecting a specific architecture of the model. We can do this by fine-tuning an existing pre-trained model (YOLO, RT-DETR etc) or by defining a custom architecture of the model using frameworks like PyTorch, TensorFlow, Keras etc. These frameworks provide libraries and packages to define, train, test and evaluate models.

When setting up for the training, hyperparameters are configured to control and manage the learning process. They influence how the training progresses and they directly impact the model's performance and efficiency. Like how, hyperparameters determine how quickly and effectively model learns from patterns of the data. Hyperparameters are epochs, learning rate, batch size, momentum and weight decay etc. Let's have a simple overview of each and see what is role of each hyperparameter for affecting the training process at all.

**Learning Rate**: Very crucial unit for the training process like it covers the 70% of the model efficiency during training process because it defines the steps models need to take to learn patterns from the data. It controls how much model's weights are updated during training. Too high can unstable the model training and too low can slowdown the training.

**Batch Size**: Number of random samples from the dataset are processed before model updates its weights.

**Epochs**: Number of times entire training dataset is passed through the model. More epochs let the model learn better. More epochs more iteration over the dataset and more detail learning.

**Momentum**: Used in optimizers to accelerate in consistent directions improving convergence speed.

**Weight Decay**: A regularization term added to prevent overfitting by penalizing large weights.

---

### PyTorch

**PyTorch framework:** Its an open-source framework that provides tools to build, train, and fine-tune models (neural networks). It provides each step we need to make a working model. It has vast network of libraries:
- torch for core operations and inference applications
- torchvision for computer vision image processing work. It has preloaded datasets that we can just import from library and the pretrained models that can be loaded to finetune them and also it provides the structure for defining custom models for different tasks
- torchaudio for audio processing
- torchtext for natural language processing tasks

Minimal implementation of the PyTorch code for fine-tuning (training) object detection model on custom dataset.

[Training](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/_training.py)

And this one is for inference:

[Inference](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/_inference.py)

**Code implementation examples:**
- [torch](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torch_.py)
- [torchaudio](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torchaudio_.py)
- [torchvision](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torchvision_.py)

#### Official documentations of framework/references:
- [Docs](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) - This documentation covers everything need to define a custom neural network/model using PyTorch
- [Explanation of neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
- [Fine-tuning using torchvision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

---

### TensorFlow

**TensorFlow framework:** Its an open-source framework for building training and deploying AI models. It also provides libraries and architectures to build custom neural network/models and also some pre-trained models for inference, pre-loaded datasets, post-processing and preprocessing tools for datasets handling. Major libraries from TensorFlow and their code implementations:

"tensorflow" core library for defining models performing operations and training models:
[tensorflow](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_core.py)

tensorflow_hub its model zoo of the TensorFlow a great repository for reusable pre-trained models to fine-tune and integrate them directly into the applications:
[tensorflow_hub](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_hub_.py)

tf.data for loading, preprocessing and handling dataset:
[tf.data](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tf_data.py)

tf.image utilities for image processing tasks:
[tf.image](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tf_image.py)

[Object detection with TensorFlow](https://www.tensorflow.org/hub/tutorials/tf2_object_detection)
[Inference](https://github.com/aaliyanahmed1/tensorflow_/blob/main/tf2_object_detection.ipynb)

[Action recognition from video](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub) - Recognizing action and events from the videos using TensorFlow.

This TensorFlow implementation contains deep detailed explanation of fine-tuning an object detection model on custom dataset. All are pretrained and preloaded in TensorFlow no need to download them manually. All of stuff is built-in.

[Fine-tuning Explanation](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_explain.py)

This was all about introduction of the frameworks now let's get back to the training.

[Training using TensorFlow](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/training.py) - This is the sample practical implementation of training a model on custom dataset for object detection.

---

### Fine-Tuning Models for Custom Detections

**Selecting models:** There are many state of the art object detection models that can be finetuned and integrated into applications. But there are some checks we need to see first before going ahead, e.g., License, Resources (Hardware), size and speed according to use case as we discussed earlier.

**1: RF-DETR** by [Roboflow](https://roboflow.com/) is real-time transformer-based object detection model. It excels in both accuracy and speed, fit for most of the use case in application. It's licensed under Apache 2.0 so it can be used freely for commercial applications. It has variants that varies in speed and size to fit in with the environment.

| Variant        | Size (MB) | Speed (ms/image) |
| -------------- | --------- | ---------------- |
| RF-DETR Nano   | ~15 MB    | 2.3              |
| RF-DETR Small  | ~35 MB    | 3.5              |
| RF-DETR Base   | ~45 MB    | 4.5              |
| RF-DETR Large  | ~300 MB   | 6.5              |

RF-DETR nano fits for integration in edge devices, mobile apps and real-time application where speed is crucial and low-memory is required.
[RF-DETR-nano](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetr-nano.py)

RF-DETR small is slightly bigger but still it's fast and good fit for real-time applications and performs best on GPUs.
[RF-DETR-small](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetr-small.py)

RF-DETR base is ideal for server inferences for real-time application deployments.
[RF-DETR-base](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetrbase.py)

RF-DETR Large is heavy-weight model best for high accuracies on GPUs. Ideal to use where accuracy is more crucial than speed. Not ideal for real-time systems.
[RF-DETR-large](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/rfdetrlarge.py)

For training RF-DETR on custom dataset first preprocess the dataset using Roboflow and then download it according to the RF-DETR format by selecting format in Roboflow then feed it into the code and then start training by defining which model you want to train. Let's have a hands on example of it at all.
[FINE-TUNE-RF-DETR](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/train_rfdetr.py)

**2: YOLO** by [Ultralytics](https://www.ultralytics.com/) commonly used model throughout best fit for real-time applications and fast easy to finetune. It takes image of 640x640 pixels as standard input. But it's not under Apache 2.0 license so it can't be used freely for commercial applications. You have to pay to the company.

| Variant       | Size (MB) | Speed (ms/image) |
| ------------- | --------- | ---------------- |
| YOLO12 Nano   | ~14 MB    | 2.5              |
| YOLO12 Small  | ~27 MB    | 3.8              |
| YOLO12 Medium | ~44 MB    | 5.0              |
| YOLO12 Large  | ~89 MB    | 8.0              |

YOLO12n is ultralight and it's optimized for edge devices and real-time inferences can be used in applications where speed is required and hardware is small.
[yolo12n_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12n_.py)

YOLO12s is balanced with speed and accuracy performs well comparatively nano variant when integrated on the GPU based hardware.
[yolo12s_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12s_.py)

YOLO12m has significant accuracy difference from smaller ones and moderate speed ideal when deployed on server based inferences.
[yolo12m_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12m_.py)

YOLO12Large is high-speed model best for where precision is crucial more than speed. Mainly for medical imaging systems.
[yolo12l_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12l_.py)

[Fine-tuning YOLO](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/training.py) - These are the simple implementations of the YOLO model variants for object detection tasks.

**3: Fast R-CNN** by Microsoft Research is a two-stage detector object detection known for its precision and high accuracy. It's slightly slower than other single-stage detectors. Two-stage detector means first it processes ROI (region of interests) in the image and then classifies and refines bounding boxes for each region this process reduces the false positive and overlapping of objects. That's why mostly it's used where speed and accuracy both are required and mainly it can be seen deployed on medical imaging systems. And its variants are mainly the backbones it uses like CNNs layers (ResNet-50, ResNet-101, MobileNet) which cause difference in speed and accuracy.

| Variant        | Backbone    | Size (MB) | Speed (ms/image) |
| -------------- | ----------- | --------- | ---------------- |
| Fast R-CNN 50  | ResNet-50   | ~120 MB   | 30               |
| Fast R-CNN 101 | ResNet-101  | ~180 MB   | 45               |
| Fast R-CNN M   | MobileNetV2 | ~60 MB    | 20               |

ResNet-50: This backbone is balanced for speed and accuracy so where both are crucial then this would be ideal fit and commonly from Fast R-CNN this backbone is commonly used.
[FastR-CNN-ResNet50](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_resnet50.py)

ResNet-101: This has higher accuracy and slower inference so it should be integrated on precision mandatory applications.
[FastR-CNN-ResNet101](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_resnet101.py)

MobileNet: This variant is again lightweight faster but accuracy is compromised so not so ideal.
[FastR-CNN_MobileNet](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_mobile.py)

These are the mostly used object detection models for commercial enterprise applications, research works and medical analysis. And all of them have multiple use case centric variants having specialization for the specific task. We have discussed them and now just we will make a list of all the possible open source object detection models that are available for integration in production grade applications, research and development etc.

---

## Hugging Face

**AI platform:** [Hugging Face](https://huggingface.co/) is AI platform that provides tools, datasets and pre-trained models for Machine learning tasks. It has its wide transformer library that offer multiple ready to use open source models. It's called models zoo where you can get any type of model for GenAI, Machine learning, Computer vision and Natural language processing etc.

One of its most powerful feature is it provides inference API which allows to run models in cloud without setting up local environment, just using API for sending request and all the computation will be handled by Hugging Face. There are two ways to use it: 1) free API good for testing and personal use and 2) paid plan for large applications and faster responses.

Example to use Hugging Face API for inference:
```python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

output = client.image_segmentation("cats.jpg", model="facebook/mask2former-swin-base-coco-panoptic")
```

### Transformers

**Understanding transformers:** Transformers are type of deep learning architectures designed to handle sequential data using self-attention mechanisms instead of traditional or convolution. They excel at capturing long-range dependencies in data. Unlike older approaches that process sequences step by step, transformers compute relationships between all elements in a sequence simultaneously, allowing them to capture long-range dependencies. For our context we have to focus on ViTs (Vision Transformers).

**Computer Vision Transformers**: They adapt this architecture to computer vision by splitting an image into small patches, treating each patch like word in a sentence and applying same attention mechanism to learn how different parts of the image relate to each other. Mainly used for image-to-text, text-to-image transformers for generating captions and images.

**ViT (Vision Transformer)**: The first pure transformer for image classification, treating images as sequence of patches not as pixels.
[ViTs](https://huggingface.co/google/vit-base-patch16-224-in21k); Hugging-Face.

**Swin-Transformer**: It uses shifted window attention mechanism for efficient scaling to high-resolution images. It excels in segmentation, detection and classification.
[swin](https://huggingface.co/keras-io/swin-transformers)

**BLIP/BLIP-2**: A Vision language model for tasks like image captioning, VQA (Visual Question Answering) and retrieval. It takes images as input and generate its caption by defining what's happening inside the image. BLIP-2 improves the efficiency by using pre-trained language models for better reasoning over visual inputs. Patches understanding goes to language models and then they generate accurate caption.
[Blip](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)

**Florence**: Large scale vision foundation model for various multimodal vision-language applications. It supports tasks such as image-text matching, captioning in enterprise and real-world production grade deployments.
[florence](https://huggingface.co/microsoft/Florence-2-base)

**Note**: These models like ViT, Swin-Transformer, BLIP/BLIP-2, and Florence are not ideal for real-time object detection on RTSP streams. They are mainly designed for high-accuracy image classification, vision-language tasks, and image captioning. These models typically require high-end GPUs with substantial memory (≥16 GB VRAM) for inference and fine-tuning, and are generally unsuitable for CPU-only or edge deployments.

### Models from Hugging Face

**Models for Object Detection with High Speed:**

[Object detection models on Hugging Face](https://huggingface.co/models?pipeline_tag=object-detection&sort=trending)

- **YOLOv4**
  Balanced speed and accuracy; highly optimized for real-time detection tasks.
  **Speed:** ~65 FPS (V100)
  **Accuracy:** ~43.5% AP (COCO-dataset)
  [Yolov4Tiny](https://huggingface.co/gbahlnxp/yolov4tiny)

  **YOLO-S-Tiny**
  [yolos-tiny](https://huggingface.co/hustvl/yolos-tiny)

- **YOLOv7**
  State-of-the-art real-time detection model with top-tier accuracy.
  **Speed:** 30–160 FPS
  **Accuracy:** ~56.8% AP (30+ FPS)
  [Yolov7](https://huggingface.co/kadirnar/yolov7-tiny-v0.1)

- **SSD (Single-Shot Detector)**
  Lightweight single-stage detector suitable for real-time applications.
  **Speed:** ~58 FPS
  **Accuracy:** ~72.1% (Pascal VOC)

- **EfficientDet (D0–D7)**
  Scalable and efficient detectors with excellent COCO performance.
  **Speed:** 30–50 FPS (varies by variant)
  **Accuracy:** Up to ~55.1% AP (COCO)
  [EfficientNet](https://huggingface.co/google/efficientnet-b7)

- **RetinaNet**
  One-stage detector with Focal Loss to handle class imbalance effectively.
  **Speed:** ~30 FPS
  **Accuracy:** High
  [RetinaNet](https://huggingface.co/keras-io/Object-Detection-RetinaNet)

- **RT-DETR (R50)**
  Real-Time DETR optimized for fast inference.
  **Speed:** 35 FPS
  **Accuracy:** Good overall performance
  [RT-DETR](https://huggingface.co/PekingU/rtdetr_r101vd_coco_o365)

---

### MLflow

**Experiment tracking platform:** MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects ensuring that each phase is manageable, traceable and reproducible. MLflow provides comprehensive support for traditional machine learning and deep learning workflows. From experiment tracking and model versioning to deployment and monitoring, MLflow streamlines every aspect of ML lifecycles. Whether you're working with scikit-learn models, training deep neural networks, or managing complex ML pipelines, MLflow provides the tools you need to build reliable, scalable machine learning systems.

**Core features**: MLflow Tracking provides comprehensive experiment logging, parameters tracking, metrics tracking, model versioning and artifact management.

- Experiment Organization: Track and compare multiple models experiments
- Metric Visualization: Built-in plots and charts for model performance
- Artifact Storage: Store models, plots and other files each run
- Collaboration: Share experiments and models with team members

[MLflow implementations](https://github.com/aaliyanahmed1/ML-Guide/tree/main/MLFlow_)

---

## Deployment

**Model deployment:** Deployment is very typical part of every Machine learning workflow. When it comes to deployment maintaining FPS for real-time systems becomes nightmare of MLOps architects so that's why the universal way to deploy model and maintain performance is to decouple it from training framework, that simplifies and reduces down burden of heavy dependencies and speed up the process is exporting model in ONNX (Open Neural Network Exchange) format. This simplifies integration of model and makes it compatible.

### ONNX

**Model format:** It is an open standard format for representing machine learning models. Exporting models to ONNX decouples them from the original training framework, making them easier to integrate into different platforms, whether on a server, multiple edge devices, or in the cloud. It ensures compatibility across various tools and allows optimized inference on different hardware setups, helping maintain real-time performance.

### ONNX Runtime

**Inference engine:** It is a high-performance inference engine designed to run ONNX models efficiently across different platforms. It takes the ONNX model and applies graph optimization, operator fusion and quantizations to reduce memory usage and computation time. So models run faster on servers, cloud environments and on multiple edge devices without needing original training framework. It can also speed up training process of large models by just making simple changes in code it can make training faster and efficient without changing workflow too much.

[ONNX Runtime Docs](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime)

[ONNX Runtime for Training](https://onnxruntime.ai/training)

---

### Model Export to ONNX

**Converting models to ONNX:** After training your model in PyTorch or TensorFlow, export it to ONNX format for deployment. This makes model framework-independent and ready for production deployment on any platform.

**File: `deployment/export_to_onnx.py`**
```python
"""
Model Export to ONNX Format
----------------------------
Export trained models from PyTorch/TensorFlow to ONNX format.
Essential for cross-platform deployment and optimization.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np


def export_pytorch_to_onnx(model, dummy_input, onnx_path, input_names=['input'], output_names=['output']):
    """
    Export PyTorch model to ONNX format.

    Why export to ONNX:
    - Framework independent deployment
    - Optimized inference on different hardware
    - Compatible with ONNX Runtime, TensorRT, OpenVINO
    - Smaller model size with optimizations

    Args:
        model: Trained PyTorch model
        dummy_input: Sample input tensor matching model's expected input
        onnx_path: Path to save ONNX model
        input_names: Names for input nodes
        output_names: Names for output nodes

    Example:
        model = torch.load('yolo_model.pt')
        dummy_input = torch.randn(1, 3, 640, 640)
        export_pytorch_to_onnx(model, dummy_input, 'model.onnx')
    """
    # Set model to evaluation mode
    model.eval()

    print(f"Exporting PyTorch model to ONNX...")

    # Export model to ONNX
    torch.onnx.export(
        model,                      # Model to export
        dummy_input,                # Sample input
        onnx_path,                  # Output path
        export_params=True,         # Store trained parameters
        opset_version=12,          # ONNX version
        do_constant_folding=True,   # Optimize constant folding
        input_names=input_names,    # Input tensor names
        output_names=output_names,  # Output tensor names
        dynamic_axes={              # Dynamic batch size
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
    )

    print(f"✓ Model exported to {onnx_path}")

    # Verify exported model
    verify_onnx_model(onnx_path)


def export_tensorflow_to_onnx(model, onnx_path, input_shape=(1, 640, 640, 3)):
    """
    Export TensorFlow model to ONNX format.

    Requires: tf2onnx library
    Install: pip install tf2onnx

    Args:
        model: Trained TensorFlow/Keras model
        onnx_path: Path to save ONNX model
        input_shape: Input tensor shape

    Example:
        model = tf.keras.models.load_model('model.h5')
        export_tensorflow_to_onnx(model, 'model.onnx')
    """
    import tf2onnx
    import tensorflow as tf

    print(f"Exporting TensorFlow model to ONNX...")

    # Convert model
    spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
    output_path = onnx_path

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=12,
        output_path=output_path
    )

    print(f"✓ Model exported to {onnx_path}")

    # Verify exported model
    verify_onnx_model(onnx_path)


def verify_onnx_model(onnx_path):
    """
    Verify exported ONNX model is valid.

    Checks:
    - Model structure is valid
    - Can be loaded by ONNX Runtime
    - Inference works correctly

    Args:
        onnx_path: Path to ONNX model file
    """
    print(f"\nVerifying ONNX model...")

    # Load and check model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model structure is valid")

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Print input/output info
    print("\nModel Inputs:")
    for input in session.get_inputs():
        print(f"  - {input.name}: {input.shape} ({input.type})")

    print("\nModel Outputs:")
    for output in session.get_outputs():
        print(f"  - {output.name}: {output.shape} ({output.type})")

    print("\n✓ Model verification complete")


def optimize_onnx_model(onnx_path, optimized_path):
    """
    Optimize ONNX model for faster inference.

    Optimizations applied:
    - Constant folding
    - Redundant node elimination
    - Operator fusion
    - Shape inference

    Args:
        onnx_path: Path to original ONNX model
        optimized_path: Path to save optimized model
    """
    from onnxruntime.transformers import optimizer

    print(f"Optimizing ONNX model...")

    # Load model
    model = onnx.load(onnx_path)

    # Apply optimizations
    from onnx import optimizer as onnx_optimizer

    passes = [
        'eliminate_nop_transpose',
        'eliminate_nop_pad',
        'fuse_consecutive_transposes',
        'fuse_transpose_into_gemm',
    ]

    optimized_model = onnx_optimizer.optimize(model, passes)

    # Save optimized model
    onnx.save(optimized_model, optimized_path)

    # Compare sizes
    import os
    original_size = os.path.getsize(onnx_path) / (1024 * 1024)
    optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)

    print(f"✓ Optimization complete")
    print(f"Original size: {original_size:.2f} MB")
    print(f"Optimized size: {optimized_size:.2f} MB")
    print(f"Size reduction: {((original_size - optimized_size) / original_size * 100):.1f}%")


# Example usage
if __name__ == "__main__":
    # PyTorch export example
    print("="*50)
    print("PyTorch to ONNX Export")
    print("="*50)

    # Load PyTorch model (example)
    # model = torch.load('trained_model.pt')
    # dummy_input = torch.randn(1, 3, 640, 640)
    # export_pytorch_to_onnx(model, dummy_input, 'model.onnx')

    # TensorFlow export example
    print("\n" + "="*50)
    print("TensorFlow to ONNX Export")
    print("="*50)

    # Load TensorFlow model (example)
    # import tensorflow as tf
    # model = tf.keras.models.load_model('trained_model.h5')
    # export_tensorflow_to_onnx(model, 'model.onnx')

    # Optimize ONNX model
    # optimize_onnx_model('model.onnx', 'model_optimized.onnx')
```

---

### Real-Time Inference

**Running inference on live streams:** Deploy model for real-time object detection on webcam, RTSP streams, or video files. Critical for surveillance systems, autonomous vehicles, and live monitoring applications.

**File: `deployment/real_time_inference.py`**
```python
"""
Real-Time Object Detection Inference
-------------------------------------
Run object detection on webcam, RTSP streams, and video files.
Optimized for real-time performance with FPS tracking.
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
from collections import deque


class RealTimeDetector:
    """
    Real-time object detection using ONNX models.

    Supports:
    - Webcam inference
    - RTSP stream inference
    - Video file inference
    - FPS calculation and display
    """

    def __init__(self, onnx_model_path, conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize real-time detector.

        Args:
            onnx_model_path: Path to ONNX model file
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Load ONNX model
        print(f"Loading model from {onnx_model_path}...")
        self.session = ort.InferenceSession(onnx_model_path)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # Get input shape
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

        print(f"✓ Model loaded successfully")
        print(f"Input shape: {self.input_shape}")

        # FPS calculation
        self.fps_queue = deque(maxlen=30)
        self.prev_time = time.time()


    def preprocess(self, frame):
        """
        Preprocess frame for model input.

        Steps:
        - Resize to model input size
        - Normalize pixel values
        - Convert to correct format (NCHW)

        Args:
            frame: Input frame (numpy array)

        Returns:
            Preprocessed input tensor
        """
        # Resize frame
        resized = cv2.resize(frame, (self.input_width, self.input_height))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to NCHW format (batch, channels, height, width)
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        input_tensor = np.expand_dims(transposed, axis=0)

        return input_tensor


    def postprocess(self, outputs, original_shape):
        """
        Post-process model outputs to get final detections.

        Steps:
        - Apply confidence threshold
        - Apply NMS (Non-Maximum Suppression)
        - Scale boxes to original image size

        Args:
            outputs: Raw model outputs
            original_shape: Original frame shape (height, width)

        Returns:
            List of detections: [x1, y1, x2, y2, confidence, class_id]
        """
        # Extract predictions
        predictions = outputs[0][0]

        # Filter by confidence
        mask = predictions[:, 4] > self.conf_threshold
        filtered = predictions[mask]

        if len(filtered) == 0:
            return []

        # Extract boxes, scores, class_ids
        boxes = filtered[:, :4]
        scores = filtered[:, 4]
        class_ids = filtered[:, 5:].argmax(axis=1)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.iou_threshold
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                box = boxes[i]
                score = scores[i]
                class_id = int(class_ids[i])

                # Scale boxes to original image size
                h, w = original_shape
                x1 = int(box[0] * w / self.input_width)
                y1 = int(box[1] * h / self.input_height)
                x2 = int(box[2] * w / self.input_width)
                y2 = int(box[3] * h / self.input_height)

                detections.append([x1, y1, x2, y2, score, class_id])

        return detections


    def draw_detections(self, frame, detections, class_names=None):
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of detections
            class_names: List of class names (optional)

        Returns:
            Frame with drawn detections
        """
        for detection in detections:
            x1, y1, x2, y2, score, class_id = detection

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label
            if class_names and class_id < len(class_names):
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"

            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        return frame


    def calculate_fps(self):
        """
        Calculate current FPS.

        Returns:
            Current FPS value
        """
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        self.fps_queue.append(fps)

        return sum(self.fps_queue) / len(self.fps_queue)


    def run_webcam(self, camera_id=0, class_names=None):
        """
        Run inference on webcam feed.

        Args:
            camera_id: Camera device ID (default: 0)
            class_names: List of class names for labels

        Press 'q' to quit.
        """
        print(f"Starting webcam inference (Camera {camera_id})...")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess
            input_tensor = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Post-process
            detections = self.postprocess(outputs, frame.shape[:2])

            # Draw detections
            frame = self.draw_detections(frame, detections, class_names)

            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Display frame
            cv2.imshow('Object Detection', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✓ Webcam inference stopped")


    def run_rtsp_stream(self, rtsp_url, class_names=None):
        """
        Run inference on RTSP stream.

        Args:
            rtsp_url: RTSP stream URL
            class_names: List of class names for labels

        Example:
            rtsp_url = "rtsp://username:password@ip:port/stream"

        Press 'q' to quit.
        """
        print(f"Connecting to RTSP stream...")
        print(f"URL: {rtsp_url}")
        print("Press 'q' to quit")

        cap = cv2.VideoCapture(rtsp_url)

        if not cap.isOpened():
            print(f"Error: Cannot connect to RTSP stream")
            return

        print("✓ Connected to stream")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Connection lost. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(rtsp_url)
                continue

            # Preprocess
            input_tensor = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Post-process
            detections = self.postprocess(outputs, frame.shape[:2])

            # Draw detections
            frame = self.draw_detections(frame, detections, class_names)

            # Calculate and display FPS
            fps = self.calculate_fps()
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )

            # Display frame
            cv2.imshow('RTSP Stream Detection', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("✓ RTSP stream inference stopped")


    def run_video_file(self, video_path, output_path=None, class_names=None):
        """
        Run inference on video file.

        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            class_names: List of class names for labels

        Press 'q' to quit early.
        """
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Cannot open video file")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height} @ {fps} FPS, {total_frames} frames")

        # Setup output video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Preprocess
            input_tensor = self.preprocess(frame)

            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

            # Post-process
            detections = self.postprocess(outputs, frame.shape[:2])

            # Draw detections
            frame = self.draw_detections(frame, detections, class_names)

            # Calculate and display FPS
            fps_current = self.calculate_fps()
            progress = (frame_count / total_frames) * 100

            info_text = f"FPS: {fps_current:.1f} | Progress: {progress:.1f}% ({frame_count}/{total_frames})"
            cv2.putText(
                frame, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            # Save frame to output video
            if writer:
                writer.write(frame)

            # Display frame
            cv2.imshow('Video Processing', frame)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing interrupted by user")
                break

            # Print progress
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"✓ Video processing complete")
        if output_path:
            print(f"✓ Output saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = RealTimeDetector(
        onnx_model_path="model.onnx",
        conf_threshold=0.5,
        iou_threshold=0.4
    )

    # Class names (example)
    class_names = ['person', 'car', 'bike', 'bus', 'truck']

    # Run on webcam
    # detector.run_webcam(camera_id=0, class_names=class_names)

    # Run on RTSP stream
    # rtsp_url = "rtsp://username:password@192.168.1.100:554/stream"
    # detector.run_rtsp_stream(rtsp_url, class_names=class_names)

    # Run on video file
    # detector.run_video_file(
    #     video_path="input_video.mp4",
    #     output_path="output_video.mp4",
    #     class_names=class_names
    # )
```

---

### Model Optimization

**Optimizing models for deployment:** Reduce model size and increase inference speed using quantization, pruning, and knowledge distillation. Critical for deploying on edge devices with limited resources.

**File: `optimization/model_optimization.py`**
```python
"""
Model Optimization Techniques
------------------------------
Quantization, pruning, and optimization for faster inference.
Essential for edge device deployment and reducing latency.
"""

import torch
import torch.nn as nn
import numpy as np


def quantize_pytorch_model(model, calibration_data_loader, quantized_model_path):
    """
    Quantize PyTorch model from FP32 to INT8.

    Why quantize:
    - 4x smaller model size
    - 2-4x faster inference
    - Lower memory usage
    - Suitable for edge devices

    Args:
        model: Trained PyTorch model (FP32)
        calibration_data_loader: DataLoader with calibration data
        quantized_model_path: Path to save quantized model

    Example:
        Quantization reduces model from 100MB to 25MB
        and increases inference speed from 50ms to 15ms
    """
    print("Quantizing PyTorch model to INT8...")

    # Set model to evaluation mode
    model.eval()

    # Fuse modules (Conv + BN + ReLU)
    model = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])

    # Specify quantization configuration
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    torch.quantization.prepare(model, inplace=True)

    # Calibrate with representative dataset
    print("Calibrating model...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(calibration_data_loader):
            model(data)
            if batch_idx >= 100:  # Use 100 batches for calibration
                break

    # Convert to quantized model
    torch.quantization.convert(model, inplace=True)

    # Save quantized model
    torch.save(model.state_dict(), quantized_model_path)

    print(f"✓ Quantized model saved to {quantized_model_path}")

    # Compare model sizes
    import os
    original_size = os.path.getsize('original_model.pt') / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_model_path) / (1024 * 1024)

    print(f"\nModel Size Comparison:")
    print(f"Original (FP32): {original_size:.2f} MB")
    print(f"Quantized (INT8): {quantized_size:.2f} MB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")


def quantize_onnx_model(onnx_model_path, quantized_onnx_path, calibration_data):
    """
    Quantize ONNX model to INT8.

    Args:
        onnx_model_path: Path to original ONNX model
        quantized_onnx_path: Path to save quantized model
        calibration_data: Numpy array of calibration data

    Example:
        calibration_data = np.random.randn(100, 3, 640, 640).astype(np.float32)
        quantize_onnx_model('model.onnx', 'model_int8.onnx', calibration_data)
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("Quantizing ONNX model to INT8...")

    # Dynamic quantization (no calibration data needed)
    quantize_dynamic(
        onnx_model_path,
        quantized_onnx_path,
        weight_type=QuantType.QUInt8
    )

    print(f"✓ Quantized model saved to {quantized_onnx_path}")

    # Compare model sizes
    import os
    original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(quantized_onnx_path) / (1024 * 1024)

    print(f"\nModel Size Comparison:")
    print(f"Original: {original_size:.2f} MB")
    print(f"Quantized: {quantized_size:.2f} MB")
    print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.1f}%")


def prune_pytorch_model(model, amount=0.3):
    """
    Prune PyTorch model by removing less important weights.

    Why prune:
    - Reduces model size
    - Faster inference
    - Lower memory usage
    - Can be combined with quantization

    Args:
        model: PyTorch model to prune
        amount: Fraction of weights to prune (0-1)
                0.3 = remove 30% of weights

    Returns:
        Pruned model
    """
    import torch.nn.utils.prune as prune

    print(f"Pruning model (removing {amount*100:.1f}% of weights)...")

    # Prune all Conv2d and Linear layers
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))

    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)

    print(f"✓ Model pruned successfully")

    # Calculate sparsity
    total_params = 0
    zero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params * 100
    print(f"Model sparsity: {sparsity:.1f}% (zeros: {zero_params:,} / {total_params:,})")

    return model


def benchmark_model_speed(model, input_shape=(1, 3, 640, 640), num_iterations=100, device='cuda'):
    """
    Benchmark model inference speed.

    Measures:
    - Average inference time
    - FPS (frames per second)
    - Throughput

    Args:
        model: Model to benchmark (PyTorch or ONNX session)
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary with benchmark results
    """
    import time
    import torch

    print(f"\nBenchmarking model on {device.upper()}...")
    print(f"Input shape: {input_shape}")
    print(f"Iterations: {num_iterations}")

    # Create dummy input
    if device == 'cuda':
        dummy_input = torch.randn(input_shape).cuda()
        model = model.cuda()
    else:
        dummy_input = torch.randn(input_shape)

    model.eval()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Benchmark
    print("Running benchmark...")
    times = []

    with torch.no_grad():
        for i in range(num_iterations):
            if device == 'cuda':
                torch.cuda.synchronize()

            start_time = time.time()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{num_iterations}")

    # Calculate statistics
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1000 / avg_time

    # Print results
    print("\n" + "="*50)
    print("Benchmark Results")
    print("="*50)
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Std deviation: {std_time:.2f} ms")
    print(f"Min time: {min_time:.2f} ms")
    print(f"Max time: {max_time:.2f} ms")
    print(f"FPS: {fps:.1f}")
    print("="*50)

    return {
        'avg_time_ms': avg_time,
        'std_time_ms': std_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps
    }


# Example usage
if __name__ == "__main__":
    # Quantization example
    print("="*50)
    print("Model Quantization")
    print("="*50)

    # model = torch.load('model.pt')
    # calibration_loader = ... # Your data loader
    # quantize_pytorch_model(model, calibration_loader, 'model_int8.pt')

    # ONNX quantization
    # quantize_onnx_model('model.onnx', 'model_int8.onnx', calibration_data)

    # Pruning example
    print("\n" + "="*50)
    print("Model Pruning")
    print("="*50)

    # model = torch.load('model.pt')
    # pruned_model = prune_pytorch_model(model, amount=0.3)
    # torch.save(pruned_model, 'model_pruned.pt')

    # Benchmark example
    print("\n" + "="*50)
    print("Speed Benchmark")
    print("="*50)

    # model = torch.load('model.pt')
    # results = benchmark_model_speed(model, device='cuda')
```

---

### Model Serving with API

**Deploying model as REST API:** Serve your trained model as REST API using FastAPI. Allows easy integration with web applications, mobile apps, and other services.

**File: `deployment/api_server.py`**
```python
"""
Model Serving with FastAPI
---------------------------
Deploy object detection model as REST API.
Supports image upload, batch processing, and real-time inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import cv2
import numpy as np
import onnxruntime as ort
import io
from PIL import Image


# Initialize FastAPI app
app = FastAPI(
    title="Object Detection API",
    description="REST API for object detection using ONNX models",
    version="1.0.0"
)


# Global model session
model_session = None
INPUT_SIZE = (640, 640)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.4


class DetectionResult(BaseModel):
    """Detection result schema"""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: Optional[str] = None


class PredictionResponse(BaseModel):
    """API response schema"""
    detections: List[DetectionResult]
    inference_time_ms: float
    image_size: List[int]  # [width, height]


@app.on_event("startup")
async def load_model():
    """
    Load ONNX model on startup.

    This runs once when server starts.
    Model stays loaded in memory for fast inference.
    """
    global model_session

    model_path = "model.onnx"  # Change to your model path

    print(f"Loading model from {model_path}...")
    model_session = ort.InferenceSession(model_path)
    print("✓ Model loaded successfully")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for model input.

    Args:
        image: Input image (BGR format)

    Returns:
        Preprocessed tensor ready for inference
    """
    # Resize
    resized = cv2.resize(image, INPUT_SIZE)

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize
    normalized = rgb.astype(np.float32) / 255.0

    # Transpose to NCHW
    transposed = normalized.transpose(2, 0, 1)

    # Add batch dimension
    input_tensor = np.expand_dims(transposed, axis=0)

    return input_tensor


def postprocess_predictions(outputs, original_shape, class_names=None):
    """
    Post-process model outputs.

    Args:
        outputs: Raw model outputs
        original_shape: Original image shape (height, width)
        class_names: List of class names (optional)

    Returns:
        List of DetectionResult objects
    """
    predictions = outputs[0][0]

    # Filter by confidence
    mask = predictions[:, 4] > CONFIDENCE_THRESHOLD
    filtered = predictions[mask]

    if len(filtered) == 0:
        return []

    # Extract boxes, scores, class_ids
    boxes = filtered[:, :4]
    scores = filtered[:, 4]
    class_ids = filtered[:, 5:].argmax(axis=1)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        CONFIDENCE_THRESHOLD,
        IOU_THRESHOLD
    )

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            score = float(scores[i])
            class_id = int(class_ids[i])

            # Scale to original size
            h, w = original_shape
            x1 = float(box[0] * w / INPUT_SIZE[0])
            y1 = float(box[1] * h / INPUT_SIZE[1])
            x2 = float(box[2] * w / INPUT_SIZE[0])
            y2 = float(box[3] * h / INPUT_SIZE[1])

            # Get class name if provided
            class_name = None
            if class_names and class_id < len(class_names):
                class_name = class_names[class_id]

            detection = DetectionResult(
                bbox=[x1, y1, x2, y2],
                confidence=score,
                class_id=class_id,
                class_name=class_name
            )
            detections.append(detection)

    return detections


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload image for detection",
            "/predict/batch": "POST - Batch image detection",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns model status and system info.
    """
    return {
        "status": "healthy",
        "model_loaded": model_session is not None,
        "input_size": INPUT_SIZE,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), class_names: Optional[List[str]] = None):
    """
    Run object detection on uploaded image.

    Args:
        file: Image file (JPEG, PNG)
        class_names: Optional list of class names

    Returns:
        Detection results with bounding boxes, confidence scores

    Example:
        curl -X POST "http://localhost:8000/predict" \
             -F "file=@image.jpg"
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        original_shape = image.shape[:2]

        # Preprocess
        input_tensor = preprocess_image(image)

        # Run inference
        import time
        start_time = time.time()

        input_name = model_session.get_inputs()[0].name
        output_names = [output.name for output in model_session.get_outputs()]
        outputs = model_session.run(output_names, {input_name: input_tensor})

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Post-process
        detections = postprocess_predictions(outputs, original_shape, class_names)

        # Prepare response
        response = PredictionResponse(
            detections=detections,
            inference_time_ms=round(inference_time, 2),
            image_size=[image.shape[1], image.shape[0]]
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Run detection on multiple images.

    Args:
        files: List of image files

    Returns:
        List of detection results for each image

    Example:
        curl -X POST "http://localhost:8000/predict/batch" \
             -F "files=@image1.jpg" \
             -F "files=@image2.jpg"
    """
    if model_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []

    for file in files:
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue

            original_shape = image.shape[:2]

            # Preprocess
            input_tensor = preprocess_image(image)

            # Run inference
            import time
            start_time = time.time()

            input_name = model_session.get_inputs()[0].name
            output_names = [output.name for output in model_session.get_outputs()]
            outputs = model_session.run(output_names, {input_name: input_tensor})

            inference_time = (time.time() - start_time) * 1000

            # Post-process
            detections = postprocess_predictions(outputs, original_shape)

            results.append({
                "filename": file.filename,
                "detections": [det.dict() for det in detections],
                "inference_time_ms": round(inference_time, 2)
            })

        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })

    return JSONResponse(content={"results": results})


if __name__ == "__main__":
    # Run server
    print("Starting Object Detection API Server...")
    print("API Docs: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
```

**Client example to test API:**

**File: `deployment/api_client.py`**
```python
"""
API Client Example
------------------
Test object detection API with images.
"""

import requests
import json


def test_single_image(image_path, api_url="http://localhost:8000/predict"):
    """
    Test API with single image.

    Args:
        image_path: Path to image file
        api_url: API endpoint URL
    """
    print(f"Testing API with: {image_path}")

    # Open image file
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(api_url, files=files)

    if response.status_code == 200:
        result = response.json()
        print(f"\n✓ Detection successful!")
        print(f"Inference time: {result['inference_time_ms']:.2f} ms")
        print(f"Detections: {len(result['detections'])}")

        for idx, det in enumerate(result['detections']):
            print(f"\nDetection {idx + 1}:")
            print(f"  Class ID: {det['class_id']}")
            print(f"  Confidence: {det['confidence']:.3f}")
            print(f"  Bbox: {det['bbox']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def test_batch_images(image_paths, api_url="http://localhost:8000/predict/batch"):
    """
    Test API with multiple images.

    Args:
        image_paths: List of image file paths
        api_url: API endpoint URL
    """
    print(f"Testing API with {len(image_paths)} images")

    files = [('files', open(path, 'rb')) for path in image_paths]

    response = requests.post(api_url, files=files)

    # Close files
    for _, f in files:
        f.close()

    if response.status_code == 200:
        results = response.json()['results']
        print(f"\n✓ Batch processing successful!")

        for result in results:
            print(f"\nFile: {result['filename']}")
            if 'error' in result:
                print(f"  Error: {result['error']}")
            else:
                print(f"  Detections: {len(result['detections'])}")
                print(f"  Time: {result['inference_time_ms']:.2f} ms")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    # Test single image
    test_single_image("test_image.jpg")

    # Test batch
    # test_batch_images(["image1.jpg", "image2.jpg", "image3.jpg"])
```

---

### Common Issues & Troubleshooting

**Resolving common errors:** When working with object detection models, you'll encounter various issues. Here are solutions to most common problems.

**1: CUDA Out of Memory Error**

```python
"""
Problem: RuntimeError: CUDA out of memory

Solution: Reduce batch size or use gradient accumulation
"""

# Bad - Large batch size
batch_size = 32  # Too large, causes OOM

# Good - Smaller batch with gradient accumulation
batch_size = 8
accumulation_steps = 4  # Effective batch size = 8 * 4 = 32

optimizer.zero_grad()
for i, (images, targets) in enumerate(train_loader):
    loss = model(images, targets)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**2: CUDA Not Available**

```python
"""
Problem: torch.cuda.is_available() returns False

Solutions:
"""

# Check CUDA installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# If False, reinstall PyTorch with CUDA:
# pip uninstall torch torchvision
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Or use CPU fallback
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
```

**3: Slow Training Speed**

```python
"""
Problem: Training is very slow

Solutions:
"""

# 1. Use mixed precision training (FP16)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, targets in train_loader:
    optimizer.zero_grad()

    with autocast():  # Use FP16
        loss = model(images, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 2. Use multiple workers for data loading
train_loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,  # Use multiple CPU cores
    pin_memory=True  # Faster data transfer to GPU
)

# 3. Use DataParallel for multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

**4: Model Not Learning (Loss Not Decreasing)**

```python
"""
Problem: Training loss stays constant or increases

Solutions:
"""

# 1. Check learning rate
learning_rate = 0.001  # Try different values: 0.0001, 0.01

# 2. Use learning rate scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    scheduler.step(train_loss)

# 3. Check data normalization
# Images should be normalized to [0, 1] or [-1, 1]
images = images / 255.0

# 4. Verify labels are correct
print(f"Sample labels: {targets[:5]}")
print(f"Label range: min={targets.min()}, max={targets.max()}")
```

**5: Poor mAP After Training**

```python
"""
Problem: Model achieves low mAP on validation set

Solutions:
"""

# 1. Train longer
epochs = 100  # Instead of 50

# 2. Use data augmentation
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomRotation(degrees=10),
])

# 3. Adjust confidence and IoU thresholds
conf_threshold = 0.25  # Lower threshold finds more objects
iou_threshold = 0.45   # Adjust based on validation results

# 4. Check for class imbalance
from collections import Counter
class_counts = Counter([label for labels in dataset for label in labels])
print(f"Class distribution: {class_counts}")

# Use weighted loss for imbalanced classes
class_weights = compute_class_weights(class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

**6: Inference Too Slow**

```python
"""
Problem: Model inference is too slow for real-time use

Solutions:
"""

# 1. Use model.eval() and torch.no_grad()
model.eval()
with torch.no_grad():
    predictions = model(image)

# 2. Reduce input size
input_size = (416, 416)  # Instead of (640, 640)

# 3. Use TensorRT or ONNX Runtime
# Convert to ONNX first, then use ONNX Runtime
session = ort.InferenceSession("model.onnx")

# 4. Batch multiple images together
batch = torch.stack([img1, img2, img3, img4])
predictions = model(batch)  # Process 4 images at once
```

**7: OpenCV Video Capture Not Working**

```python
"""
Problem: cv2.VideoCapture() fails to open camera/stream

Solutions:
"""

# 1. Check camera index
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        cap.release()

# 2. For RTSP streams, add parameters
rtsp_url = "rtsp://username:password@ip:port/stream"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency

# 3. Check video file path
import os
if not os.path.exists(video_path):
    print(f"Error: File not found: {video_path}")
```

---

### Best Practices Summary

**Production deployment checklist:** Follow these guidelines for deploying object detection models in production.

**Performance:**
- Always use model.eval() during inference
- Use torch.no_grad() to reduce memory usage
- Batch images when possible for higher throughput
- Use GPU for training, consider CPU for edge deployment
- Profile code to find bottlenecks

**Model Optimization:**
- Export to ONNX for framework-independent deployment
- Apply quantization for 4x size reduction
- Use pruning to remove unnecessary weights
- Benchmark on target hardware before deployment

**Data Quality:**
- Clean dataset before training (remove duplicates, corrupted images)
- Use train/val/test split (70/15/15)
- Apply data augmentation for better generalization
- Balance classes to avoid bias

**Monitoring:**
- Track FPS and latency in production
- Log failed predictions for debugging
- Monitor GPU/CPU usage and memory
- Set up alerts for performance degradation

**Security:**
- Validate all input images
- Set file size limits for API uploads
- Rate limit API requests
- Use authentication for production APIs
