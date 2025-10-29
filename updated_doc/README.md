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

Due to length constraints, I need to continue in the next message. Should I continue with the rest of the sections?