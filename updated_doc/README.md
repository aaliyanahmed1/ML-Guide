# Machine Learning Hands-On Guide

Hi, in this guide we are going to deeply learn about machine learning's fundamental concepts to core advanced frameworks, model selection, benchmarking, fine-tuning and custom training and so on. From prototyping to production level real-time deployments, from computer systems/servers to edge devices cross-platform deployments. This guide will cover all the topics that are essentially needed to become zero-to-hero in Machine learning. In this documentation we will be focusing mainly on the object detection branch that belongs to Deep learning.

AI is simply the ability to think and act like humans. There are its branches:
Machine Learning > Deep Learning > Computer Vision.

**Machine Learning**: A branch of AI in which machines learn from labeled data patterns instead of fixed rules-based systems.
Three main types of Machine Learning:

**1: Supervised Learning 2: Unsupervised Learning 3: Reinforcement Learning.**

**Deep Learning**: Its branch of machine learning that is based on the special type of neural networks that learns complex patterns from data. It mainly falls under supervised learning where models are trained on labeled datasets to learn the mapping between inputs and outputs.

One of the special branches of deep learning is Computer Vision that uses Convolutional Neural Networks to learn complex patterns from data and perform predictions in different environments efficiently in real-time deployments as well. In computer vision, images are processed to extract meaningful features, which can then be used for various tasks such as classification, segmentation, and one of the most important applications—object detection, where models not only recognize objects but also locate them within the image. In this guide we will deeply learn about object detection.

## Prerequisites

Before starting with this guide, you should have:
- Basic Python programming knowledge
- Understanding of basic math (linear algebra, calculus)
- Familiarity with command line operations
- Python 3.8+ installed
- GPU with CUDA support (recommended for training)
- At least 8GB RAM (16GB+ recommended for training)

## Environment Setup

First, set up your Python environment:

```bash
# Create virtual environment
python -m venv ml_env

# Activate environment
# Windows:
ml_env\Scripts\activate
# Linux/Mac:
source ml_env/bin/activate

# Install essential packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow opencv-python numpy pandas matplotlib
pip install onnx onnxruntime mlflow
```

## Object Detection

Object detection is a specific branch of computer vision that reads the images and detects the desired objects inside of the image along with their areas (x, y coordinates). Like if you input an image to the object detection model (YOLO, RF-DETR etc), it won't only tell that there is an object like cat/dog in the image, but also tells where they are by drawing bounding boxes on the areas and mentioning the class (name of object).

image input > model > inference > detection > output

![Object Detection](/images/onnxpy.png)

Like on this image we can see bounding boxes and the names of the objects like person and vehicle (bus) etc. That's the whole thing that is called object detection. It finds objects and labels objects within an image.

This was the verbal introduction of the object detection.
Now move towards all the technical details and requirements needed to develop, select, fine-tune and train the object detection model. From dataset collection to preprocessing.

## Object Detection Model Selection

When selecting the model for the object detection tasks first, we must thoroughly do research according to requirements and use case like where we want to deploy it and what we want to detect from it so this is the must check before moving forward towards data-preprocessing and fine-tuning. For example, you are developing a real-time security surveillance application that needs to detect events from the images under 50ms. Then model's speed shouldn't be compromised at any cost.

On the other hand if you are developing medical imaging system to analyze medical reports like X-rays, MRIs and CT-scans etc. to detect diseases then accuracy must be the top check. No any minor compromise on the accuracy no matter how long model takes to detect because here the use case isn't real-time so we can leverage the speed but not accuracy. But balance must be there in all cases.

### Model Performance Evaluation

Now let's see what are the evaluation steps to compare the speeds and accuracies of different models and select the best one according to the requirements.

#### Speed Metrics

**Latency**: How long the model takes to process one image/frame (milliseconds). Lower latency means faster processing.

**FPS (Frames Per Second)**: How many frames model can process in 1 second. Valid for the real-time applications.

**Throughput**: Number of images processed per second when running in batch mode. Important for server deployments.

**Model Size**: Models' variants like nano, medium and large and the difference in speed and accuracy, the tradeoffs among them and what fits in the required use case.

#### Accuracy Metrics

**1: Mean Average Precision (mAP)**

It evaluates how precisely model detects objects among crowded frames. It measures the accuracy of the model in identifying and localizing objects within an image. It combines precision (the proportion of correctly identified objects among all predicted objects) and recall (the proportion of correctly identified objects among all the actual objects). It gives a single score that shows how well model finds objects and how well it avoids false positives. A higher mAP means the model is more reliable and consistent. It helps to compare different models and select according to the use case and requirement.

```python
# mAP calculation example
def calculate_mAP(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate mean Average Precision for object detection.

    Args:
        predictions: List of predicted boxes with confidence scores
        ground_truths: List of actual boxes
        iou_threshold: IoU threshold to consider prediction as correct

    Returns:
        mAP score
    """
    aps = []
    for class_id in range(num_classes):
        # Get predictions and ground truths for this class
        class_preds = [p for p in predictions if p['class'] == class_id]
        class_gts = [gt for gt in ground_truths if gt['class'] == class_id]

        # Sort predictions by confidence
        class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)

        # Calculate precision and recall at each threshold
        tp = 0
        fp = 0
        precisions = []
        recalls = []

        for pred in class_preds:
            # Check if prediction matches any ground truth
            matched = False
            for gt in class_gts:
                if calculate_iou(pred['box'], gt['box']) >= iou_threshold:
                    tp += 1
                    matched = True
                    break
            if not matched:
                fp += 1

            precision = tp / (tp + fp)
            recall = tp / len(class_gts)
            precisions.append(precision)
            recalls.append(recall)

        # Calculate AP for this class
        ap = calculate_ap(precisions, recalls)
        aps.append(ap)

    # Return mean of all APs
    return sum(aps) / len(aps)
```

**2: Recall**

The ratio of correctly predicted positive detections to all actual objects present. Out of all real objects present, how many did the model successfully detect. High recall = fewer false negatives. It evaluates the model's ability to capture every possible object without missing them. Recall is calculated by dividing true positives by the sum of true positives and false negatives. It ensures that even subtle and partially visible objects are not overlooked. Higher recall makes the model reliable for the scenarios where missing object is critical like medical imaging and security surveillance deployments.

```python
# Recall calculation
def calculate_recall(true_positives, false_negatives):
    """
    Calculate recall metric.

    Recall = TP / (TP + FN)
    """
    if (true_positives + false_negatives) == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)
```

**3: Precision**

It measures the accuracy of a model's positive predictions, indicating the proportion of items predicted as positive that were actually correct. It is calculated as True Positives / (True Positives + False Positives). High precision means the model has few false positives. This metric helps you understand how trustworthy your model is when it predicts a specific outcome.

```python
# Precision calculation
def calculate_precision(true_positives, false_positives):
    """
    Calculate precision metric.

    Precision = TP / (TP + FP)
    """
    if (true_positives + false_positives) == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)
```

**4: IoU (Intersection over Union)**

It measures how much the predicted object bounding box overlaps with the real (ground truth) box. It's the ratio between overlap area/total combined area. Higher IoU = better prediction accuracy by the model. It compares the difference between ground truths and predictions, making visible the accuracy of the model. As visible in the image below green box is ground truth actual object area and the red box is predicted area by model so we can see slightly difference in overlapping of the boxes this visibly shows the accuracy of the model.

![IOU](/images/IOU__.png)

Certain threshold is set for predicting the class with accuracy.

![IOU_threshold](/images/IOU_THR.png)

```python
# IoU calculation
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2] first box coordinates
        box2: [x1, y1, x2, y2] second box coordinates

    Returns:
        IoU score between 0 and 1
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

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou
```

**5: F1 Score**

F1 score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. Useful when you need to balance between false positives and false negatives.

```python
# F1 Score calculation
def calculate_f1_score(precision, recall):
    """
    Calculate F1 score from precision and recall.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    """
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

**6: Confusion Matrix**

Confusion matrix visualizes the performance of the model by showing true positives, false positives, true negatives and false negatives for each class. It helps identify which classes the model confuses with each other.

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(predictions, ground_truths, class_names):
    """
    Plot confusion matrix for object detection.

    Args:
        predictions: List of predicted class labels
        ground_truths: List of actual class labels
        class_names: List of class names
    """
    num_classes = len(class_names)
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for pred, gt in zip(predictions, ground_truths):
        matrix[gt][pred] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
```

These are the steps needed to consider and critically evaluate before selecting and finalizing the model for the system.

## Data Preprocessing

This is the most important and critical part of whole Machine Learning system. Whole performance of the model depends upon the dataset on which it's been trained. There is a famous saying (garbage in garbage out). Dataset must be cleaned, well balanced and must cover all the features that are required, and systems need to detect.

### Steps for Data Preprocessing

Dataset is split into 3 folders: Train/Validation/Test.

**Train**: This folder contains the larger amount of the dataset and models learn patterns and features from the images and annotations. Typically 70-80% of total data.

**Validation**: This folder is used during training to tune hyperparameters and monitor model performance. It helps prevent overfitting. Typically 10-15% of total data.

**Test**: This contains the final set of images that model hasn't seen before. It used after training to check how accurate the model has performed in real-time scenarios, simply testing the model on new images that were not included in the train dataset on which labels were drawn. Typically 10-15% of total data.

```python
import os
import shutil
import random

def split_dataset(source_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset into train, validation and test sets.

    Args:
        source_folder: Path to folder containing all images
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    # Get all image files
    images = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)

    # Calculate split indices
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # Split files
    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Create directories and move files
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = os.path.join(source_folder, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for file in files:
            src = os.path.join(source_folder, file)
            dst = os.path.join(split_dir, file)
            shutil.move(src, dst)

            # Also move corresponding label file if exists
            label_file = file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            label_src = os.path.join(source_folder, label_file)
            if os.path.exists(label_src):
                label_dst = os.path.join(split_dir, label_file)
                shutil.move(label_src, label_dst)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
```

#### Data Cleaning

Removing duplicates to remove the wrong information, incorrect annotations and irrelevant images that are not required because model training requires resources like GPU and memory for data-saving so be sure to manage all the resources efficiently.

```python
import hashlib
from PIL import Image

def remove_duplicate_images(folder_path):
    """
    Remove duplicate images from dataset based on image hash.

    Args:
        folder_path: Path to folder containing images
    """
    seen_hashes = set()
    duplicates = []

    for filename in os.listdir(folder_path):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue

        filepath = os.path.join(folder_path, filename)

        # Calculate image hash
        with open(filepath, 'rb') as f:
            img_hash = hashlib.md5(f.read()).hexdigest()

        if img_hash in seen_hashes:
            duplicates.append(filename)
            os.remove(filepath)
            print(f"Removed duplicate: {filename}")
        else:
            seen_hashes.add(img_hash)

    print(f"Total duplicates removed: {len(duplicates)}")
    return duplicates

def validate_annotations(images_folder, labels_folder):
    """
    Validate that all images have corresponding annotations and vice versa.

    Args:
        images_folder: Path to images folder
        labels_folder: Path to labels folder
    """
    images = set([os.path.splitext(f)[0] for f in os.listdir(images_folder)])
    labels = set([os.path.splitext(f)[0] for f in os.listdir(labels_folder)])

    # Find mismatches
    images_without_labels = images - labels
    labels_without_images = labels - images

    if images_without_labels:
        print(f"Images without labels: {images_without_labels}")
    if labels_without_images:
        print(f"Labels without images: {labels_without_images}")

    return images_without_labels, labels_without_images
```

#### Image Resizing

Resizing all the images to a uniform size that is required by the selected model for the input.

```python
from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(640, 640)):
    """
    Resize all images in folder to target size.

    Args:
        input_folder: Source folder containing images
        output_folder: Destination folder for resized images
        target_size: Tuple of (width, height) for target size
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue

        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open and resize image
        img = Image.open(input_path)
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(output_path)

    print(f"Resized {len(os.listdir(output_folder))} images to {target_size}")
```

#### Data Augmentation

This step of data preprocessing plays vital role in model performance and generalization. This step includes the rotation, flipping (horizontal/vertical) to increase the diversity. Scaling: zooming in/out to simulate at different distances. Brightness adjustment: this step changes the brightness levels of the images to simulate the different conditions like sunlight, cloudy weather or any lighting changes that cause the colors (pixels) difference. By training model on the images with varying brightness, it learns to recognize objects accurately regardless of environmental changes. This helps reduce the false positives and improves the model's performance for every condition.

```python
import cv2
import numpy as np
import albumentations as A

def augment_image(image, bboxes, class_labels):
    """
    Apply augmentations to image and bounding boxes.

    Args:
        image: Input image (numpy array)
        bboxes: List of bounding boxes in format [x_min, y_min, x_max, y_max]
        class_labels: List of class labels for each bbox

    Returns:
        Augmented image and bounding boxes
    """
    # Define augmentation pipeline
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.RandomScale(scale_limit=0.2, p=0.5),
        A.RandomSizedBBoxSafeCrop(width=640, height=640, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # Apply augmentation
    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

    return augmented['image'], augmented['bboxes'], augmented['class_labels']

# Example usage
def augment_dataset(input_folder, output_folder, augmentations_per_image=3):
    """
    Create augmented versions of dataset.

    Args:
        input_folder: Source folder with images and labels
        output_folder: Output folder for augmented data
        augmentations_per_image: Number of augmented versions per image
    """
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue

        # Load image
        img_path = os.path.join(input_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load corresponding bboxes (assuming YOLO format)
        label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
        bboxes, class_labels = load_yolo_labels(label_path, image.shape)

        # Generate augmented versions
        for i in range(augmentations_per_image):
            aug_img, aug_bboxes, aug_labels = augment_image(image, bboxes, class_labels)

            # Save augmented image and labels
            aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.jpg"
            aug_path = os.path.join(output_folder, aug_filename)
            cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

            # Save augmented labels
            aug_label_path = aug_path.replace('.jpg', '.txt')
            save_yolo_labels(aug_label_path, aug_bboxes, aug_labels, image.shape)

    print(f"Augmentation complete: {augmentations_per_image}x increase in dataset size")
```

#### Class Imbalance Handling

When dataset has unequal number of samples for different classes, the model may become biased towards the majority class. Here are techniques to handle class imbalance:

```python
def check_class_distribution(labels_folder):
    """
    Check distribution of classes in dataset.

    Args:
        labels_folder: Path to folder containing label files

    Returns:
        Dictionary with class counts
    """
    class_counts = {}

    for label_file in os.listdir(labels_folder):
        if not label_file.endswith('.txt'):
            continue

        with open(os.path.join(labels_folder, label_file), 'r') as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print("Class Distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"Class {class_id}: {count} instances")

    return class_counts

def oversample_minority_classes(images_folder, labels_folder, target_count=None):
    """
    Oversample minority classes by duplicating images.

    Args:
        images_folder: Path to images folder
        labels_folder: Path to labels folder
        target_count: Target number of instances per class (uses max if None)
    """
    # Get class distribution
    class_counts = check_class_distribution(labels_folder)

    if target_count is None:
        target_count = max(class_counts.values())

    # Oversample each class
    for class_id, count in class_counts.items():
        if count >= target_count:
            continue

        # Find all images containing this class
        images_with_class = []
        for label_file in os.listdir(labels_folder):
            label_path = os.path.join(labels_folder, label_file)
            with open(label_path, 'r') as f:
                if any(int(line.split()[0]) == class_id for line in f):
                    images_with_class.append(label_file.replace('.txt', '.jpg'))

        # Duplicate images until reaching target
        duplicates_needed = target_count - count
        for i in range(duplicates_needed):
            src_img = random.choice(images_with_class)
            # Copy and augment
            # Implementation here

    print("Oversampling completed")
```

#### Formats Checking/Conversion

Every model has its own specific annotation format for reading labels. So the annotations must be in it. Here are some examples.

**YOLO Format**

YOLO models use TXT file as label file containing the (class + bounding boxes x, y coordinates) for each image file and names of both files image and label file should be exact same at all.
Like eg: "Image-1.jpg = image-1.txt"

![YOLO format labels file](/images/yololabel.png)

```python
def convert_to_yolo_format(bbox, img_width, img_height):
    """
    Convert bounding box to YOLO format.

    Args:
        bbox: [x_min, y_min, x_max, y_max] in pixels
        img_width: Image width
        img_height: Image height

    Returns:
        [x_center, y_center, width, height] normalized to [0, 1]
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate center and dimensions
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height

    return [x_center, y_center, width, height]

def save_yolo_labels(output_path, bboxes, class_labels, img_shape):
    """
    Save bounding boxes in YOLO format.

    Args:
        output_path: Path to save label file
        bboxes: List of bounding boxes
        class_labels: List of class labels
        img_shape: (height, width, channels)
    """
    img_height, img_width = img_shape[:2]

    with open(output_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)
            line = f"{class_id} {' '.join(map(str, yolo_bbox))}\n"
            f.write(line)
```

**COCO JSON Format**

**Detectron2, Faster R-CNN (PyTorch), Mask R-CNN (PyTorch), RF-DETR.**

These models take JSON files as annotations. This file contains the metadata of the dataset including the information of each image (filename, size), the objects in each image (bounding boxes, categories of the object like class) and also the list of all objects/classes. Annotations are linked to the corresponding images using IDs.

![JSON file structure](/images/examplee.png)

```python
import json

def convert_to_coco_format(images_folder, labels_folder, output_json, class_names):
    """
    Convert dataset to COCO JSON format.

    Args:
        images_folder: Path to images folder
        labels_folder: Path to YOLO format labels folder
        output_json: Output JSON file path
        class_names: List of class names
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Add categories
    for class_id, class_name in enumerate(class_names):
        coco_format["categories"].append({
            "id": class_id,
            "name": class_name,
            "supercategory": "object"
        })

    annotation_id = 1

    # Process each image
    for img_id, img_file in enumerate(os.listdir(images_folder)):
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(images_folder, img_file)
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Add image info
        coco_format["images"].append({
            "id": img_id,
            "file_name": img_file,
            "width": img_width,
            "height": img_height
        })

        # Read YOLO labels
        label_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')
        label_path = os.path.join(labels_folder, label_file)

        if not os.path.exists(label_path):
            continue

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])

                # Convert from YOLO to COCO format
                x_min = (x_center - width / 2) * img_width
                y_min = (y_center - height / 2) * img_height
                bbox_width = width * img_width
                bbox_height = height * img_height

                # Add annotation
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    # Save JSON
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

    print(f"COCO format saved to {output_json}")
```

#### Data Preprocessing Platform

[Roboflow](https://roboflow.com/)

It's a web-based tool that has functionality to organize the data, preprocess including augmentation and format conversion among all the different models. It allows users to upload their dataset and annotate it, augment it. Select the model and it will generate well-balanced dataset including the Train & Validation & Test. It allows you to train model in it as well on some free credits and then you can choose the "paid version".

![](/images/roboflow.png)
![](/images/code_snipper.png)

It guides you through all the necessary steps you need to complete and then generates an API key for direct dataset integration via API or as a downloadable .zip file.

Here you can select the model format.
![](/images/format.png)

### Dataset Collection

Now comes the main part that where to find and get the dataset to train model, so the universal platform where multiple datasets are available is [Kaggle](https://www.kaggle.com/). It's widely used and most of the general datasets are available on it for free (check for the license of usage for each).

Other popular dataset sources:
- **COCO Dataset**: Common Objects in Context - 330K images, 80 object categories
- **Pascal VOC**: 20 object classes, widely used for benchmarking
- **Open Images**: 9M images with bounding boxes
- **LVIS**: Large Vocabulary Instance Segmentation - 1000+ categories
- **Objects365**: 365 object categories, 2M images

**Video Frames**

In case if dataset isn't available on these platforms then we have second option. Fetching frames from videos and real-time recordings and then defining the objects names. Let's assume you are collecting dataset for any company that has some products and they want to count all of them on the last stage of conveyor belt to count the production of units. So object can be anything special and not available, so then it comes that way of collecting dataset from the video recordings. Just taking the video of those products where they are eg on the conveyor belt and then extracting frames from it using program as well (PYTHON).

This is the simple Python code snippet that can be used to extract frames from the videos to collect the dataset.

```python
"""Video frame extraction script."""

import cv2
import os


def extract_frames_from_video(video_path, num_frames_to_extract, output_folder, skip_frames=10):
    """Extract frames from video file.

    Args:
        video_path: Path to input video file
        num_frames_to_extract: Number of frames to extract
        output_folder: Output directory for extracted frames
        skip_frames: Extract every nth frame to avoid similar frames
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    print(f"Video Info: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")

    count = 0
    frame_number = 0

    while count < num_frames_to_extract:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames to get diverse samples
        if frame_number % skip_frames == 0:
            output_path = f"{output_folder}/frame_{count:05d}.jpg"
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {count} at {frame_number}/{total_frames}")
            count += 1

        frame_number += 1

    cap.release()
    print(f"Done extracting {count} frames.")


def extract_frames_by_interval(video_path, output_folder, interval_seconds=1):
    """Extract frames at specific time intervals.

    Args:
        video_path: Path to input video file
        output_folder: Output directory for extracted frames
        interval_seconds: Time interval between extracted frames
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)

    count = 0
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            output_path = f"{output_folder}/frame_{count:05d}.jpg"
            cv2.imwrite(output_path, frame)
            count += 1

        frame_number += 1

    cap.release()
    print(f"Extracted {count} frames at {interval_seconds}s intervals")


def main():
    """Main function to execute frame extraction."""
    # Set your variables here
    video_path = "input_video.mp4"
    num_frames_to_extract = 100
    output_folder = "frames"

    extract_frames_from_video(video_path, num_frames_to_extract, output_folder, skip_frames=10)

    # Or use interval-based extraction
    # extract_frames_by_interval(video_path, output_folder, interval_seconds=1)


if __name__ == "__main__":
    main()
```

After you have collected your dataset images, you can use Roboflow to annotate them and convert the dataset into the format required by your model.

[Tutorial](https://youtu.be/Dk-6MCQ9j-c?si=dIzQyNsWWxoysQLV) - Complete guide how to prepare the dataset and getting it ready for the training.

## Training

This step is similar to training any human being for doing a specific task. So like that AI models are also trained on the specific dataset to perform the predictions based on the patterns learned from the dataset. To start the training firstly we prepare dataset (discussed earlier). Then selecting a specific architecture of the model. We can do this by fine-tuning an existing pre-trained model (YOLO, RT-DETR etc) or by defining a custom architecture of the model using frameworks like PyTorch, TensorFlow, Keras etc. These frameworks provide libraries and packages to define, train, test and evaluate models.

When setting up for the training, hyperparameters are configured to control and manage the learning process. They influence how the training progresses and they directly impact the model's performance and efficiency. Like how, hyperparameters determine how quickly and effectively model learns from patterns of the data. Hyperparameters are epochs, learning rate, batch size, momentum and weight decay etc. Let's have a simple overview of each and see what is role of each hyperparameter for affecting the training process at all.

### Hyperparameters

**Learning Rate**

Very crucial unit for the training process like it covers the 70% of the model efficiency during training process because it defines the steps model needs to take to learn patterns from the data. It controls how much model's weights are updated during training. Too high can make the model training unstable and too low can slow down the training.

Common learning rates: 0.001, 0.0001, 0.01
Best practice: Start with 0.001 and adjust based on loss curves.

```python
# Learning rate example
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate with different values for different layers
optimizer = torch.optim.SGD([
    {'params': model.backbone.parameters(), 'lr': 0.0001},
    {'params': model.head.parameters(), 'lr': 0.001}
])
```

**Batch Size**

Number of random samples from the dataset are processed before model updates its weights. Larger batch size = faster training but requires more memory. Smaller batch size = slower but better generalization.

Common batch sizes: 8, 16, 32, 64 (depends on GPU memory)

```python
# Batch size in dataloader
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

**Epochs**

Number of times entire training dataset is passed through the model. More epochs let the model learn better. More epochs = more iteration over the dataset and more detailed learning. But too many epochs can cause overfitting.

Common range: 50-300 epochs depending on dataset size.

```python
# Training loop with epochs
num_epochs = 100
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training code here
        pass
```

**Momentum**

Used in optimizers to accelerate in consistent directions improving convergence speed. It helps the optimizer to keep moving in the same direction and avoid getting stuck in local minima.

Common value: 0.9

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

**Weight Decay**

A regularization term added to prevent overfitting by penalizing large weights. It forces the model to keep weights small which helps in generalization.

Common values: 0.0001, 0.0005

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
```

**Learning Rate Scheduling**

Learning rate scheduling adjusts the learning rate during training to improve convergence. Common strategies include step decay, exponential decay and cosine annealing.

```python
import torch.optim.lr_scheduler as lr_scheduler

# Step decay: reduce LR every N epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Exponential decay
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

# Cosine annealing: smooth decrease
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

# ReduceLROnPlateau: reduce when metric stops improving
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

# Use in training loop
for epoch in range(num_epochs):
    train_one_epoch()
    val_loss = validate()
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    # scheduler.step()  # For other schedulers
```

**Warmup**

Learning rate warmup gradually increases the learning rate from a small value to the initial learning rate over the first few epochs. This helps stabilize training in the beginning.

```python
def warmup_lr_scheduler(optimizer, warmup_epochs, initial_lr):
    """
    Create warmup learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of warmup epochs
        initial_lr: Target learning rate after warmup
    """
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

# Usage
warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs=5, initial_lr=0.001)
main_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=95)

for epoch in range(100):
    if epoch < 5:
        warmup_scheduler.step()
    else:
        main_scheduler.step()
```

### Loss Functions

Loss functions measure how well the model's predictions match the ground truth. Different loss functions are used for different aspects of object detection.

**Classification Loss**

Measures how well the model predicts the correct class for detected objects.

```python
# Cross Entropy Loss for classification
classification_loss = nn.CrossEntropyLoss()

# Focal Loss: handles class imbalance by focusing on hard examples
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model predictions [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Usage
focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
loss = focal_loss_fn(class_predictions, targets)
```

**Localization Loss**

Measures how accurately the predicted bounding boxes match the ground truth boxes.

```python
# Smooth L1 Loss (used in Faster R-CNN)
localization_loss = nn.SmoothL1Loss()

# IoU Loss: directly optimizes IoU metric
def iou_loss(pred_boxes, target_boxes):
    """
    Calculate IoU loss for bounding boxes.

    Args:
        pred_boxes: Predicted boxes [N, 4] (x1, y1, x2, y2)
        target_boxes: Target boxes [N, 4] (x1, y1, x2, y2)
    """
    # Calculate IoU
    iou = calculate_iou(pred_boxes, target_boxes)

    # IoU loss = 1 - IoU
    loss = 1 - iou
    return loss.mean()

# GIoU Loss: Generalized IoU handles non-overlapping boxes better
def giou_loss(pred_boxes, target_boxes):
    """
    Calculate GIoU loss.
    """
    iou = calculate_iou(pred_boxes, target_boxes)

    # Calculate enclosing box
    x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    enclosing_area = (x2_c - x1_c) * (y2_c - y1_c)
    union = calculate_union(pred_boxes, target_boxes)

    giou = iou - (enclosing_area - union) / enclosing_area
    loss = 1 - giou
    return loss.mean()

# CIoU Loss: Complete IoU considers aspect ratio
def ciou_loss(pred_boxes, target_boxes):
    """
    Calculate CIoU loss (Complete IoU).
    More advanced, considers distance, overlap, and aspect ratio.
    """
    iou = calculate_iou(pred_boxes, target_boxes)

    # Calculate center distance
    pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2

    center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2

    # Calculate diagonal of enclosing box
    x1_c = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_c = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_c = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_c = torch.max(pred_boxes[:, 3], target_boxes[:, 3])

    diagonal = (x2_c - x1_c) ** 2 + (y2_c - y1_c) ** 2

    # Calculate aspect ratio term
    pred_w = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_h = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_w = target_boxes[:, 2] - target_boxes[:, 0]
    target_h = target_boxes[:, 3] - target_boxes[:, 1]

    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2)
    alpha = v / (1 - iou + v + 1e-7)

    ciou = iou - (center_distance / diagonal) - alpha * v
    loss = 1 - ciou
    return loss.mean()
```

**Combined Loss**

Object detection models typically use a combination of classification and localization losses:

```python
def combined_detection_loss(pred_classes, pred_boxes, target_classes, target_boxes,
                           class_weight=1.0, box_weight=1.0):
    """
    Combined loss for object detection.

    Args:
        pred_classes: Predicted class logits
        pred_boxes: Predicted bounding boxes
        target_classes: Ground truth classes
        target_boxes: Ground truth boxes
        class_weight: Weight for classification loss
        box_weight: Weight for localization loss
    """
    # Classification loss
    class_loss = F.cross_entropy(pred_classes, target_classes)

    # Localization loss
    box_loss = ciou_loss(pred_boxes, target_boxes)

    # Combined loss
    total_loss = class_weight * class_loss + box_weight * box_loss

    return total_loss, class_loss, box_loss
```

### Optimizers

Optimizers update model weights based on the gradients computed during backpropagation.

**SGD (Stochastic Gradient Descent)**

Simple and effective optimizer. Works well with momentum.

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0001
)
```

**Adam (Adaptive Moment Estimation)**

Adapts learning rate for each parameter. Good default choice for most cases.

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.0001
)
```

**AdamW**

Adam with decoupled weight decay. Better for training transformers and modern architectures.

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
```

**Comparison**

- SGD: Best for large-scale training, requires careful tuning
- Adam: Faster convergence, easier to tune, good for most cases
- AdamW: Best for transformer-based models (DETR, Vision Transformers)

### Regularization Techniques

Regularization helps prevent overfitting and improves model generalization.

**Dropout**

Randomly drops neurons during training to prevent co-adaptation.

```python
import torch.nn as nn

class DetectionHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout = nn.Dropout(0.5)  # Drop 50% of neurons
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during training
        x = self.fc2(x)
        return x
```

**Batch Normalization**

Normalizes layer inputs to stabilize training and allow higher learning rates.

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # Normalize across batch
        x = self.relu(x)
        return x
```

**Early Stopping**

Stop training when validation performance stops improving.

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

**Label Smoothing**

Prevents the model from becoming overconfident by using soft targets instead of hard labels.

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, predictions, targets):
        """
        Args:
            predictions: [batch_size, num_classes]
            targets: [batch_size] class indices
        """
        log_probs = F.log_softmax(predictions, dim=-1)

        # Create smooth targets
        smooth_targets = torch.zeros_like(log_probs)
        smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)

        loss = (-smooth_targets * log_probs).sum(dim=-1).mean()
        return loss

# Usage
criterion = LabelSmoothingLoss(num_classes=80, smoothing=0.1)
```

### Transfer Learning

Transfer learning uses pre-trained models to leverage knowledge learned from large datasets.

**Why Transfer Learning?**

- Reduces training time significantly
- Requires less data
- Often achieves better accuracy
- Saves computational resources

**How to Use Transfer Learning**

```python
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze backbone layers (don't update during training)
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for your number of classes
num_classes = 10  # Your dataset classes
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Only train the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Or fine-tune all layers with different learning rates
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 0.0001},  # Later layers
    {'params': model.fc.parameters(), 'lr': 0.001}         # New layer
])
```

**Fine-tuning Strategy**

1. **Stage 1**: Train only the new head with frozen backbone (5-10 epochs)
2. **Stage 2**: Unfreeze last few layers, train with lower learning rate (10-20 epochs)
3. **Stage 3**: Fine-tune entire model with very low learning rate (optional)

```python
def train_with_progressive_unfreezing(model, train_loader, val_loader):
    """
    Progressive unfreezing strategy for transfer learning.
    """
    # Stage 1: Train only head
    print("Stage 1: Training head only")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.head.parameters(), lr=0.001)
    train_epochs(model, train_loader, val_loader, optimizer, epochs=10)

    # Stage 2: Unfreeze last layers
    print("Stage 2: Fine-tuning last layers")
    for param in model.backbone.layer4.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam([
        {'params': model.backbone.layer4.parameters(), 'lr': 0.0001},
        {'params': model.head.parameters(), 'lr': 0.0005}
    ])
    train_epochs(model, train_loader, val_loader, optimizer, epochs=20)

    # Stage 3: Fine-tune all
    print("Stage 3: Fine-tuning entire model")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    train_epochs(model, train_loader, val_loader, optimizer, epochs=10)
```

## PyTorch

It's an open-source framework that provides tools to build, train, and fine-tune models (neural networks). It provides each step we need to make a working model. It has vast network of libraries.

**torch** for core operations and inference applications.

**torchvision** for computer vision image processing work. It has preloaded datasets that we can just import from library and the pre-trained models that can be loaded to fine-tune them and also it provides the structure for defining custom models for different tasks.

**torchaudio** for audio processing and **torchtext** for natural language processing tasks.

Minimal implementation of the PyTorch code for fine-tuning (training) object detection model on custom dataset.

[Training](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/_training.py)

And this one is for inference:

[Inference](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/_inference.py)

**Code implementation example**
[torch](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torch_.py)
[torchaudio](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torchaudio_.py)
[torchvision](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Pytorch/torchvision_.py)

### Official Documentations of Framework/References

[Docs](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html) - This documentation covers everything needed to define a custom neural network/model using PyTorch.

[Explanation of neural network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)

[Fine-tuning using torchvision](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html)

## TensorFlow

It's an open-source framework for building, training and deploying AI models. It also provides libraries and architectures to build custom neural network/models and also some pre-trained models for inference. Preloaded datasets, post-processing and preprocessing tools for datasets handling. Major libraries from TensorFlow and their code implementations.

**tensorflow** - Core library for defining models, performing operations and training models.
[tensorflow](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_core.py)

**tensorflow_hub** - It's model zoo of the TensorFlow, a great repository for reusable pre-trained models to fine-tune and integrate them directly into the applications.
[tensorflow_hub](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_hub_.py)

**tf.data** - For loading, preprocessing and handling dataset.
[tf.data](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tf_data.py)

**tf.image** - Utilities for image processing tasks.
[tf.image](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tf_image.py)

[Object detection with TensorFlow](https://www.tensorflow.org/hub/tutorials/tf2_object_detection)
[Inference](https://github.com/aaliyanahmed1/tensorflow_/blob/main/tf2_object_detection.ipynb)

[Action recognition from video](https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub) - Recognizing action and events from the videos using TensorFlow.

This TensorFlow implementation contains deep detailed explanation of fine-tuning an object detection model on custom dataset. All are pre-trained and preloaded in TensorFlow, no need to download them manually. All of the stuff is built-in.

[Fine-tuning Explanation](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/tensorflow_explain.py)

This was all about introduction of the frameworks. Now let's get back to the training.

[Training using TensorFlow](https://github.com/aaliyanahmed1/ML-Guide/blob/main/tensorflow/training.py) - This is the sample practical implementation of training a model on custom dataset for object detection.

## Fine-Tuning Models for Custom Detections

There are many state-of-the-art object detection models that can be fine-tuned and integrated into applications. But there are some checks we need to see first before going ahead. eg License, Resources (Hardware), size and speed according to use case as we discussed earlier.

### 1: RF-DETR

By [Roboflow](https://roboflow.com/) is real-time transformer-based object detection model. It excels in both accuracy and speed, fit for most of the use cases in application. It's licensed under Apache 2.0 so it can be used freely for commercial applications. It has variants that vary in speed and size to fit in with the environment.

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

For training RF-DETR on custom dataset first preprocess the dataset using Roboflow and then download it according to the RF-DETR format by selecting format in Roboflow then feed it into the code and then start training by defining which model you want to train. Let's have a hands-on example of it at all.

[FINE-TUNE RF-DETR](https://github.com/aaliyanahmed1/ML-Guide/blob/main/RF-DETR_/train_rfdetr.py)

### 2: YOLO

By [Ultralytics](https://www.ultralytics.com/) commonly used model throughout, best fit for real-time applications and fast, easy to fine-tune. It takes image of 640x640 pixels as standard input. But it's not under Apache 2.0 license so it can't be used freely for commercial applications. You have to pay to the company.

| Variant       | Size (MB) | Speed (ms/image) |
| ------------- | --------- | ---------------- |
| YOLO12 Nano   | ~14 MB    | 2.5              |
| YOLO12 Small  | ~27 MB    | 3.8              |
| YOLO12 Medium | ~44 MB    | 5.0              |
| YOLO12 Large  | ~89 MB    | 8.0              |

YOLO12n is ultralight and it's optimized for edge devices and real-time inferences. Can be used in applications where speed is required and hardware is small.
[yolo12n_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12n_.py)

YOLO12s is balanced with speed and accuracy, performs well comparatively to nano variant when integrated on the GPU based hardware.
[yolo12s_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12s_.py)

YOLO12m has significant accuracy difference from smaller ones and moderate speed, ideal when deployed on server based inferences.
[yolo12m_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12m_.py)

YOLO12Large is high-speed model best for where precision is crucial more than speed. Mainly for medical imaging systems.
[yolo12l_code](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/yolo12l_.py)

[Fine-tuning YOLO](https://github.com/aaliyanahmed1/ML-Guide/blob/main/Yolo_/YOLOS_/training.py) - These are the simple implementations of the YOLO model variants for object detection tasks.

### 3: Faster R-CNN

By Microsoft Research, it's a two-stage detector object detection known for its precision and high accuracy. It's slightly slower than other single-stage detectors. Two-stage detector means first it processes ROI (region of interests) in the image and then classifies and refines bounding boxes for each region. This process reduces the false positives and overlapping of objects. That's why mostly it's used where speed and accuracy both are required and mainly it can be seen deployed on medical imaging systems. And its variants are mainly the backbones it uses like CNN layers (ResNet-50, ResNet-101, MobileNet) which cause difference in speed and accuracy.

| Variant        | Backbone    | Size (MB) | Speed (ms/image) |
| -------------- | ----------- | --------- | ---------------- |
| Faster R-CNN 50  | ResNet-50   | ~120 MB   | 30               |
| Faster R-CNN 101 | ResNet-101  | ~180 MB   | 45               |
| Faster R-CNN M   | MobileNetV2 | ~60 MB    | 20               |

ResNet-50: This backbone is balanced for speed and accuracy so where both are crucial then this would be ideal fit and commonly from Faster R-CNN this backbone is commonly used.
[Faster R-CNN-ResNet50](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_resnet50.py)

ResNet-101: This has higher accuracy and slower inference so it should be integrated on precision mandatory applications.
[Faster R-CNN-ResNet101](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_resnet101.py)

MobileNet: This variant is again lightweight, faster but accuracy is compromised so not so ideal.
[Faster R-CNN_MobileNet](https://github.com/aaliyanahmed1/ML-Guide/blob/main/FasR-CNN_/fastrcnn_mobile.py)

These are the mostly used object detection models for commercial enterprise applications, research works and medical analysis. And all of them have multiple use case centric variants having specialization for the specific task. We have discussed them and now we will make a list of all the possible open source object detection models that are available for integration in production grade applications, research and development etc.

## Hugging Face

[Hugging Face](https://huggingface.co/) is AI platform that provides tools, datasets and pre-trained models for Machine learning tasks. It has its wide transformer library that offers multiple ready to use open source models. It's called models zoo where you can get any type of model for GenAI, Machine learning, Computer vision and Natural language processing etc.

One of its most powerful features is it provides inference API which allows to run models in cloud without setting up local environment. Just using API for sending request and all the computation will be handled by Hugging Face. There are two ways to use it: 1: Free API good for testing and personal use and 2 is paid plan for large applications and faster responses.

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

Transformers are type of deep learning architectures designed to handle sequential data using self-attention mechanisms instead of traditional or convolution. They excel at capturing long-range dependencies in data. Unlike older approaches that process sequences step by step, transformers compute relationships between all elements in a sequence simultaneously, allowing them to capture long-range dependencies. For our context we have to focus on ViTs (Vision Transformers).

**Computer Vision Transformers**

They adapt this architecture to computer vision by splitting an image into small patches, treating each patch like a word in a sentence and applying same attention mechanism to learn how different parts of the image relate to each other. Mainly used for image-to-text, text-to-image transformers for generating captions and images.

**ViT (Vision Transformer)**

The first pure transformer for image classification, treating images as sequence of patches not as pixels.
[ViTs](https://huggingface.co/google/vit-base-patch16-224-in21k) - Hugging Face

**Swin Transformer**

It uses shifted window attention mechanism for efficient scaling to high-resolution images. It excels in segmentation, detection and classification.
[swin](https://huggingface.co/keras-io/swin-transformers)

**BLIP/BLIP-2**

A Vision language model for tasks like image captioning, VQA (Visual Question Answering) and retrieval. It takes images as input and generates its caption by defining what's happening inside the image. BLIP-2 improves the efficiency by using pre-trained language models for better reasoning over visual inputs. Patches understanding goes to language models and then they generate accurate caption.
[Blip](https://huggingface.co/Salesforce/blip2-flan-t5-xxl)

**Florence**

Large scale vision foundation model for various multimodal vision-language applications. It supports tasks such as image-text matching, captioning in enterprise and real-world production grade deployments.
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

  **YOLOS-Tiny**
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

## MLflow

MLflow is an open-source platform, purpose-built to assist machine learning practitioners and teams handling the complexities of the machine learning process. MLflow focuses on the full lifecycle for machine learning projects ensuring that each phase is manageable, traceable and reproducible. MLflow provides comprehensive support for traditional machine learning and deep learning workflows. From experiment tracking and model versioning to deployment and monitoring, MLflow streamlines every aspect of ML lifecycles. Whether you're working with scikit-learn models, training deep neural networks, or managing complex ML pipelines, MLflow provides the tools you need to build reliable, scalable machine learning systems.

**Core Features**: MLflow Tracking provides comprehensive experiment logging, parameters tracking, metrics tracking, model versioning and artifact management.

**Experiment Organization**: Track and compare multiple model experiments.

**Metric Visualization**: Built-in plots and charts for model performance.

**Artifact Storage**: Store models, plots and other files for each run.

**Collaboration**: Share experiments and models with team members.

```python
import mlflow
import mlflow.pytorch

# Set experiment name
mlflow.set_experiment("object_detection_training")

# Start a run
with mlflow.start_run(run_name="yolo_v8_experiment"):
    # Log parameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 100)
    mlflow.log_param("model", "YOLOv8n")

    # Train model
    for epoch in range(num_epochs):
        train_loss = train_one_epoch()
        val_loss = validate()

        # Log metrics
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("learning_rate", get_current_lr(), step=epoch)

    # Log final metrics
    mlflow.log_metric("final_mAP", final_map)
    mlflow.log_metric("final_precision", final_precision)
    mlflow.log_metric("final_recall", final_recall)

    # Log model
    mlflow.pytorch.log_model(model, "model")

    # Log artifacts (plots, confusion matrix, etc.)
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("training_curves.png")

# Compare experiments
experiments = mlflow.search_runs(experiment_names=["object_detection_training"])
print(experiments[["params.learning_rate", "metrics.final_mAP", "metrics.final_precision"]])
```

[MLflow Implementations](https://github.com/aaliyanahmed1/ML-Guide/tree/main/MLFlow_)

## Post-Processing

Post-processing techniques are applied to model outputs to improve detection quality and remove redundant predictions.

### Non-Maximum Suppression (NMS)

NMS removes duplicate detections of the same object by keeping only the detection with highest confidence and removing overlapping boxes.

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression to remove duplicate detections.

    Args:
        boxes: List of bounding boxes [N, 4] (x1, y1, x2, y2)
        scores: Confidence scores [N]
        iou_threshold: IoU threshold for considering boxes as duplicates

    Returns:
        Indices of boxes to keep
    """
    # Sort by confidence scores
    indices = np.argsort(scores)[::-1]
    keep = []

    while len(indices) > 0:
        # Keep box with highest score
        current = indices[0]
        keep.append(current)

        if len(indices) == 1:
            break

        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])

        # Keep boxes with IoU less than threshold
        indices = indices[1:][ious < iou_threshold]

    return keep

# Usage
filtered_indices = nms(predicted_boxes, confidence_scores, iou_threshold=0.5)
final_boxes = predicted_boxes[filtered_indices]
final_scores = confidence_scores[filtered_indices]
final_classes = predicted_classes[filtered_indices]
```

### Soft-NMS

Soft-NMS reduces scores of overlapping boxes instead of completely removing them. Better for crowded scenes.

```python
def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.3):
    """
    Soft-NMS reduces scores instead of removing boxes completely.

    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        sigma: Gaussian sigma parameter
        score_threshold: Minimum score to keep

    Returns:
        Filtered boxes and scores
    """
    keep_boxes = []
    keep_scores = []

    indices = np.argsort(scores)[::-1]

    while len(indices) > 0:
        current = indices[0]
        keep_boxes.append(boxes[current])
        keep_scores.append(scores[current])

        if len(indices) == 1:
            break

        current_box = boxes[current]
        remaining_boxes = boxes[indices[1:]]

        # Calculate IoU
        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])

        # Reduce scores using Gaussian function
        weight = np.exp(-(ious ** 2) / sigma)
        scores[indices[1:]] *= weight

        # Keep boxes above threshold
        indices = indices[1:][scores[indices[1:]] > score_threshold]

    return np.array(keep_boxes), np.array(keep_scores)
```

### Confidence Thresholding

Filter out low-confidence detections to reduce false positives.

```python
def filter_by_confidence(boxes, scores, classes, threshold=0.5):
    """
    Filter detections by confidence threshold.

    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        classes: Class predictions [N]
        threshold: Minimum confidence to keep

    Returns:
        Filtered boxes, scores, and classes
    """
    mask = scores >= threshold
    return boxes[mask], scores[mask], classes[mask]

# Usage
filtered_boxes, filtered_scores, filtered_classes = filter_by_confidence(
    boxes, scores, classes, threshold=0.6
)
```

### Class-Specific NMS

Apply NMS separately for each class to avoid removing detections of different objects that overlap.

```python
def class_specific_nms(boxes, scores, classes, num_classes, iou_threshold=0.5):
    """
    Apply NMS separately for each class.

    Args:
        boxes: Bounding boxes [N, 4]
        scores: Confidence scores [N]
        classes: Class IDs [N]
        num_classes: Total number of classes
        iou_threshold: IoU threshold for NMS

    Returns:
        Filtered boxes, scores, and classes
    """
    keep_boxes = []
    keep_scores = []
    keep_classes = []

    for class_id in range(num_classes):
        # Get detections for this class
        class_mask = classes == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]

        if len(class_boxes) == 0:
            continue

        # Apply NMS for this class
        keep_indices = nms(class_boxes, class_scores, iou_threshold)

        keep_boxes.append(class_boxes[keep_indices])
        keep_scores.append(class_scores[keep_indices])
        keep_classes.append(np.full(len(keep_indices), class_id))

    if len(keep_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    return (np.concatenate(keep_boxes),
            np.concatenate(keep_scores),
            np.concatenate(keep_classes))
```

## Model Deployment

Deployment is very typical part of every Machine learning workflow. When it comes to deployment, maintaining FPS for real-time systems becomes nightmare of MLOps architects so that's why the universal way to deploy model and maintain performance is to decouple it from training framework, that simplifies and reduces down burden of heavy dependencies and speeds up the process is exporting model in ONNX (Open Neural Network Exchange) format. This simplifies integration of model and makes it compatible.

### ONNX (Open Neural Network Exchange)

It is an open standard format for representing machine learning models. Exporting models to ONNX decouples them from the original training framework, making them easier to integrate into different platforms, whether on a server, multiple edge devices, or in the cloud. It ensures compatibility across various tools and allows optimized inference on different hardware setups, helping maintain real-time performance.

Here is simple minimal code implementation to export model that is in its training framework to export it to ONNX format for cross-platform integration and further optimizations.

```python
import torch
import torch.onnx

def export_pytorch_to_onnx(model, output_path, input_size=(1, 3, 640, 640)):
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_size: Input tensor size (batch, channels, height, width)
    """
    # Set model to evaluation mode
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(input_size)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Model exported to {output_path}")

# Usage
model = load_trained_model()
export_pytorch_to_onnx(model, "model.onnx", input_size=(1, 3, 640, 640))
```

```python
# TensorFlow to ONNX
import tf2onnx
import tensorflow as tf

def export_tensorflow_to_onnx(model_path, output_path):
    """
    Export TensorFlow model to ONNX format.

    Args:
        model_path: Path to TensorFlow saved model
        output_path: Path to save ONNX model
    """
    model = tf.saved_model.load(model_path)

    spec = (tf.TensorSpec((None, 640, 640, 3), tf.float32, name="input"),)

    output_path_onnx, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        output_path=output_path
    )

    print(f"Model exported to {output_path}")
```

### ONNXRuntime

It is a high-performance inference engine designed to run ONNX models efficiently across different platforms. It takes the ONNX model and applies graph optimization, operator fusion and quantization to reduce memory usage and computation time. So models run faster on servers, cloud environments and on multiple edge devices without needing original training framework. It can also speed up training process of large models by just making simple changes in code it can make training faster and efficient without changing workflow too much.

```python
import onnxruntime as ort
import numpy as np

def run_onnx_inference(onnx_model_path, input_image):
    """
    Run inference using ONNXRuntime.

    Args:
        onnx_model_path: Path to ONNX model file
        input_image: Input image as numpy array

    Returns:
        Model predictions
    """
    # Create inference session
    session = ort.InferenceSession(onnx_model_path)

    # Get input name
    input_name = session.get_inputs()[0].name

    # Preprocess image
    input_data = preprocess_image(input_image)
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)

    # Run inference
    outputs = session.run(None, {input_name: input_data})

    return outputs

def preprocess_image(image):
    """
    Preprocess image for model input.

    Args:
        image: Input image (numpy array or PIL Image)

    Returns:
        Preprocessed image
    """
    # Resize to model input size
    image = cv2.resize(image, (640, 640))

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Transpose to (C, H, W) format
    image = np.transpose(image, (2, 0, 1))

    return image

# Usage
predictions = run_onnx_inference("model.onnx", input_image)
boxes, scores, classes = postprocess_predictions(predictions)
```

### Model Optimization Techniques

**Quantization**

Reduces model size and speeds up inference by converting weights from float32 to int8.

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_model_path, output_model_path):
    """
    Quantize ONNX model to reduce size and improve speed.

    Args:
        input_model_path: Path to original ONNX model
        output_model_path: Path to save quantized model
    """
    quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8
    )

    print(f"Quantized model saved to {output_model_path}")

    # Compare sizes
    original_size = os.path.getsize(input_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_model_path) / (1024 * 1024)

    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")

# Usage
quantize_onnx_model("model.onnx", "model_quantized.onnx")
```

**TensorRT Optimization**

NVIDIA TensorRT provides highly optimized inference for NVIDIA GPUs.

```python
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

def convert_onnx_to_tensorrt(onnx_model_path, engine_path, precision="fp16"):
    """
    Convert ONNX model to TensorRT engine for faster inference on NVIDIA GPUs.

    Args:
        onnx_model_path: Path to ONNX model
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', or 'int8')
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_model_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # Build engine
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)

    engine = builder.build_engine(network, config)

    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"TensorRT engine saved to {engine_path}")

# Usage
convert_onnx_to_tensorrt("model.onnx", "model.trt", precision="fp16")
```

**Batch Inference**

Process multiple images in a single forward pass for better throughput.

```python
def batch_inference(model, images, batch_size=8):
    """
    Process multiple images in batches for better throughput.

    Args:
        model: Model or inference session
        images: List of images
        batch_size: Number of images per batch

    Returns:
        List of predictions for all images
    """
    all_predictions = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]

        # Preprocess batch
        batch_input = np.stack([preprocess_image(img) for img in batch_images])

        # Run inference
        batch_output = model.run(None, {'input': batch_input})

        all_predictions.extend(batch_output)

    return all_predictions
```

### Deployment Strategies

**Docker Containerization**

Package model and dependencies for consistent deployment.

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY model.onnx .
COPY inference.py .
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]
```

```python
# app.py - Flask API for model serving
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = load_onnx_model("model.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for object detection.
    """
    # Get image from request
    file = request.files['image']
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run inference
    predictions = run_inference(model, image)

    # Format response
    results = {
        'boxes': predictions['boxes'].tolist(),
        'scores': predictions['scores'].tolist(),
        'classes': predictions['classes'].tolist()
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

**Edge Deployment**

Optimize for resource-constrained devices like Raspberry Pi, Jetson Nano.

```python
# Optimized inference for edge devices
def edge_inference(model_path, image, use_gpu=False):
    """
    Optimized inference for edge devices.

    Args:
        model_path: Path to quantized ONNX model
        image: Input image
        use_gpu: Whether to use GPU if available

    Returns:
        Predictions
    """
    # Set execution providers
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

    # Create session with optimization
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4

    session = ort.InferenceSession(model_path, sess_options, providers=providers)

    # Run inference
    input_data = preprocess_image(image)
    outputs = session.run(None, {'input': input_data})

    return outputs
```

**RTSP Stream Processing**

Process real-time video streams from cameras.

```python
import cv2
import threading
from queue import Queue

class RTSPProcessor:
    def __init__(self, rtsp_url, model_path):
        """
        Process RTSP stream for real-time object detection.

        Args:
            rtsp_url: RTSP stream URL
            model_path: Path to ONNX model
        """
        self.rtsp_url = rtsp_url
        self.model = ort.InferenceSession(model_path)
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.running = False

    def capture_frames(self):
        """
        Capture frames from RTSP stream.
        """
        cap = cv2.VideoCapture(self.rtsp_url)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

        cap.release()

    def process_frames(self):
        """
        Process frames with object detection.
        """
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()

                # Run inference
                predictions = run_onnx_inference(self.model, frame)

                # Post-process
                boxes, scores, classes = postprocess_predictions(predictions)

                # Draw results
                output_frame = draw_detections(frame, boxes, scores, classes)

                if not self.result_queue.full():
                    self.result_queue.put(output_frame)

    def start(self):
        """
        Start processing pipeline.
        """
        self.running = True

        # Start capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.start()

        # Start processing thread
        process_thread = threading.Thread(target=self.process_frames)
        process_thread.start()

    def stop(self):
        """
        Stop processing pipeline.
        """
        self.running = False

# Usage
processor = RTSPProcessor("rtsp://camera_ip:port/stream", "model.onnx")
processor.start()

# Display results
while True:
    if not processor.result_queue.empty():
        frame = processor.result_queue.get()
        cv2.imshow('Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

processor.stop()
cv2.destroyAllWindows()
```

[ONNXRuntime Docs](https://onnxruntime.ai/docs/get-started/with-python.html#install-onnx-runtime)

[ONNXRuntime for training](https://onnxruntime.ai/training)

## Monitoring and Production

### Model Monitoring

Track model performance in production to detect issues early.

```python
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self, log_file="model_monitor.log"):
        """
        Monitor model performance in production.

        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
        self.predictions = []
        self.inference_times = []

        # Setup logging
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def log_prediction(self, image_id, boxes, scores, classes, inference_time):
        """
        Log prediction details.

        Args:
            image_id: Unique identifier for image
            boxes: Predicted boxes
            scores: Confidence scores
            classes: Predicted classes
            inference_time: Time taken for inference (ms)
        """
        num_detections = len(boxes)
        avg_confidence = float(np.mean(scores)) if len(scores) > 0 else 0.0

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'num_detections': num_detections,
            'avg_confidence': avg_confidence,
            'inference_time_ms': inference_time,
            'classes': classes.tolist() if hasattr(classes, 'tolist') else classes
        }

        logging.info(f"Prediction: {log_entry}")

        self.predictions.append(log_entry)
        self.inference_times.append(inference_time)

    def get_statistics(self):
        """
        Get performance statistics.

        Returns:
            Dictionary with statistics
        """
        if not self.predictions:
            return None

        stats = {
            'total_predictions': len(self.predictions),
            'avg_inference_time_ms': np.mean(self.inference_times),
            'max_inference_time_ms': np.max(self.inference_times),
            'min_inference_time_ms': np.min(self.inference_times),
            'fps': 1000 / np.mean(self.inference_times),
            'avg_detections_per_image': np.mean([p['num_detections'] for p in self.predictions]),
            'avg_confidence': np.mean([p['avg_confidence'] for p in self.predictions])
        }

        return stats

    def check_anomalies(self):
        """
        Check for anomalies in predictions.
        """
        stats = self.get_statistics()

        # Check for performance degradation
        if stats['avg_inference_time_ms'] > 100:  # Threshold
            logging.warning(f"High inference time detected: {stats['avg_inference_time_ms']:.2f}ms")

        # Check for low confidence
        if stats['avg_confidence'] < 0.5:  # Threshold
            logging.warning(f"Low average confidence detected: {stats['avg_confidence']:.2f}")

        # Check for too many/few detections
        if stats['avg_detections_per_image'] > 50:
            logging.warning(f"High number of detections: {stats['avg_detections_per_image']:.1f}")

# Usage
monitor = ModelMonitor("production_monitor.log")

for image in images:
    start_time = time.time()
    boxes, scores, classes = run_inference(model, image)
    inference_time = (time.time() - start_time) * 1000

    monitor.log_prediction(image_id, boxes, scores, classes, inference_time)
    monitor.check_anomalies()

# Get overall statistics
stats = monitor.get_statistics()
print(f"Average FPS: {stats['fps']:.1f}")
print(f"Average inference time: {stats['avg_inference_time_ms']:.2f}ms")
```

### A/B Testing

Compare different model versions in production.

```python
import random

class ABTester:
    def __init__(self, model_a_path, model_b_path, traffic_split=0.5):
        """
        A/B test two model versions.

        Args:
            model_a_path: Path to model A
            model_b_path: Path to model B
            traffic_split: Proportion of traffic for model A (0-1)
        """
        self.model_a = load_model(model_a_path)
        self.model_b = load_model(model_b_path)
        self.traffic_split = traffic_split

        self.results_a = []
        self.results_b = []

    def predict(self, image):
        """
        Route prediction to either model A or B.

        Args:
            image: Input image

        Returns:
            Predictions and model version used
        """
        use_model_a = random.random() < self.traffic_split

        if use_model_a:
            predictions = run_inference(self.model_a, image)
            model_version = 'A'
            self.results_a.append(predictions)
        else:
            predictions = run_inference(self.model_b, image)
            model_version = 'B'
            self.results_b.append(predictions)

        return predictions, model_version

    def compare_results(self):
        """
        Compare performance of both models.

        Returns:
            Comparison statistics
        """
        stats_a = calculate_stats(self.results_a)
        stats_b = calculate_stats(self.results_b)

        comparison = {
            'model_a': stats_a,
            'model_b': stats_b,
            'winner': 'A' if stats_a['avg_confidence'] > stats_b['avg_confidence'] else 'B'
        }

        return comparison

# Usage
ab_tester = ABTester("model_v1.onnx", "model_v2.onnx", traffic_split=0.5)

for image in test_images:
    predictions, version = ab_tester.predict(image)
    print(f"Used model {version}")

results = ab_tester.compare_results()
print(f"Winner: Model {results['winner']}")
```

## Troubleshooting Common Issues

### Training Issues

**Problem: Loss not decreasing**

Solutions:
- Check learning rate (try 0.001, 0.0001)
- Verify data loading and augmentation
- Check for label errors in dataset
- Reduce model complexity
- Use learning rate scheduler

```python
# Debugging training
def debug_training(model, train_loader):
    """
    Debug common training issues.
    """
    # Check if gradients are flowing
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for {name}")
        else:
            print(f"{name}: grad mean = {param.grad.mean():.6f}")

    # Check data loading
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['image'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"Data range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
```

**Problem: Overfitting**

Solutions:
- Add more data augmentation
- Increase dropout
- Add weight decay
- Use early stopping
- Reduce model size

**Problem: Out of memory**

Solutions:
- Reduce batch size
- Use gradient accumulation
- Reduce image size
- Use mixed precision training

```python
# Gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(train_loader):
    loss = compute_loss(model(batch))
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Inference Issues

**Problem: Slow inference**

Solutions:
- Export to ONNX and use ONNXRuntime
- Use quantization
- Reduce image size
- Use batch inference
- Enable GPU acceleration

**Problem: Low accuracy in production**

Solutions:
- Check image preprocessing
- Verify input size matches training
- Adjust confidence threshold
- Check for distribution shift
- Retrain with production-like data

## Conclusion

This guide covered the complete machine learning workflow for object detection from fundamentals to production deployment. You learned about:

- **Model selection** based on speed and accuracy requirements
- **Data preprocessing** including cleaning, augmentation, and format conversion
- **Training** with proper hyperparameters, loss functions, and optimizers
- **Evaluation** using mAP, precision, recall, IoU and other metrics
- **Popular frameworks** like PyTorch and TensorFlow
- **State-of-the-art models** like YOLO, RF-DETR, and Faster R-CNN
- **Post-processing** techniques like NMS and confidence thresholding
- **Deployment** strategies using ONNX, ONNXRuntime, and containerization
- **Production monitoring** and troubleshooting

## Next Steps

To continue your machine learning journey:

1. **Practice**: Implement a complete object detection pipeline from scratch
2. **Experiment**: Try different models and hyperparameters on your own dataset
3. **Deploy**: Deploy a model to production and monitor its performance
4. **Advanced topics**: Explore instance segmentation, 3D object detection, or video understanding
5. **Stay updated**: Follow latest research papers and model releases
6. **Contribute**: Share your knowledge and contribute to open source projects

## Useful Resources

**Documentation:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [TensorFlow Guides](https://www.tensorflow.org/guide)
- [ONNXRuntime Documentation](https://onnxruntime.ai/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

**Papers:**
- YOLO: You Only Look Once (Redmon et al.)
- Faster R-CNN: Towards Real-Time Object Detection (Ren et al.)
- DETR: End-to-End Object Detection with Transformers (Carion et al.)
- Focal Loss for Dense Object Detection (Lin et al.)

**Communities:**
- [PyTorch Forum](https://discuss.pytorch.org/)
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Papers with Code](https://paperswithcode.com/)
- [Kaggle Competitions](https://www.kaggle.com/competitions)

**Tools:**
- [Roboflow](https://roboflow.com/) - Dataset management
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [Label Studio](https://labelstud.io/) - Data annotation

Good luck with your object detection projects!
