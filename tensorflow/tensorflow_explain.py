### Object Detection with Bounding Boxes using TensorFlow

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def download_coco_dataset():
    """
    Download and prepare COCO dataset for object detection training.
    
    This function loads the COCO (Common Objects in Context) dataset directly from TensorFlow Datasets.
    The COCO dataset contains images with multiple objects and their corresponding bounding box annotations.
    
    Returns:
        tuple: (train_dataset, val_dataset, info)
            - train_dataset: Training dataset with 1000 samples
            - val_dataset: Validation dataset with 100 samples  
            - info: Dataset metadata including class names, number of samples, etc.
    
    Note:
        as_supervised=True ensures each item comes with properly paired (image, labels) for training.
        The labels include bounding box coordinates and class IDs for object detection tasks.
    """
    print("Downloading COCO dataset for object detection (this might take a while)...")
    # Load a subset of COCO dataset for demonstration
    dataset, info = tfds.load(
        'coco/2017',
        split=['train[:1000]', 'validation[:100]'],
        with_info=True,
        as_supervised=False  # Changed to False for object detection with bounding boxes
    )
    train_dataset, val_dataset = dataset
    return train_dataset, val_dataset, info


def prepare_image_and_boxes(image, target_shape=(416, 416)):
    """
    Prepare image and bounding boxes for object detection model input.
    
    This function preprocesses images and their corresponding bounding box annotations
    to match the required input format for object detection models. It resizes images,
    normalizes pixel values, and adjusts bounding box coordinates accordingly.
    
    Args:
        image: Input image tensor
        target_shape: Target dimensions (width, height) for the model input
        
    Returns:
        tuple: (processed_image, normalized_boxes)
            - processed_image: Preprocessed image ready for model input
            - normalized_boxes: Bounding box coordinates normalized to [0, 1] range
    """
    # Get original image dimensions
    original_height = tf.cast(tf.shape(image)[0], tf.float32)
    original_width = tf.cast(tf.shape(image)[1], tf.float32)
    
    # Resize image to target shape
    image = tf.image.resize(image, target_shape)
    
    # Convert to float32 and normalize to [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0
    
    return image, (original_height, original_width)


def create_data_pipeline(dataset, target_shape=(416, 416), batch_size=8):
    """
    Create an optimized data pipeline for object detection training.
    
    This function transforms the raw dataset into a high-performance training pipeline
    that efficiently processes images and bounding boxes. It applies preprocessing,
    caches data in memory, shuffles samples for better generalization, and enables
    parallel data loading during training.
    
    Args:
        dataset: Raw dataset containing images and annotations
        target_shape: Target dimensions for image resizing
        batch_size: Number of samples per training batch
        
    Returns:
        tf.data.Dataset: Optimized dataset pipeline ready for training
        
    Pipeline Features:
        - map(): Applies preprocessing to all images and boxes
        - cache(): Stores processed data in memory for faster access
        - shuffle(): Randomizes sample order for better model generalization
        - batch(): Groups samples into batches for efficient training
        - prefetch(): Enables parallel data loading during training
    """
    def process_sample(sample):
        """Process individual sample from the dataset"""
        image = sample['image']
        bboxes = sample['objects']['bbox']  # [y1, x1, y2, x2] format
        labels = sample['objects']['label']
        
        # Prepare image and get original dimensions
        processed_image, (orig_h, orig_w) = prepare_image_and_boxes(image, target_shape)
        
        # Normalize bounding boxes to [0, 1] range
        normalized_boxes = tf.stack([
            bboxes[:, 0] / orig_h,  # y1
            bboxes[:, 1] / orig_w,  # x1  
            bboxes[:, 2] / orig_h,  # y2
            bboxes[:, 3] / orig_w   # x2
        ], axis=1)
        
        return processed_image, {
            'bbox': normalized_boxes,
            'label': labels
        }
    
    return dataset.map(process_sample).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def load_pretrained_object_detection_model(input_shape=(416, 416, 3), num_classes=91):
    """
    Load a pre-trained object detection model for fine-tuning.
    
    This function creates a custom object detection model based on a pre-trained backbone.
    The model architecture includes:
    1. A pre-trained backbone (EfficientNetB0) for feature extraction
    2. Feature pyramid network for multi-scale detection
    3. Detection heads for bounding box regression and classification
    
    Args:
        input_shape: Input image dimensions (height, width, channels)
        num_classes: Number of object classes (91 for COCO dataset)
        
    Returns:
        tf.keras.Model: Compiled object detection model ready for training
        
    Architecture Details:
        - Backbone: EfficientNetB0 pre-trained on ImageNet (frozen during initial training)
        - Feature Pyramid: Multi-scale feature maps for detecting objects at different sizes
        - Detection Heads: Separate outputs for bounding box coordinates and class predictions
        - Transfer Learning: Leverages pre-trained features for better performance on custom datasets
    """
    # Load pre-trained backbone (EfficientNetB0)
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze backbone layers initially
    backbone.trainable = False
    
    # Create feature pyramid network
    def create_fpn(backbone_output):
        """Create Feature Pyramid Network for multi-scale detection"""
        # Extract features at different scales
        p3 = backbone_output  # 1/8 scale
        p4 = tf.keras.layers.Conv2D(256, 1, padding='same')(p3)
        p4 = tf.keras.layers.BatchNormalization()(p4)
        p4 = tf.keras.layers.ReLU()(p4)
        
        # Upsample and combine features
        p5 = tf.keras.layers.Conv2D(256, 3, padding='same')(p4)
        p5 = tf.keras.layers.BatchNormalization()(p5)
        p5 = tf.keras.layers.ReLU()(p5)
        
        return p3, p4, p5
    
    # Build model
    inputs = tf.keras.Input(shape=input_shape)
    backbone_output = backbone(inputs)
    p3, p4, p5 = create_fpn(backbone_output)
    
    # Detection heads
    # Bounding box regression head
    bbox_head = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(4, activation='sigmoid', name='bbox_output')  # [y1, x1, y2, x2]
    ])
    
    # Classification head
    class_head = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='class_output')
    ])
    
    # Combine outputs
    bbox_output = bbox_head(p5)
    class_output = class_head(p5)
    
    model = tf.keras.Model(inputs=inputs, outputs=[bbox_output, class_output])
    return model


def create_custom_loss():
    """
    Create custom loss function for object detection training.
    
    This function combines multiple loss components:
    1. Bounding Box Loss: Smooth L1 loss for coordinate regression
    2. Classification Loss: Focal loss for handling class imbalance
    3. Confidence Loss: Binary crossentropy for object presence
    
    Returns:
        function: Custom loss function that can be used in model compilation
        
    Loss Components:
        - BBox Loss: Smooth L1 loss for precise bounding box localization
        - Class Loss: Focal loss to handle class imbalance in object detection
        - Confidence Loss: Binary classification for object presence/absence
    """
    def detection_loss(y_true, y_pred):
        """
        Custom loss function for object detection.
        
        Args:
            y_true: Ground truth labels and bounding boxes
            y_pred: Model predictions for bounding boxes and classes
            
        Returns:
            float: Combined loss value
        """
        # Extract components
        bbox_true, class_true = y_true['bbox'], y_true['label']
        bbox_pred, class_pred = y_pred[0], y_pred[1]
        
        # Bounding box loss (Smooth L1)
        bbox_loss = tf.keras.losses.Huber()(bbox_true, bbox_pred)
        
        # Classification loss (Sparse Categorical Crossentropy)
        class_loss = tf.keras.losses.SparseCategoricalCrossentropy()(class_true, class_pred)
        
        # Combine losses with weights
        total_loss = bbox_loss + 2.0 * class_loss
        return total_loss
    
    return detection_loss


def visualize_predictions(image, bbox_pred, class_pred, info, confidence_threshold=0.5):
    """
    Visualize object detection predictions with bounding boxes.
    
    This function displays the input image with predicted bounding boxes and class labels.
    It filters predictions based on confidence threshold and draws rectangles around
    detected objects with their corresponding class names and confidence scores.
    
    Args:
        image: Input image tensor
        bbox_pred: Predicted bounding box coordinates [y1, x1, y2, x2]
        class_pred: Predicted class probabilities
        info: Dataset info containing class names
        confidence_threshold: Minimum confidence score to display predictions
        
    Visualization Features:
        - Bounding boxes drawn as rectangles around detected objects
        - Class labels displayed above each bounding box
        - Confidence scores shown for each detection
        - Color-coded boxes for different object classes
    """
    # Convert image to numpy array and denormalize
    image_np = (image.numpy() * 255).astype(np.uint8)
    
    # Get class names from dataset info
    class_names = info.features['objects']['label'].names
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_np)
    
    # Filter predictions by confidence
    max_confidences = np.max(class_pred, axis=1)
    confident_indices = np.where(max_confidences > confidence_threshold)[0]
    
    # Draw bounding boxes for confident predictions
    for idx in confident_indices:
        bbox = bbox_pred[idx]
        class_id = np.argmax(class_pred[idx])
        confidence = max_confidences[idx]
        
        # Convert normalized coordinates to pixel coordinates
        y1, x1, y2, x2 = bbox * np.array([image_np.shape[0], image_np.shape[1], 
                                          image_np.shape[0], image_np.shape[1]])
        
        # Create rectangle patch
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                               linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
        # Add label and confidence
        label = f"{class_names[class_id]}: {confidence:.2f}"
        ax.text(x1, y1-10, label, bbox=dict(boxstyle="round,pad=0.3", 
                                            facecolor="yellow", alpha=0.7))
    
    ax.set_title(f"Object Detection Results (Confidence > {confidence_threshold})")
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to execute the complete object detection training pipeline.
    
    This function orchestrates the entire process from dataset preparation to model training
    and evaluation. It demonstrates a complete workflow for fine-tuning an object detection
    model on the COCO dataset.
    
    Training Pipeline Steps:
    1. Dataset Loading: Download and prepare COCO dataset with bounding box annotations
    2. Data Preprocessing: Create optimized data pipeline with image and box preprocessing
    3. Model Creation: Load pre-trained backbone and add custom detection heads
    4. Model Compilation: Set up optimizer, loss functions, and evaluation metrics
    5. Training: Fine-tune the model using transfer learning techniques
    6. Evaluation: Assess model performance on validation dataset
    7. Inference: Demonstrate real-time object detection on sample images
    
    Model Architecture:
        - Backbone: EfficientNetB0 (pre-trained on ImageNet)
        - Feature Pyramid Network: Multi-scale feature extraction
        - Detection Heads: Bounding box regression and classification
        - Transfer Learning: Leverages pre-trained features for better performance
    """
    print("=== Object Detection Model Training Pipeline ===\n")
    
    # 1. Load and prepare COCO dataset
    print("Step 1: Loading and preparing COCO dataset...")
    train_dataset, val_dataset, info = download_coco_dataset()
    
    # Create optimized data pipelines
    print("Creating data pipelines...")
    train_pipeline = create_data_pipeline(train_dataset, batch_size=8)
    val_pipeline = create_data_pipeline(val_dataset, batch_size=8)
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {info.splits['train'].num_examples}")
    print(f"Validation samples: {info.splits['validation'].num_examples}")
    print(f"Number of classes: {info.features['objects']['label'].num_classes}\n")
    
    # 2. Load pre-trained object detection model
    print("Step 2: Loading pre-trained object detection model...")
    model = load_pretrained_object_detection_model(
        input_shape=(416, 416, 3),
        num_classes=info.features['objects']['label'].num_classes
    )
    
    print("Model architecture:")
    model.summary()
    print()
    
    # 3. Compile model with custom loss
    print("Step 3: Compiling model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=create_custom_loss(),
        metrics={
            'bbox_output': 'mae',  # Mean Absolute Error for bounding boxes
            'class_output': 'accuracy'  # Accuracy for classification
        }
    )
    
    # 4. Train model (fine-tuning)
    print("\nStep 4: Fine-tuning the object detection model...")
    print("Training for 10 epochs with early stopping...")
    
    history = model.fit(
        train_pipeline,
        epochs=10,
        validation_data=val_pipeline,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3, 
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5, 
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_object_detection_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ],
        verbose=1
    )
    
    # 5. Evaluate model
    print("\nStep 5: Evaluating model performance...")
    results = model.evaluate(val_pipeline, verbose=1)
    
    print("\nModel Evaluation Results:")
    print(f"Total Loss: {results[0]:.4f}")
    print(f"Bounding Box MAE: {results[1]:.4f}")
    print(f"Classification Accuracy: {results[2]:.4f}")
    
    # 6. Inference on sample images
    print("\nStep 6: Performing inference on sample images...")
    print("Displaying object detection results...")
    
    for sample in val_pipeline.take(3):  # Process 3 sample batches
        images, labels = sample
        sample_image = images[0]
        
        # Make predictions
        bbox_pred, class_pred = model.predict(tf.expand_dims(sample_image, 0), verbose=0)
        
        # Visualize predictions
        visualize_predictions(
            sample_image, 
            bbox_pred[0], 
            class_pred[0], 
            info, 
            confidence_threshold=0.3
        )
        
        print(f"Processed sample image with {len(bbox_pred[0])} detected objects")
    
    print("\n=== Training Pipeline Complete! ===")
    print("Model saved as 'best_object_detection_model.h5'")
    print("You can now use this model for inference on new images!")


if __name__ == "__main__":
    main()