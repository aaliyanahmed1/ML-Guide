"""Minimal TensorFlow object detection training script with COCO dataset.

This script demonstrates training a custom object detection model using:
- TensorFlow Hub for pre-trained models
- TensorFlow Datasets for COCO data loading
- Keras for model building and training
- Hardcoded paths and hyperparameters for simplicity
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras

# -----------------------------------------------------------------------------
# User-configurable constants (edit these paths/values)
# -----------------------------------------------------------------------------
TRAIN_SAMPLES = 1000  # Number of training samples to use
VAL_SAMPLES = 100     # Number of validation samples to use
EPOCHS = 5            # Number of training epochs
BATCH_SIZE = 32       # Batch size for training
LEARNING_RATE = 0.001 # Learning rate for optimizer
SAVE_PATH = "tf_object_detection_model.h5"  # Path to save trained model


def download_coco_dataset() -> Tuple[tf.data.Dataset, tf.data.Dataset, tfds.core.DatasetInfo]:
    """Download and prepare COCO dataset for object detection training.

    Returns:
        Tuple of (train_dataset, val_dataset, dataset_info).
    """
    print("Downloading COCO dataset (this might take a while)...")
    # Load a subset of COCO dataset for demonstration
    dataset, info = tfds.load(
        'coco/2017',
        split=[f'train[:{TRAIN_SAMPLES}]', f'validation[:{VAL_SAMPLES}]'],
        with_info=True,
        as_supervised=True
    )
    train_dataset, val_dataset = dataset
    return train_dataset, val_dataset, info

def prepare_image(image: tf.Tensor, target_shape: Tuple[int, int] = (224, 224)) -> tf.Tensor:
    """Prepare image for the model by resizing and normalizing.

    Args:
        image: Input image tensor.
        target_shape: Target size (height, width) for resizing.

    Returns:
        Preprocessed image tensor.
    """
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image


def create_data_pipeline(dataset: tf.data.Dataset, batch_size: int = BATCH_SIZE) -> tf.data.Dataset:
    """Create an optimized data pipeline for training.

    Args:
        dataset: Input dataset.
        batch_size: Batch size for training.

    Returns:
        Optimized dataset with preprocessing, caching, and batching.
    """
    return dataset.map(
        lambda image, labels: (prepare_image(image), labels)
    ).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_pretrained_model() -> tf.keras.Model:
    """Load a pre-trained MobileNetV2 model for object detection.

    Creates a transfer learning model using MobileNetV2 as backbone
    with custom classification head for COCO object detection.

    Returns:
        Compiled Keras model ready for training.
    """
    # Load pre-trained MobileNetV2 backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze backbone for transfer learning
    
    # Build custom model with classification head
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(90, activation='sigmoid')  # COCO has 90 categories
    ])
    
    return model

def visualize_predictions(image: tf.Tensor, predictions: np.ndarray, info: tfds.core.DatasetInfo, top_k: int = 3) -> None:
    """Visualize image with its top predictions.

    Args:
        image: Input image tensor.
        predictions: Model predictions array.
        info: Dataset information object.
        top_k: Number of top predictions to display.
    """
    plt.imshow(image)
    plt.axis('off')
    
    # Get top k predictions
    top_scores = np.argsort(predictions)[-top_k:][::-1]
    
    # Print predictions
    for score in top_scores:
        print(f"Class: {score}, Score: {predictions[score]:.4f}")


def train() -> None:
    """Main training function with hardcoded configuration.

    This function sets up the dataset, model, optimizer, and runs
    the training loop using the constants defined at the top.
    """
    # 1. Load and prepare COCO dataset
    print(f"Loading COCO dataset with {TRAIN_SAMPLES} train and {VAL_SAMPLES} val samples...")
    train_dataset, val_dataset, info = download_coco_dataset()
    train_dataset = create_data_pipeline(train_dataset)
    val_dataset = create_data_pipeline(val_dataset)

    # 2. Load pre-trained model
    print("Loading pre-trained MobileNetV2 model...")
    model = load_pretrained_model()

    # 3. Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # 4. Train model (fine-tuning)
    print(f"\nStarting training for {EPOCHS} epochs...")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=1)
        ]
    )

    # 5. Evaluate model
    print("\nEvaluating model...")
    results = model.evaluate(val_dataset)
    print(f"Test accuracy: {results[1]:.4f}")

    # 6. Save trained model
    model.save(SAVE_PATH)
    print(f"Model saved to: {SAVE_PATH}")

    # 7. Inference on a sample image
    print("\nPerforming inference on a sample image...")
    for images, labels in val_dataset.take(1):
        sample_image = images[0]
        sample_pred = model.predict(tf.expand_dims(sample_image, 0))[0]
        visualize_predictions(sample_image, sample_pred, info)

if __name__ == "__main__":
    train()
