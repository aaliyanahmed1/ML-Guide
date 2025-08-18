"""
training.py
Example: Load COCO dataset and use a pre-trained model from TensorFlow Hub for object detection.
Demonstrates usage of multiple TensorFlow libraries:
- tensorflow.keras for data processing
- tensorflow_hub for pre-trained models
- tensorflow.data for data pipeline
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

def download_coco_dataset():
    """Download and prepare COCO dataset"""
    print("Downloading COCO dataset (this might take a while)...")
    # Load a subset of COCO dataset for demonstration
    dataset, info = tfds.load(
        'coco/2017',
        split=['train[:1000]', 'validation[:100]'],
        with_info=True,
        as_supervised=True
    )
    train_dataset, val_dataset = dataset
    return train_dataset, val_dataset, info

def prepare_image(image, target_shape=(224, 224)):
    """Prepare image for the model"""
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def create_data_pipeline(dataset, batch_size=32):
    """Create an optimized data pipeline"""
    return dataset.map(
        lambda image, labels: (prepare_image(image), labels)
    ).cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

def load_pretrained_model():
    """Load a pre-trained MobileNetV2 model from keras.applications"""
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(90, activation='sigmoid')  # COCO has 90 categories
    ])
    return model

def visualize_predictions(image, predictions, info, top_k=3):
    """Visualize image with its predictions"""
    plt.imshow(image)
    plt.axis('off')
    
    # Get top k predictions
    top_scores = np.argsort(predictions)[-top_k:][::-1]
    
    # Print predictions
    for score in top_scores:
        print(f"Class: {score}, Score: {predictions[score]:.4f}")

def main():
    # 1. Load and prepare COCO dataset
    train_dataset, val_dataset, info = download_coco_dataset()
    train_dataset = create_data_pipeline(train_dataset)
    val_dataset = create_data_pipeline(val_dataset)

    # 2. Load pre-trained model
    model = load_pretrained_model()

    # 3. Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )

    # 4. Train model (fine-tuning)
    print("\nFine-tuning the model...")
    history = model.fit(
        train_dataset,
        epochs=5,
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

    # 6. Inference on a sample image
    print("\nPerforming inference on a sample image...")
    for images, labels in val_dataset.take(1):
        sample_image = images[0]
        sample_pred = model.predict(tf.expand_dims(sample_image, 0))[0]
        visualize_predictions(sample_image, sample_pred, info)

if __name__ == "__main__":
    main()
