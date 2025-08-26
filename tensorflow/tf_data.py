"""Efficient loading, preprocessing, and handling of datasets using tf.data.

Includes practical examples: synthetic data, CSV loading, image folder loading,
and advanced preprocessing.
"""

import os

import tensorflow as tf

# Example 1: Create a tf.data pipeline for synthetic image data
def create_synthetic_data_pipeline():
    """Create a tf.data pipeline for random images and labels."""
    images = tf.random.uniform([100, 28, 28], maxval=255, dtype=tf.float32)
    labels = tf.random.uniform([100], maxval=10, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(buffer_size=100).batch(16).map(
        lambda x, y: (x / 255.0, y)
    )
    for batch_images, batch_labels in dataset.take(1):
        print(f"[Synthetic] Batch images shape: {batch_images.shape}")
        print(f"[Synthetic] Batch labels: {batch_labels}")

# Example 2: Load data from a CSV file using tf.data
def create_csv_pipeline(csv_path=None):
    """Create a tf.data pipeline for CSV data."""
    # If no CSV, create a dummy one
    if csv_path is None:
        csv_path = "dummy.csv"
        import pandas as pd
        df = pd.DataFrame({"feature1": range(10), "feature2": range(10, 20), "label": range(10)})
        df.to_csv(csv_path, index=False)
    dataset = tf.data.experimental.make_csv_dataset(
        csv_path,
        batch_size=4,
        label_name="label",
        num_epochs=1,
        shuffle=True
    )
    for batch in dataset.take(1):
        features, labels = batch
        print(f"[CSV] Features: {features}, Labels: {labels}")

# Example 3: Load images from a directory using tf.data
def create_image_folder_pipeline(image_dir=None):
    """Create a tf.data pipeline for images in a folder."""
    # If no folder, create dummy images
    if image_dir is None:
        image_dir = "dummy_images"
        os.makedirs(image_dir, exist_ok=True)
        import numpy as np
        from PIL import Image
        for i in range(3):
            arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(image_dir, f"img_{i}.png"))
    list_ds = tf.data.Dataset.list_files(os.path.join(image_dir, '*.png'))
    def process_path(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [32, 32])
        img = img / 255.0
        return img
    img_ds = list_ds.map(process_path).batch(2)
    for batch in img_ds.take(1):
        print(f"[Image Folder] Batch image shape: {batch.shape}")

# Example 4: Advanced preprocessing (filter, repeat, prefetch)
def advanced_preprocessing_pipeline():
    """Showcase filter, repeat, and prefetch in tf.data."""
    ds = tf.data.Dataset.range(10)
    ds = ds.filter(lambda x: x % 2 == 0).repeat(2).batch(3).prefetch(1)
    for batch in ds:
        print(f"[Advanced] Batch: {batch.numpy()}")

if __name__ == "__main__":
    print("--- Synthetic Data Pipeline ---")
    create_synthetic_data_pipeline()
    print("\n--- CSV Data Pipeline ---")
    create_csv_pipeline()
    print("\n--- Image Folder Pipeline ---")
    create_image_folder_pipeline()
    print("\n--- Advanced Preprocessing Pipeline ---")
    advanced_preprocessing_pipeline()
