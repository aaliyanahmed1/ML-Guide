"""
tf_image.py
Utilities for image processing tasks in TensorFlow.
"""
import tensorflow as tf

# Example: Image augmentation and processing
def image_processing_example():
    # Create a dummy image (28x28x3)
    image = tf.random.uniform([28, 28, 3], maxval=255, dtype=tf.float32)
    # Resize
    resized = tf.image.resize(image, [64, 64])
    # Flip
    flipped = tf.image.flip_left_right(resized)
    # Adjust brightness
    bright = tf.image.adjust_brightness(flipped, delta=0.1)
    print(f"Original shape: {image.shape}, Resized: {resized.shape}")
    print(f"Bright pixel sample: {bright[0,0,:].numpy()}")

if __name__ == "__main__":
    image_processing_example()
