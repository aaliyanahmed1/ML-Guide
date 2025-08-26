"""TensorFlow Hub: Model zoo for reusable pre-trained models.

This module demonstrates how to load and use pre-trained models from
TensorFlow Hub for various machine learning tasks.
"""

import tensorflow as tf
import tensorflow_hub as hub

def use_tf_hub_model():
    """Load and use a pre-trained image classifier from TensorFlow Hub."""
    # Load a MobileNetV2 model from TF Hub
    model_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    model = hub.KerasLayer(model_url)
    # Dummy image (224x224x3)
    image = tf.random.uniform([1, 224, 224, 3], maxval=1.0)
    # Run inference
    result = model(image)
    print(f"Model output shape: {result.shape}")

if __name__ == "__main__":
    use_tf_hub_model()
