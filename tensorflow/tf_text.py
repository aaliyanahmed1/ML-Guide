"""
tf_text.py
Tools for natural language processing in TensorFlow.
"""
import tensorflow as tf

# Example: Basic text vectorization and tokenization
def text_processing_example():
    # Example sentences
    sentences = ["TensorFlow is great!", "Deep learning with tf.text"]
    # Tokenize
    tokenizer = tf.keras.layers.TextVectorization(output_mode='int')
    tokenizer.adapt(sentences)
    tokenized = tokenizer(sentences)
    print(f"Tokenized sentences: {tokenized.numpy()}")

if __name__ == "__main__":
    text_processing_example()
