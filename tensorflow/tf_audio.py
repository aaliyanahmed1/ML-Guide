"""
tf_audio.py
Functions for audio processing tasks in TensorFlow.
"""
import tensorflow as tf

# Example: Generate and process a sine wave audio signal
def audio_processing_example():
    # Generate a 1-second sine wave at 440 Hz
    sample_rate = 16000
    t = tf.linspace(0.0, 1.0, sample_rate)
    audio = tf.math.sin(2 * 3.14159265 * 440 * t)
    # Add noise
    noisy_audio = audio + tf.random.normal(audio.shape, stddev=0.05)
    # Normalize
    normalized = (noisy_audio - tf.reduce_mean(noisy_audio)) / tf.math.reduce_std(noisy_audio)
    print(f"Audio shape: {audio.shape}, Normalized mean: {tf.reduce_mean(normalized).numpy():.4f}")

if __name__ == "__main__":
    audio_processing_example()
