"""
tensorflow_core.py
Core TensorFlow: Define models, perform operations, and train models.
"""
import tensorflow as tf

# Example: Define a simple linear model and train it
class LinearModel(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.w = tf.Variable(5.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def __call__(self, x):
        return self.w * x + self.b

# Training loop for the linear model
def train_linear_model():
    model = LinearModel()
    x = tf.constant([1.0, 2.0, 3.0, 4.0])
    y = tf.constant([3.0, 5.0, 7.0, 9.0])
    optimizer = tf.optimizers.SGD(learning_rate=0.01)
    for step in range(100):
        with tf.GradientTape() as tape:
            pred = model(x)
            loss = tf.reduce_mean(tf.square(pred - y))
        grads = tape.gradient(loss, [model.w, model.b])
        optimizer.apply_gradients(zip(grads, [model.w, model.b]))
        if step % 20 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")
    print(f"Trained weight: {model.w.numpy()}, bias: {model.b.numpy()}")

if __name__ == "__main__":
    train_linear_model()
