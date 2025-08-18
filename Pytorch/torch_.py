
# Import torch for tensor operations and neural networks
import torch
# Import torch.nn for neural network layers
import torch.nn as nn
# Import torch.optim for optimization algorithms
import torch.optim as optim
# DataLoader and TensorDataset for batching and dataset handling
from torch.utils.data import DataLoader, TensorDataset

def train_simple_model():

    # Create a dummy dataset: 100 samples, 10 features each
    X = torch.randn(100, 10)
    # Binary labels (0 or 1), shape (100, 1)
    y = torch.randint(0, 2, (100, 1)).float()
    # Wrap data in a TensorDataset for easy loading
    dataset = TensorDataset(X, y)
    # DataLoader for batching and shuffling
    loader = DataLoader(dataset, batch_size=16, shuffle=True)


    # Define a simple feedforward neural network
    model = nn.Sequential(
        nn.Linear(10, 8),  # Input layer to hidden layer
        nn.ReLU(),         # Activation function
        nn.Linear(8, 1),   # Hidden layer to output
        nn.Sigmoid()       # Sigmoid for binary output
    )
    # Binary cross-entropy loss for binary classification
    criterion = nn.BCELoss()
    # Adam optimizer for training
    optimizer = optim.Adam(model.parameters(), lr=0.01)


    # Training loop for 3 epochs
    for epoch in range(3):
        for xb, yb in loader:
            optimizer.zero_grad()   # Reset gradients
            preds = model(xb)       # Forward pass
            loss = criterion(preds, yb)  # Compute loss
            loss.backward()         # Backpropagation
            optimizer.step()        # Update weights
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


    # Inference: generate new test data and predict
    test_x = torch.randn(5, 10)
    with torch.no_grad():
        out = model(test_x)
    print('Test predictions:', out.squeeze().numpy())

if __name__ == "__main__":
    # Run the full example
    print('--- Torch Full Example ---')
    train_simple_model()
