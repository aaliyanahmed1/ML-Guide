# PyTorch Example Library

This repo simple, well-documented  examples of the pytorch ecosystem. Each file demonstrates core functionality of a major PyTorch library:

## Contents

- **torch_.py**  
  Core PyTorch: Tensors, neural networks, training loops, and inference. Example: Train a simple feedforward neural network on dummy data.

- **torchvision_.py**  
  TorchVision: Image processing, pre-trained models, and datasets. Example: Load a pre-trained ResNet, preprocess images, run inference, and train a simple model on CIFAR10.

- **torchaudio_.py**  
  TorchAudio: Audio processing, feature extraction, and audio classification. Example: Generate dummy audio, extract Mel spectrograms, and train a simple classifier.

- **requirements.txt**  
  List of required packages: `torch`, `torchvision`, `torchaudio`.

## Usage

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run any example:
   ```powershell
   python torch_.py
   python torchvision_.py
   python torchaudio_.py
   ```

Each script is self-contained and includes comments to help you understand the workflow.
