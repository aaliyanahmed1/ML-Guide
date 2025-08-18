import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image

def load_pretrained_model():
    # Load a pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    print('Loaded pre-trained ResNet18')
    return model

def load_and_preprocess_image():
    # Create a dummy image
    img = Image.new('RGB', (224, 224), color='green')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img).unsqueeze(0)
    print('Preprocessed image shape:', img_tensor.shape)
    return img_tensor

def run_inference(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    print('Inference output shape:', output.shape)

def train_on_cifar10():
    # Download and load CIFAR10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32*32*3, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(1):
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

def main():
    print('--- TorchVision: Load Pretrained Model ---')
    model = load_pretrained_model()
    print('\n--- TorchVision: Preprocess Image ---')
    img_tensor = load_and_preprocess_image()
    print('\n--- TorchVision: Inference ---')
    run_inference(model, img_tensor)
    print('\n--- TorchVision: Training on CIFAR10 (1 epoch) ---')
    train_on_cifar10()

if __name__ == "__main__":
    main()
