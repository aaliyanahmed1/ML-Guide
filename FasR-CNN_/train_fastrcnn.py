"""
train_fastrcnn.py
Fine-tune Faster R-CNN on a custom dataset.
"""

import os
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    fasterrcnn_resnet101_fpn,
    fasterrcnn_mobilenet_v3_large_fpn
)
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class CustomDetectionDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        """
        Custom dataset for object detection
        Args:
            root_dir: Directory with images
            annotation_file: Path to COCO format annotations
            transform: Optional transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
            
        # Create category mapping
        self.cat_ids = {cat['id']: idx + 1 for idx, cat in enumerate(self.annotations['categories'])}
    
    def __len__(self):
        return len(self.annotations['images'])
    
    def __getitem__(self, idx):
        # Load image
        img_info = self.annotations['images'][idx]
        image_path = os.path.join(self.root_dir, img_info['file_name'])
        image = Image.open(image_path).convert('RGB')
        
        # Get annotations for this image
        boxes = []
        labels = []
        for ann in self.annotations['annotations']:
            if ann['image_id'] == img_info['id']:
                # Get box coordinates
                x, y, w, h = ann['bbox']
                boxes.append([x, y, x + w, y + h])
                labels.append(self.cat_ids[ann['category_id']])
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Prepare target
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target

def train_one_epoch(model, optimizer, data_loader, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader):
        # Move to device
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
    
    return total_loss / len(data_loader)

def main():
    # Configuration
    config = {
        'data_dir': 'path/to/images',  # Update with your image directory
        'train_annotations': 'path/to/train.json',  # Update with your annotations
        'val_annotations': 'path/to/val.json',
        'model_type': 'resnet50',  # Options: resnet50, resnet101, mobile
        'num_epochs': 10,
        'batch_size': 4,
        'learning_rate': 0.005,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print(f"Using device: {config['device']}")
    
    # Create dataset
    transform = T.Compose([
        T.ToTensor(),
    ])
    
    train_dataset = CustomDetectionDataset(
        config['data_dir'],
        config['train_annotations'],
        transform
    )
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model based on configuration
    num_classes = len(train_dataset.cat_ids) + 1  # +1 for background
    
    if config['model_type'] == 'resnet50':
        model = fasterrcnn_resnet50_fpn_v2(num_classes=num_classes)
    elif config['model_type'] == 'resnet101':
        model = fasterrcnn_resnet101_fpn(num_classes=num_classes)
    else:  # mobile
        model = fasterrcnn_mobilenet_v3_large_fpn(num_classes=num_classes)
    
    model.to(config['device'])
    
    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    # Training loop
    train_losses = []
    print(f"Starting training for {config['num_epochs']} epochs...")
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train one epoch
        loss = train_one_epoch(
            model, optimizer, train_loader, config['device']
        )
        train_losses.append(loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        print(f"Loss: {loss:.4f}")
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, f'fasterrcnn_{config["model_type"]}_epoch_{epoch+1}.pth')
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.close()
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main()
