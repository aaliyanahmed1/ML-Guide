"""
training_pytorch.py
A simplified object detection training script using PyTorch and VOC dataset.
Key features:
- Automatic dataset download via torchvision
- Pre-trained Faster R-CNN model
- Object detection training pipeline
"""

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# PASCAL VOC dataset classes (20 objects + background)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def get_dataset():
    """Load VOC dataset and prepare data loaders"""
    
    # Basic transform: just convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Load VOC dataset (downloads automatically if not present)
    print("Loading/Downloading VOC dataset...")
    train_dataset = torchvision.datasets.VOCDetection(
        root='./data', year='2012', image_set='train',
        download=True, transform=transform
    )
    val_dataset = torchvision.datasets.VOCDetection(
        root='./data', year='2012', image_set='val',
        download=True, transform=transform
    )

    def prepare_data(image, target):
        """Convert VOC annotations to model format"""
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = transform(image)

        # Extract bounding boxes and labels
        boxes, labels = [], []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            boxes.append([
                float(bbox['xmin']), float(bbox['ymin']),
                float(bbox['xmax']), float(bbox['ymax'])
            ])
            labels.append(VOC_CLASSES.index(obj['name']))
        
        # Convert to tensors
        return image, {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }

    def collate_fn(batch):
        """Process batch of samples"""
        images, targets = zip(*[prepare_data(img, tgt) for img, tgt in batch])
        return list(images), list(targets)

    # Create data loaders with common settings
    loader_args = {
        'batch_size': 8,
        'num_workers': 2,
        'pin_memory': True,
        'collate_fn': collate_fn
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, len(VOC_CLASSES)

def get_model(device, num_classes):
    """Create pre-trained Faster R-CNN model for object detection"""
    # Load pre-trained model
    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    
    # Modify final layer for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    
    return model.to(device)

def train_epoch(model, train_loader, optimizer, device):
    """Train model for one epoch"""
    model.train()
    total_loss = 0
    
    # Process each batch with progress bar
    for images, targets in tqdm(train_loader, desc='Training'):
        # Move data to device
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Training step
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass only
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def plot_training_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(10, 5))
    
    # Plot loss
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Object Detection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()

def visualize_predictions(model, images, targets, device, num_images=2, confidence_threshold=0.5):
    """
    Visualize ground truth and model predictions side by side
    
    Args:
        model: The trained model
        images: Batch of images
        targets: Ground truth targets
        device: Device to run inference on
        num_images: Number of images to visualize
        confidence_threshold: Threshold for showing predictions
    """
    model.eval()
    images_list = list(images)[:num_images]
    
    # Get model predictions
    with torch.no_grad():
        predictions = model([img.to(device) for img in images_list])
    
    # Plot each image
    for i in range(num_images):
        plt.figure(figsize=(12, 6))
        
        # Original image with ground truth
        plt.subplot(1, 2, 1)
        plt.imshow(images_list[i].permute(1, 2, 0).cpu())
        plt.title('Ground Truth')
        
        # Draw ground truth boxes in green
        for box in targets[i]['boxes'].cpu():
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
        
        # Predictions
        plt.subplot(1, 2, 2)
        plt.imshow(images_list[i].permute(1, 2, 0).cpu())
        plt.title('Predictions')
        
        # Draw predicted boxes in red if confidence > threshold
        pred_boxes = predictions[i]['boxes'].cpu()
        pred_scores = predictions[i]['scores'].cpu()
        pred_labels = predictions[i]['labels'].cpu()
        
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > confidence_threshold:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                      linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                # Add score and class label
                plt.text(x1, y1-5, f'{VOC_CLASSES[label]}: {score:.2f}', 
                        color='r', backgroundcolor='white')
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get data loaders
    train_loader, val_loader, num_classes = get_dataset()
    
    # Get model
    model = get_model(device, num_classes + 1)  # +1 for background class
    
    # Set up training
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }

    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Plot training progress
        plot_training_history(history)
        
        # Visualize predictions on validation data
        if (epoch + 1) % 5 == 0:  # Show predictions every 5 epochs
            images, targets = next(iter(val_loader))
            visualize_predictions(model, images, targets, device)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
