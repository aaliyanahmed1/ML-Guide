"""Object detection training example on VOC-style data with torchvision.

Contains utilities to load VOC, train Faster R-CNN, validate, and visualize.
Designed to be readable and CI-friendly.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# PASCAL VOC dataset classes (20 objects + background)
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]


def get_dataset():
    """Load VOC dataset and prepare data loaders."""
    # Basic transform: just convert to tensor
    transform = transforms.Compose([transforms.ToTensor()])

    # Load VOC dataset (downloads automatically if not present)
    print("Loading/Downloading VOC dataset...")
    train_dataset = torchvision.datasets.VOCDetection(
        root='./data', year='2012', image_set='train', download=True, transform=transform
    )
    val_dataset = torchvision.datasets.VOCDetection(
        root='./data', year='2012', image_set='val', download=True, transform=transform
    )

    def prepare_data(image, target):
        """Convert VOC annotations to model format."""
        # Ensure image is tensor
        if not isinstance(image, torch.Tensor):
            image = transform(image)

        # Extract bounding boxes and labels
        boxes, labels = [], []
        for obj in target['annotation']['object']:
            bbox = obj['bndbox']
            boxes.append([
                float(bbox['xmin']), float(bbox['ymin']), float(bbox['xmax']), float(bbox['ymax'])
            ])
            labels.append(VOC_CLASSES.index(obj['name']))

        # Convert to tensors
        return image, {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
        }

    def collate_fn(batch):
        """Process batch of samples for detection models."""
        images, targets = zip(*[prepare_data(img, tgt) for img, tgt in batch])
        return list(images), list(targets)

    # Create data loaders with common settings
    loader_args = {'batch_size': 8, 'num_workers': 2, 'pin_memory': True, 'collate_fn': collate_fn}

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

    return train_loader, val_loader, len(VOC_CLASSES)


def get_model(device, num_classes):
    """Create pre-trained Faster R-CNN model for object detection."""
    model = models.detection.fasterrcnn_resnet50_fpn(
        weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model.to(device)


def train_epoch(model, train_loader, optimizer, device):
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0

    for images, targets in tqdm(train_loader, desc='Training'):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, device):
    """Evaluate model on validation set and return average loss."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validating'):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item()

    return total_loss / len(val_loader)


def plot_training_history(history):
    """Plot training and validation loss curves to a PNG file."""
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Object Detection Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()


def visualize_predictions(model, images, targets, device, num_images=2, confidence_threshold=0.5):
    """Visualize ground truth and model predictions for a few images."""
    model.eval()
    images_list = list(images)[:num_images]

    with torch.no_grad():
        predictions = model([img.to(device) for img in images_list])

    for i in range(num_images):
        plt.figure(figsize=(12, 6))
        # Ground truth
        plt.subplot(1, 2, 1)
        plt.imshow(images_list[i].permute(1, 2, 0).cpu())
        plt.title('Ground Truth')
        for box in targets[i]['boxes'].cpu():
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)

        # Predictions
        plt.subplot(1, 2, 2)
        plt.imshow(images_list[i].permute(1, 2, 0).cpu())
        plt.title('Predictions')
        pred_boxes = predictions[i]['boxes'].cpu()
        pred_scores = predictions[i]['scores'].cpu()
        pred_labels = predictions[i]['labels'].cpu()
        for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
            if score > confidence_threshold:
                x1, y1, x2, y2 = box
                rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
                plt.gca().add_patch(rect)
                plt.text(x1, y1 - 5, f'{VOC_CLASSES[label]}: {score:.2f}', color='r', backgroundcolor='white')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def main():
    """End-to-end training routine with validation and visualization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = get_dataset()
    model = get_model(device, num_classes + 1)  # +1 for background class

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    history = {'train_loss': [], 'val_loss': []}

    num_epochs = 10
    best_val_loss = float('inf')
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        plot_training_history(history)

        if (epoch + 1) % 5 == 0:
            images, targets = next(iter(val_loader))
            visualize_predictions(model, images, targets, device)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
