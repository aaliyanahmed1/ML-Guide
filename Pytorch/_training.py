import os
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
from pycocotools.coco import COCO


# --------------------------------------------------------------
# User-configurable constants (edit these values as needed)
# --------------------------------------------------------------
# Path to the folder that contains your training images (e.g. .jpg, .png)
IMAGES_DIR = "path/to/train/images"  # e.g., "./dataset/train/images"
# Path to the COCO-format JSON annotation file that describes bounding boxes
ANNOTATIONS_JSON = "path/to/annotations.json"  # e.g., "./dataset/train/annotations.json"
# Number of passes over the dataset; keep small for quick tests
EPOCHS = 2
# How many images to process together; detection models typically use small batches
BATCH_SIZE = 2
# Learning rate for the optimizer; 0.005 is a common starting point for Faster R-CNN
LEARNING_RATE = 0.005
# Number of classes your model should predict, including background (COCO uses 91)
NUM_CLASSES = 91
# Where to save the model weights after each epoch
SAVE_PATH = "fasterrcnn_coco.pth"


# --------------------------------------------------------------
# Minimal COCO dataset wrapper for detection models
# --------------------------------------------------------------
class CocoDetectionDataset(Dataset):
    """A tiny dataset class that reads images and labels from COCO JSON.

    What it does:
    - Uses pycocotools.COCO to parse the annotation JSON file.
    - Loads an image and converts it into a PyTorch tensor.
    - Converts COCO bbox format [x, y, w, h] to [x1, y1, x2, y2].
    - Returns (image_tensor, target_dict) for each index.

    Why torchvision needs this shape:
    - torchvision detection models expect each sample as a tuple of:
      (image: Tensor[C,H,W] in [0,1], target: dict with keys 'boxes' and 'labels').
    - 'boxes': shape [N, 4], float32, with corners [x1, y1, x2, y2].
    - 'labels': shape [N], int64, holding class ids.
    - 'image_id' is optional but useful for bookkeeping.
    """

    def __init__(self, images_dir: str, annotation_json: str):
        # Save the image folder path
        self.images_dir = images_dir
        # Load the COCO annotations using pycocotools
        self.coco = COCO(annotation_json)
        # Create a sorted list of all image ids contained in the JSON
        self.image_ids: List[int] = list(sorted(self.coco.imgs.keys()))

    def __len__(self) -> int:
        # Total number of images in the dataset
        return len(self.image_ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        # Look up the COCO image id for this index
        image_id = self.image_ids[index]

        # Retrieve image metadata (e.g., file name), then build the full file path
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = os.path.join(self.images_dir, img_info["file_name"])

        # Load the image with PIL and convert to RGB (ensures 3 channels)
        image = Image.open(img_path).convert("RGB")

        # Convert PIL image to a PyTorch float tensor in range [0, 1]
        image_tensor = torchvision.transforms.functional.to_tensor(image)

        # Gather all annotations (objects) for this image id
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in anns:
            # COCO bbox is [x, y, width, height]. torchvision needs [x1, y1, x2, y2].
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            # Store the category id as the class label
            labels.append(ann["category_id"])

        # Build the target dictionary for this image
        target = {
            # Bounding boxes as float32 tensor
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            # Labels as int64 tensor
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            # Keep track of which image this target corresponds to
            "image_id": torch.tensor(image_id),
        }

        return image_tensor, target


# --------------------------------------------------------------
# Helper: collate function for variable-size images
# --------------------------------------------------------------
# Why we need this:
# - Detection models often receive images of different sizes in a batch.
# - The default collation (stacking tensors) doesn't work for variable shapes.
# - Returning lists of images and targets lets the model handle them correctly.

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


# --------------------------------------------------------------
# Build model: Faster R-CNN pretrained on COCO, re-head for num_classes
# --------------------------------------------------------------
# Explanation:
# - We use torchvision's pre-trained Faster R-CNN (ResNet50-FPN backbone).
# - Replace the classification head so it predicts NUM_CLASSES categories
#   (including background).

def build_model(num_classes: int) -> torch.nn.Module:
    # Load a COCO-pretrained Faster R-CNN backbone and head
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )

    # Get the number of input features for the classifier layer
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the predictor (classifier + bbox regressor) with a new one
    # that matches the number of classes we want to predict.
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    )
    return model


# --------------------------------------------------------------
# Training loop: short and simple
# --------------------------------------------------------------
# What happens here:
# - Move each batch to the chosen device (CPU/GPU).
# - Forward pass returns a dict of losses (classification, bbox regression, etc.).
# - Sum losses, backpropagate, optimizer step.
# - Save a checkpoint at the end of each epoch.

def train(model, dataloader, device, epochs: int, lr: float, save_path: str) -> None:
    # Collect only parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    # Stochastic Gradient Descent is a good default for detection models
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

    print(f"Training for {epochs} epoch(s) on {len(dataloader.dataset)} images...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in dataloader:
            # Move images and each key in targets to the same device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass: model returns a dict of loss components
            loss_dict = model(images, targets)
            # Total loss is the sum of individual losses
            loss = sum(loss for loss in loss_dict.values())

            # Standard training step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            epoch_loss += float(loss.item())

        # Average loss per batch in this epoch
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

        # Save weights after each epoch for simplicity
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint to: {save_path}")


def main() -> None:
    """Entry point: build dataset/dataloader, model, and run training."""
    # Pick a compute device automatically (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build dataset and dataloader using the constants defined above
    dataset = CocoDetectionDataset(IMAGES_DIR, ANNOTATIONS_JSON)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,  # critical for detection models
    )

    # Create the model and move it to the chosen device
    model = build_model(num_classes=NUM_CLASSES).to(device)

    # Train the model
    train(
        model=model,
        dataloader=dataloader,
        device=device,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        save_path=SAVE_PATH,
    )

    print("Training finished.")


if __name__ == "__main__":
    main()
