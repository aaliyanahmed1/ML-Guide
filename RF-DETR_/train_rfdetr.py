# train_rfdetr_minimal.py

import os
from rfdetr import RFDETRBase
from rfdetr import RFDETRBase  # RF-DETR core class
from rfdetr import get_model
import torch

def main():
    # 1. Install prerequisites:
    #    pip install rfdetr

    # 2. Configure your dataset path (must follow COCO format)
    dataset_dir = "dataset"  # contains train/, valid/, test/ subfolders with images + _annotations.coco.json

    # 3. Load the base RF-DETR model
    model = RFDETRBase()  # loads RF-DETR Base with pretrained weights

    # 4. Fine-tune the model on your dataset
    model.train(
        dataset_dir=dataset_dir,
        epochs=10,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        output_dir="rf_detr_finetuned"
    )

    print("Training completed. Model and checkpoints saved to 'rf_detr_finetuned'.")

if __name__ == "__main__":
    main()
