"""Minimal RF-DETR training entry (optional rfdetr package required).

This script demonstrates how to fine-tune RF-DETR using the third-party
`rfdetr` package. The Roboflow Inference SDK does not provide training; it is
for inference only. If you want to train locally, install `rfdetr` first.

Usage:
- Install: `pip install rfdetr`
- Set `DATASET_DIR` to your COCO-style dataset root
- Run: `python train_rfdetr.py`
"""

import sys

# Try to import the RF-DETR training API from the rfdetr package
try:
    from rfdetr import RFDETRBase  # type: ignore
    RFDETR_AVAILABLE = True
except Exception:  # pragma: no cover - package may be optional
    RFDETR_AVAILABLE = False

# User-configurable dataset/training settings
DATASET_DIR = "dataset"          # expects train/, valid/, test/ with *_annotations.coco.json
EPOCHS = 10
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 1e-4
OUTPUT_DIR = "rf_detr_finetuned"


def main() -> None:
    """Optionally train RF-DETR using the `rfdetr` package if installed."""
    if not RFDETR_AVAILABLE:
        print(
            "rfdetr package not found. To enable training, install it first: \n"
            "  pip install rfdetr\n\n"
            "Roboflow Inference SDK is used for inference only."
        )
        sys.exit(0)

    # Create base RF-DETR model (loads pretrained weights internally)
    model = RFDETRBase()

    # Launch fine-tuning on your COCO-style dataset
    model.train(
        dataset_dir=DATASET_DIR,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
        lr=LEARNING_RATE,
        output_dir=OUTPUT_DIR,
    )

    print(f"Training completed. Model and checkpoints saved to '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
