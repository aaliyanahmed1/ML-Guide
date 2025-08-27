"""Export RF-DETR (Roboflow) to ONNX format.

This script:
1. Loads the pretrained RF-DETR model from Hugging Face.
2. Exports it to ONNX format.
3. Saves the model as `rf_detr.onnx`.

Usage:
    python export_to_onnx.py
"""

import torch
from transformers import AutoModelForObjectDetection


def export_rfdetr_to_onnx(
    model_name: str = "roboflow/rf-detr-resnet50",
    output_path: str = "rf_detr.onnx",
) -> None:
    """Load RF-DETR and export it to ONNX format.

    Args:
        model_name (str): Hugging Face model ID.
        output_path (str): Path to save the exported ONNX file.

    Returns:
        None
    """
    # Load pretrained RF-DETR
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.eval()

    # Dummy input (batch=1, 3-channel RGB, 640x640)
    dummy_input = torch.randn(1, 3, 640, 640)

    # Export model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["pred_logits", "pred_boxes"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "pred_logits": {0: "batch_size"},
            "pred_boxes": {0: "batch_size"},
        },
    )

    print(f"✅ RF-DETR model exported successfully to: {output_path}")


if __name__ == "__main__":
    export_rfdetr_to_onnx()
