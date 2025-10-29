```python

"""RF-DETR to ONNX conversion script."""

import torch
from rfdetr import RFDETRSmall


def convert_rfdetr_to_onnx():
    """Convert RF-DETR model to ONNX format."""
    # Load pretrained RF-DETR model
    model = RFDETRSmall()
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 3, 800, 800)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "rfdetr_small.onnx",
        input_names=["input"],
        output_names=["logits", "boxes"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
            "boxes": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        verbose=True,
    )

    print("✅ RF-DETR Small exported to rfdetr_small.onnx")


if __name__ == "__main__":
    convert_rfdetr_to_onnx()

```