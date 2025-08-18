import torch
import onnx
from onnxruntime.transformers.optimizer import optimize_model

# ---------------- STEP 1: Load PyTorch model ----------------
# Load your saved PyTorch model from a .pt file
# torch.load() deserializes the model object saved previously
pytorch_model = torch.load("model.pt")

# Set the model to evaluation mode
# eval() disables things like dropout and batchnorm updates
pytorch_model.eval()

# ---------------- STEP 2: Export PyTorch model to ONNX ----------------
# ONNX requires a dummy input to trace the model's computation graph
# Here, a tensor with batch size 1, 3 channels, and 224x224 image size is used
dummy_input = torch.randn(1, 3, 224, 224)

# Define the path for the exported ONNX model
onnx_model_path = "model.onnx"

# Export the PyTorch model to ONNX format
# - pytorch_model: the model to export
# - dummy_input: example input tensor for tracing
# - onnx_model_path: path to save the ONNX file
# - input_names/output_names: names for model inputs/outputs in ONNX graph
# - opset_version: ONNX operator version (ensures compatibility)
torch.onnx.export(pytorch_model,
                  dummy_input,
                  onnx_model_path,
                  input_names=["input"],
                  output_names=["output"],
                  opset_version=13)

# ---------------- STEP 3: Apply ONNX Operator Fusion ----------------
# optimize_model performs graph-level optimizations on the ONNX model
# These optimizations include:
# 1. Operator fusion (e.g., Conv + BatchNorm -> single Conv)
# 2. Constant folding (pre-computing constant operations)
# 3. Removing redundant or unused nodes
# Apply ONNX operator fusion to model
optimized_model = optimize_model(onnx_model_path, model_type="rf-detr")

# Define path to save the optimized ONNX model
fused_model_path = "fused_model.onnx"

# Save the optimized ONNX model to disk
optimized_model.save_model_to_file(fused_model_path)

# Print confirmation message
print(f"✅ Operator fusion applied! Optimized model saved at {fused_model_path}")
