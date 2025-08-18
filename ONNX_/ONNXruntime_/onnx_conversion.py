import torch
import onnx
import onnxruntime as ort
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------
# Step 1: Load your PyTorch RF-DETR model
# ---------------------------
# Replace "rf_detr_base.pt" with your RF-DETR PyTorch model path
pytorch_model_path = "rf_detr_base.pt"
pytorch_model = torch.load(pytorch_model_path)
pytorch_model.eval()  # Set to evaluation mode, required for inference/export

# ---------------------------
# Step 2: Export PyTorch model to ONNX
# ---------------------------
# dummy_input is used to define input shape for ONNX export
# Adjust shape according to your model: [batch_size, channels, height, width]
dummy_input = torch.randn(1, 3, 800, 1333)  # Example for RF-DETR image input
onnx_model_path = "rf_detr_base.onnx"

# Export to ONNX format
torch.onnx.export(
    pytorch_model,
    dummy_input,
    onnx_model_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=13,  # Ensures compatibility with ONNX runtime and transformations
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}  # Allow dynamic batch size
)

print("✅ PyTorch model exported to ONNX!")

# ---------------------------
# Step 3: Graph Optimization
# ---------------------------
# ONNX Runtime can automatically optimize the computation graph for faster inference
# Graph optimization removes unnecessary nodes and rearranges computations
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# ORT_ENABLE_ALL applies all levels of graph optimization

# Load the ONNX model with optimization applied
session = ort.InferenceSession(onnx_model_path, sess_options=session_options)
optimized_model_bytes = session._sess.model_bytes()

# Save the optimized ONNX model
optimized_model_path = "rf_detr_base_graph_optimized.onnx"
with open(optimized_model_path, "wb") as f:
    f.write(optimized_model_bytes)

print("✅ Graph optimization applied and saved!")

# ---------------------------
# Step 4: Operator Fusion
# ---------------------------
# Operator fusion merges multiple operations into one to reduce compute overhead
# Example: Conv + BatchNorm + ReLU → single fused operator
# The 'model_type' argument is mostly for BERT/GPT-like models; for RF-DETR it applies generic fusions
fused_model = optimize_model(optimized_model_path, model_type="rf-detr")
fused_model_path = "rf_detr_base_fused.onnx"
fused_model.save_model_to_file(fused_model_path)

print("✅ Operator fusion applied and saved!")

# ---------------------------
# Step 5: Quantization
# ---------------------------
# Quantization converts FP32 weights to INT8 for faster and smaller inference
# Dynamic quantization reduces model size and speeds up inference without retraining
quantized_model_path = "rf_detr_base_quantized.onnx"
quantize_dynamic(
    model_input=fused_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8  # Quantize weights to INT8
)

print("✅ Quantization applied and saved!")

# ---------------------------
# ✅ Summary
# You now have 3 optimized versions:
# 1. Graph-optimized ONNX model
# 2. Operator-fused ONNX model
# 3. Quantized INT8 ONNX model ready for faster inference
# ---------------------------
