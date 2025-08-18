# Step 1: Import necessary modules from ONNX Runtime
from onnxruntime.quantization import quantize_dynamic, QuantType

# --------------------------------------------------------------------------
# Step 2: Define paths for your models
# "model.onnx" is your original full-precision model in FP32
# "model_quant.onnx" is the path where the quantized INT8 model will be saved
# --------------------------------------------------------------------------
input_model_path = "model.onnx"        # Original FP32 ONNX model
output_model_path = "model_quant.onnx" # Quantized INT8 ONNX model

# --------------------------------------------------------------------------
# Step 3: Apply Dynamic Quantization
# --------------------------------------------------------------------------
quantize_dynamic(
    model_input=input_model_path,        # Path to original ONNX model
    model_output=output_model_path,      # Path to save quantized model
    weight_type=QuantType.QInt8          # Quantize model weights to signed 8-bit integers
)

# --------------------------------------------------------------------------
# EXPLANATION:
# 1. Dynamic quantization: only the weights are converted to INT8 at runtime.
#    - Activations (inputs/outputs of layers) remain in FP32.
#    - This reduces model size and speeds up inference, especially on CPU.
# 2. QuantType.QInt8:
#    - QInt8 = signed 8-bit integers
#    - QUInt8 = unsigned 8-bit integers (alternative)
# 3. Benefits:
#    - Smaller model size -> easier to deploy on edge devices.
#    - Faster inference -> improves FPS for real-time applications.
#    - No retraining needed -> works directly with existing ONNX models.
# 4. Compatible across platforms:
#    - Works on servers, cloud, mobile devices, and edge hardware that supports ONNX Runtime.
# --------------------------------------------------------------------------

# Step 4: Done
print(f"✅ Dynamic quantization complete! Quantized model saved at {output_model_path}")
