import onnx
import onnxruntime as ort
from onnxruntime.transformers.fusion_options import FusionOptions

# -----------------------------
# Step 1: Define model paths
# -----------------------------
# 'onnx_model_path' = The path to your original ONNX model.
# 'optimized_model_path' = The path where you want to save the optimized model.
onnx_model_path = "model.onnx"
optimized_model_path = "model_optimized.onnx"

# -----------------------------
# Step 2: Create SessionOptions
# -----------------------------
# SessionOptions controls how ONNX Runtime executes the model.
# Here we will tell ONNX Runtime to perform graph optimizations.
so = ort.SessionOptions()

# -----------------------------
# Step 3: Set Graph Optimization Level
# -----------------------------
# ONNX Runtime can optimize the computation graph in different levels.
# Levels available:
#   - ORT_DISABLE_ALL       → No optimization.
#   - ORT_ENABLE_BASIC      → Basic optimizations (constant folding, dead node removal).
#   - ORT_ENABLE_EXTENDED   → More advanced optimizations (operator fusions, memory reuse).
#   - ORT_ENABLE_ALL        → All possible optimizations (recommended for performance).
# We use ORT_ENABLE_ALL to maximize speed.
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# -----------------------------
# Step 4: Create Inference Session
# -----------------------------
# InferenceSession loads the model and prepares it for execution.
# Passing 'so' applies our optimization settings.
session = ort.InferenceSession(onnx_model_path, so)

# -----------------------------
# Step 5: Extract Optimized Model
# -----------------------------
# '_sess' is the internal C++ session object inside ONNX Runtime.
# '.model_bytes()' gives us the fully optimized model in binary format.
# This includes all graph transformations applied by the optimizer.
optimized_model = session._sess.model_bytes()

# -----------------------------
# Step 6: Save Optimized Model
# -----------------------------
# We now write the optimized model bytes into a new ONNX file.
with open(optimized_model_path, "wb") as f:
    f.write(optimized_model)

# -----------------------------
# Step 7: Print Success Message
# -----------------------------
print("Graph optimized model saved!")
