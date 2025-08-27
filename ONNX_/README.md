### ONNX_

Utilities for exporting, optimizing, and quantizing models with ONNX and ONNX Runtime.

## Structure
- `ONNXruntime_/`
  - `onnx_conversion.py`: Helpers for exporting models to ONNX
  - `rf_detr_to_onnx.py`: Example export script for RF-DETR
  - `graphoptimization_.py`: Graph-level optimizations via ONNX/ORT
  - `operatorfusion_.py`: Operator fusion for faster inference
  - `Quantization_.py`: Post-training quantization utilities
  - `requirements.txt`: Dependencies for ONNX/ORT workflows


**Graph Optimization:** It rearranges and simplifies the model's performance computation graph to remov unnecassary steps, making it run faster.like combining adjacent layers or removing unused nodes.these all optimizations are automatically applied in ONNx Runtime when session is created with grapgh optimization enabled(oRT_ENABLE_ALL).reduing memory usage and CPU/GPU cycles,helping maintain higher FPS for real time inference 

```python
import onnxruntime as ort

# ---------------------------
# Create session options for ONNX Runtime
# ---------------------------
session_options = ort.SessionOptions()

# ---------------------------
# Set graph optimization level
# ---------------------------
# ORT_ENABLE_BASIC: Applies basic, safe optimizations like removing unused nodes
# ORT_ENABLE_EXTENDED: Includes more optimizations such as some operator fusions
# ORT_ENABLE_ALL: Applies all available optimizations including aggressive fusions and node eliminations
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# ---------------------------
# Load ONNX model with the chosen optimization level
# ---------------------------
session = ort.InferenceSession("model.onnx", sess_options=session_options)

# ---------------------------
# ✅ Explanation:
# 1. Creating SessionOptions allows you to configure how the ONNX model runs.
# 2. Graph optimization automatically rearranges computations, removes unnecessary nodes,
#    and may combine adjacent operations for faster inference.
# 3. ORT_ENABLE_ALL is recommended for real-time systems to maximize FPS.
# ---------------------------
print("✅ ONNX model loaded with graph optimization enabled!")
```

**Operator Fusion:** This function merges multiple small operations into a single, more efficient operation to reduce processing overhead. 
fusing cnv+ BtachNorm + ReLU into one step.it reduces number of kernels launches on CPU\GPU.that spped up inference and lowers memory overhead. it happens automatically when grapgh optimzation is enable=True.

```python
import onnx
from onnxruntime.transformers.optimizer import optimize_model

# ---------------------------
# Step 1: Define paths for models
# ---------------------------
onnx_model_path = "model.onnx"              # Original ONNX model
fused_model_path = "model_fused.onnx"       # Path to save operator-fused model

# ---------------------------
# Step 2: Apply Operator Fusion
# ---------------------------
# Operator fusion merges multiple operations into a single, more efficient operation
# Example: Conv + BatchNorm + ReLU → single fused operator
# This reduces kernel launches on GPU/CPU and improves inference speed
# The 'model_type' parameter can be adjusted if using a transformer-like model.
# For general vision models like RF-DETR, using "rf-detr" or "generic" is appropriate.
fused_model = optimize_model(onnx_model_path, model_type="rf-detr")

# ---------------------------
# Step 3: Save the fused model
# ---------------------------
fused_model.save_model_to_file(fused_model_path)

# ---------------------------
# ✅ Explanation:
# 1. Operator Fusion improves runtime by combining adjacent operations.
# 2. It reduces computation overhead and memory usage.
# 3. Works together with graph optimization for maximum inference speed.
# ---------------------------
print(f"✅ Operator fusion applied! Fused model saved at {fused_model_path}")
```

**Quantization:** It converts high-precision numbers(floating point32) into lower precision(INT8) to reduce memory and improve speed with minimal accuracy loss. compressing weights from 32 floating point to 8-bit integers.Quantization includes several advanced techniques beong simple dyunamuc int8 quantization .lets discuss them one-by-one .

**1: Dynamic Quantization** It Converts weights only to INT* during runtime .activation functions ( sigmoid,Relu etc) remains same FP#@.its fast adn easy and no need for training again and again for updates like that. ideal where all of the workflow is deployed on CPU .
```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# ---------------------------
# Step 1: Define paths
# ---------------------------
onnx_model_path = "model.onnx"          # Original FP32 ONNX model
quantized_model_path = "model_int8.onnx" # Path to save INT8 quantized model

# ---------------------------
# Step 2: Apply Dynamic Quantization
# ---------------------------
# Converts weights (FP32 → INT8) for selected ops (e.g., MatMul, GEMM, Attention)
quantize_dynamic(
    model_input=onnx_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8  # Quantize weights to INT8
)

# ---------------------------
# Step 3: ✅ Explanation
# ---------------------------
# 1. Reduces model size (weights now INT8).
# 2. Faster CPU inference (uses INT8 optimized kernels in ONNX Runtime).
# 3. No retraining required – works directly on exported ONNX model.
# ---------------------------
print(f"✅ Dynamic Quantization applied! Quantized model saved at {quantized_model_path}")

```

**2:Mixed Precision/Fp16 Quantization** this reduces precision from FP32 to FP16 often used on hardware with GPUs to speed inference while keeping accuracy close to full precision yet speeding up the process.

```python
import onnx
from onnxconverter_common import float16

# ---------------------------
# Step 1: Define paths
# ---------------------------
onnx_model_path = "rfdetr_small.onnx"          # Original FP32 ONNX model
fp16_model_path = "rfdetr_small_fp16.onnx"    # Path to save FP16 model

# ---------------------------
# Step 2: Load the ONNX model
# ---------------------------
model = onnx.load(onnx_model_path)

# ---------------------------
# Step 3: Convert model weights to FP16
# ---------------------------
# float16 conversion reduces memory usage and speeds up GPU inference
fp16_model = float16.convert_float_to_float16(model)

# ---------------------------
# Step 4: Save the FP16 ONNX model
# ---------------------------
onnx.save(fp16_model, fp16_model_path)

# ---------------------------
# ✅ Explanation:
# 1. Converts FP32 weights to FP16 for GPU acceleration.
# 2. Reduces model memory footprint by ~50%.
# 3. Maintains accuracy close to FP32 (slight precision loss possible).
# 4. Best used with GPU runtime; CPUs do not benefit much.
# ---------------------------
print(f"✅ FP16 Mixed Precision applied! Model saved at {fp16_model_path}")

```

**3:Pruning + Quantization** pruning removes unimportant weoghts usually small-magnitude from th network reduces models size and computations which increease FPS in real-time.

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.tools import onnx_pruning

# ---------------------------
# Step 1: Define paths
# ---------------------------
onnx_model_path = "rfdetr_small.onnx"          # existing ONNX model
pruned_model_path = "rfdetr_small_pruned.onnx" # Path for pruned model
quantized_model_path = "rfdetr_small_pruned_int8.onnx"  # Path for INT8 quantized model

# ---------------------------
# Step 2: Prune unimportant weights
# ---------------------------
# Using ONNX Runtime pruning tool (magnitude-based)
# Removes low-magnitude weights to reduce model size
pruned_model = onnx_pruning.prune_model(
    model_path=onnx_model_path,
    prune_ratio=0.2,           # Remove 20% of small-magnitude weights
    output_model_path=pruned_model_path
)

# ---------------------------
# Step 3: Apply dynamic INT8 quantization
# ---------------------------
quantize_dynamic(
    model_input=pruned_model_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QInt8
)

# ---------------------------
# ✅ Explanation:
# 1. Pruning reduces unnecessary weights → fewer computations, faster inference.
# 2. Dynamic INT8 quantization reduces model size and speeds up CPU inference.
# 3. The final ONNX model is optimized for real-time deployment.
# ---------------------------
print(f"✅ Pruned + Quantized ONNX model saved at {quantized_model_path}")
```

These optimization techniques are applied during inference to reduce model;s load on memory and remove extra operations and combine small essential operations to make model optimized task centric.this make model more suitable for the operation and reduce computational cost allowing the model to run faster, maintain higher FPS, and respond quickly to incoming data in real-time applications.


## Quickstart
1) Install deps
```bash
pip install -r ONNX_/ONNXruntime_/requirements.txt
```

2) Export a PyTorch model (RF-DETR example)
```bash
python ONNX_/ONNXruntime_/rf_detr_to_onnx.py
```
Output: `rfdetr_small.onnx`

3) Optimize graph
```bash
python ONNX_/ONNXruntime_/graphoptimization_.py --model rfdetr_small.onnx --out rfdetr_small_optimized.onnx
```

4) (Optional) Fuse operators
```bash
python ONNX_/ONNXruntime_/operatorfusion_.py --model rfdetr_small_optimized.onnx --out rfdetr_small_fused.onnx
```

5) (Optional) Quantize to INT8
```bash
python ONNX_/ONNXruntime_/Quantization_.py --model rfdetr_small_fused.onnx --out rfdetr_small_int8.onnx
```

## Notes
- Match input shape and dynamic axes with your export script
- Use Opset 17+ if supported by your environment
- For GPU inference, install `onnxruntime-gpu` with a compatible CUDA version


