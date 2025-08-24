import torch
import onnx
import onnxruntime as ort

# Try different import approaches for RF-DETR
try:
    # First try the dedicated rfdetr package
    from rfdetr import RfDetrForObjectDetection
    from transformers import AutoImageProcessor
    RFDETR_AVAILABLE = True
    RFDETR_SOURCE = "rfdetr"
except ImportError:
    try:
        # Fallback to transformers library
        from transformers import AutoImageProcessor, RfDetrForObjectDetection
        RFDETR_AVAILABLE = True
        RFDETR_SOURCE = "transformers"
    except ImportError:
        try:
            # Last fallback to AutoModelForObjectDetection
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            RFDETR_AVAILABLE = True
            RfDetrForObjectDetection = AutoModelForObjectDetection
            RFDETR_SOURCE = "auto"
        except ImportError:
            RFDETR_AVAILABLE = False
            print("Warning: rfdetr and transformers libraries not available")

# Try to import supervision for visualization
try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False
    print("Warning: supervision library not available")

def export_rfdetr_to_onnx(model_name="roboflow/rf-detr-base", output_path="rfdetr.onnx"):
    """
    Export RF-DETR model to ONNX format
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("rfdetr or transformers library is required. Install with: pip install rfdetr supervision transformers torch onnx onnxruntime")
    
    print(f"Loading model: {model_name}")
    print(f"Using RF-DETR from: {RFDETR_SOURCE}")
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = RfDetrForObjectDetection.from_pretrained(model_name)
    model.eval()
    
    # Create dummy input (batch_size=1, channels=3, height=640, width=640)
    dummy_input = torch.randn(1, 3, 640, 640)
    
    print("Exporting to ONNX...")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to: {output_path}")
    
    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    # Test ONNX inference
    print("Testing ONNX inference...")
    ort_session = ort.InferenceSession(output_path)
    
    # Prepare input
    input_name = ort_session.get_inputs()[0].name
    test_input = dummy_input.numpy()
    
    # Run inference
    outputs = ort_session.run(None, {input_name: test_input})
    print(f"ONNX inference successful! Output shape: {outputs[0].shape}")
    
    return output_path

def export_pytorch_to_onnx(model, output_path="model.onnx", input_shape=(1, 3, 640, 640)):
    """
    Generic function to export any PyTorch model to ONNX
    """
    print("Exporting PyTorch model to ONNX...")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to: {output_path}")
    
    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed!")
    
    return output_path

def test_rfdetr_inference(model_name="roboflow/rf-detr-base"):
    """
    Test RF-DETR inference with supervision for visualization
    """
    if not RFDETR_AVAILABLE:
        print("RF-DETR not available")
        return
    
    if not SUPERVISION_AVAILABLE:
        print("Supervision not available for visualization")
        return
    
    print(f"Testing RF-DETR inference with {model_name}")
    
    # Load model and processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = RfDetrForObjectDetection.from_pretrained(model_name)
    model.eval()
    
    # Create a dummy image for testing
    dummy_image = torch.randn(3, 640, 640)
    
    # Preprocess
    inputs = processor(images=dummy_image, return_tensors="pt")
    
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("RF-DETR inference test successful!")
    print(f"Output keys: {list(outputs.keys())}")
    
    return outputs

if __name__ == "__main__":
    try:
        # Try to export RF-DETR base model
        export_rfdetr_to_onnx()
        
        # Test inference if supervision is available
        if SUPERVISION_AVAILABLE:
            test_rfdetr_inference()
            
    except Exception as e:
        print(f"Error exporting RF-DETR: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install rfdetr supervision transformers torch onnx onnxruntime")
