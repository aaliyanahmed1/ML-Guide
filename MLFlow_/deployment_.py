"""MLflow Model Deployment Example with RF-DETR Object Detection Models.

This module demonstrates how to deploy RF-DETR object detection models using Flask
and export to different formats like ONNX for optimized inference.
"""

import os
import time
import json
import io
import base64
import sys
import mlflow
import mlflow.pyfunc
from flask import Flask, request, jsonify

# Try to import Roboflow Inference SDK
RFDETR_AVAILABLE = False
try:
    from inference import get_model
    from PIL import Image
    RFDETR_AVAILABLE = True
except ImportError:
    print("WARNING: Roboflow Inference SDK not available. Install with: pip install inference")

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def load_and_log_model(model_variant="base"):
    """Load an RF-DETR model and log it to MLflow.
    
    Args:
        model_variant (str, optional): RF-DETR variant ('nano', 'small', 'base', 'large'). Defaults to "base".
    
    Returns:
        str: The run ID of the logged model
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("Roboflow Inference SDK not available. Install with: pip install inference")
    
    # Map variant to model ID
    variant_map = {
        "nano": "rfdetr-nano/1",
        "small": "rfdetr-small/1",
        "base": "rfdetr-base/1",
        "large": "rfdetr-large/1"
    }
    model_id = variant_map.get(model_variant, "rfdetr-base/1")
    
    # Approximate model sizes in MB
    model_sizes = {
        "nano": 15,
        "small": 35,
        "base": 45,
        "large": 300
    }
    
    # Load model and time it
    print(f"Loading RF-DETR {model_variant} model...")
    t0 = time.perf_counter()
    model = get_model(model_id)
    load_time = (time.perf_counter() - t0) * 1000.0  # ms
    
    # Log model to MLflow
    with mlflow.start_run(run_name=f"rfdetr_{model_variant}_deployment") as run:
        # Log parameters
        mlflow.log_params({
            "model_variant": model_variant,
            "model_id": model_id,
            "model_size_mb": model_sizes.get(model_variant, 0),
            "confidence_threshold": 0.5  # Default confidence threshold
        })
        
        # Log metrics
        mlflow.log_metric("load_time_ms", load_time)
        
        # Create a simple wrapper for the model
        # Note: This is a simplified approach as RF-DETR models from Roboflow
        # don't directly support MLflow's standard logging methods
        mlflow.pyfunc.log_model(
            artifact_path="rfdetr_model",
            python_model=None,  # We're just logging metadata, not the actual model
            artifacts={"model_info": "model_info.txt"},
            code_path=[],
            conda_env={"dependencies": ["inference"]},
            registered_model_name=f"RF-DETR-{model_variant.capitalize()}"
        )
        
        run_id = run.info.run_id
        print(f"Model logged with run_id: {run_id}")
        
        # Save model metadata for inference
        model_info = {
            "model_variant": model_variant,
            "model_id": model_id,
            "confidence_threshold": 0.5
        }
        with open("model_info.txt", "w") as f:
            json.dump(model_info, f)
        mlflow.log_artifact("model_info.txt")
        
        return run_id

def deploy_model_locally(run_id, port=5000):
    """Deploy an RF-DETR model locally using Flask.
    
    Args:
        run_id (str): The run ID of the model to deploy
        port (int, optional): Port to run the Flask app on. Defaults to 5000.
        
    Note:
        This function creates a Flask app that accepts image data in base64 format
        and returns object detection results from the RF-DETR model.
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("Roboflow Inference SDK not available. Install with: pip install inference")
    
    # Get model info from MLflow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    model_variant = run.data.params.get("model_variant", "base")
    model_id = run.data.params.get("model_id", "rfdetr-base/1")
    confidence_threshold = float(run.data.params.get("confidence_threshold", 0.5))
    
    # Load the actual model using Roboflow Inference SDK
    print(f"Loading RF-DETR {model_variant} model (ID: {model_id})...")
    model = get_model(model_id)
    
    # Create Flask app
    app = Flask(__name__)
    
    @app.route("/predict", methods=["POST"])
    def predict():
        # Get input data from request
        data = request.json
        
        if not data or "image" not in data:
            return jsonify({"error": "Invalid input. Expected 'image' key with base64 encoded image."}), 400
        
        try:
            # Decode base64 image
            image_data = base64.b64decode(data["image"])
            image = Image.open(io.BytesIO(image_data))
            
            # Run inference
            t0 = time.perf_counter()
            results = model.infer(image, confidence=confidence_threshold)
            inference_time = (time.perf_counter() - t0) * 1000.0  # ms
            
            # Process results
            detections = []
            for detection in results:
                detections.append({
                    "class": detection.get("class", "unknown"),
                    "confidence": detection.get("confidence", 0),
                    "x": detection.get("x", 0),
                    "y": detection.get("y", 0),
                    "width": detection.get("width", 0),
                    "height": detection.get("height", 0)
                })
            
            return jsonify({
                "model_variant": model_variant,
                "model_id": model_id,
                "inference_time_ms": inference_time,
                "detections": detections,
                "num_detections": len(detections)
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Health check endpoint
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})
    
    # Start Flask app
    print(f"Starting model server on port {port}...")
    app.run(host="0.0.0.0", port=port)

def export_model_to_different_formats(run_id):
    """Export an RF-DETR model to different formats (primarily ONNX).
    
    Args:
        run_id (str): The run ID of the model to export
        
    Note:
        For RF-DETR models, we focus on ONNX export which is commonly used
        for deploying object detection models in production environments.
        The actual export process would require additional steps with the
        Roboflow API or SDK that are beyond the scope of this example.
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("Roboflow Inference SDK not available. Install with: pip install inference")
    
    # Get model info from MLflow
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    model_variant = run.data.params.get("model_variant", "base")
    model_id = run.data.params.get("model_id", "rfdetr-base/1")
    
    print(f"Preparing to export RF-DETR {model_variant} model (ID: {model_id})...")
    
    # Note: This is a simplified example. In a real-world scenario,
    # you would use the Roboflow API to export the model to ONNX format.
    print("\nExporting RF-DETR models to ONNX format:")
    print("1. Visit https://docs.roboflow.com/inference/exporting")
    print("2. Use the Roboflow web interface to export your model to ONNX format")
    print("3. Alternatively, use the Roboflow API with the following code:")
    print("""
    import requests
    
    API_KEY = "your_api_key"
    WORKSPACE_ID = "your_workspace_id"
    PROJECT_ID = "your_project_id"
    VERSION_ID = "1"  # Replace with your version ID
    
    url = f"https://api.roboflow.com/{WORKSPACE_ID}/{PROJECT_ID}/{VERSION_ID}/export?api_key={API_KEY}&format=onnx"
    response = requests.post(url)
    export_url = response.json()["export_url"]
    print(f"Model export URL: {export_url}")
    """)
    
    # Log the export attempt in MLflow
    with mlflow.start_run(run_id=run_id, nested=True):
        mlflow.log_param("export_attempt", "ONNX")
        mlflow.log_param("export_timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))
        
    print(f"\nExport information logged to MLflow run: {run_id}")
    print("Note: Actual model export requires Roboflow API access.")
    
    # For demonstration purposes, we'll create a placeholder file
    export_info_path = f"rfdetr_{model_variant}_export_info.txt"
    with open(export_info_path, "w") as f:
        f.write(f"RF-DETR {model_variant} model (ID: {model_id})\n")
        f.write(f"Export attempted on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("For actual export, use the Roboflow API or web interface.\n")
    
    print(f"\nExport information saved to: {export_info_path}")

if __name__ == "__main__":
    # Import utils for enhanced output
    try:
        from utils import print_section, print_subsection, print_environment_info
        
        # Print environment information
        print_section("MLflow RF-DETR Model Deployment Example")
        print_environment_info()
    except ImportError:
        # If utils.py is not available, continue without enhanced output
        print("\n===== MLflow RF-DETR Model Deployment Example =====")
    
    # Check if Roboflow Inference SDK is available
    if not RFDETR_AVAILABLE:
        print("\nWARNING: Roboflow Inference SDK not available.")
        print("Install with: pip install inference")
        print("This example will not run without the SDK.")
        import sys
        sys.exit(1)
    
    # Example usage
    print_subsection("Loading and Logging RF-DETR Model") if 'print_subsection' in locals() else print("\n--- Loading and Logging RF-DETR Model ---")
    
    # Choose a model variant
    model_variant = "base"  # Options: nano, small, base, large
    run_id = load_and_log_model(model_variant)
    print(f"RF-DETR {model_variant} model logged with run_id: {run_id}")
    
    # Deploy model locally
    print_subsection("Deploying RF-DETR Model Locally") if 'print_subsection' in locals() else print("\n--- Deploying RF-DETR Model Locally ---")
    print("Starting Flask server for model deployment...")
    print("\nAPI Usage Example:")
    print("1. Convert an image to base64")
    print("2. Send a POST request to http://localhost:5000/predict with JSON payload:")
    print("   {\"image\": \"<base64_encoded_image>\"}")
    print("3. The response will contain object detection results")
    print("\nStarting server...")
    deploy_model_locally(run_id)
    
    # Export model to different formats
    print_subsection("Exporting RF-DETR Model to Different Formats") if 'print_subsection' in locals() else print("\n--- Exporting RF-DETR Model to Different Formats ---")
    export_model_to_different_formats(run_id)
    
    print("\nRF-DETR model deployment example completed.")
    print("The model has been loaded, deployed locally, and export information provided.")
    print("To view the model in the MLflow UI, run: mlflow ui")