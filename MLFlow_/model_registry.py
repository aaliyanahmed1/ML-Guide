"""MLflow Model Registry Example with RF-DETR Object Detection Models.

This module demonstrates how to use the MLflow Model Registry for managing RF-DETR
object detection models. It shows how to register models to the registry, transition
models between lifecycle stages, and load models from the registry for inference.
"""

import os
import time
import mlflow
import mlflow.pyfunc
from PIL import Image

# Import Roboflow Inference SDK for RF-DETR models
try:
    from inference import get_model
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Roboflow Inference SDK not found. Install with: pip install inference")

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_and_register_model(model_name, model_variant="base", version_description=None):
    """Load an RF-DETR model and register it to the MLflow Model Registry.
    
    Args:
        model_name (str): Name to register the model under
        model_variant (str, optional): RF-DETR variant ('nano', 'small', 'base', 'large'). Defaults to "base".
        version_description (str, optional): Description for the model version. Defaults to None.
        
    Returns:
        RegisteredModel: The registered model object
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
    
    # Path to test image (update with your image path)
    image_path = "test_image.jpg"  # Replace with your test image
    confidence_threshold = 0.5
    
    # Log the model
    with mlflow.start_run() as run:
        # Load model and time it
        t0 = time.perf_counter()
        model = get_model(model_id)
        load_time = (time.perf_counter() - t0) * 1000.0  # ms
        
        # Log model parameters
        mlflow.log_params({
            "model_variant": model_variant,
            "model_id": model_id,
            "model_size_mb": model_sizes.get(model_variant, 0),
            "confidence_threshold": confidence_threshold
        })
        
        # Log load time metric
        mlflow.log_metric("load_time_ms", load_time)
        
        # Run inference if test image exists
        if os.path.exists(image_path):
            # Run inference and time it
            t0 = time.perf_counter()
            outputs = model.infer(image_path, confidence=confidence_threshold)
            infer_time = (time.perf_counter() - t0) * 1000.0  # ms
            
            # Log inference time metric
            mlflow.log_metric("inference_time_ms", infer_time)
        
        # Log model as a custom PyFunc model
        # Note: This is a simplified approach as RF-DETR models from Roboflow
        # don't directly support MLflow's standard logging methods
        mlflow.pyfunc.log_model(
            artifact_path="rfdetr_model",
            python_model=None,  # We're just logging metadata, not the actual model
            artifacts={"model_info": "model_info.txt"},
            code_path=[],
            conda_env={"dependencies": ["inference"]}
        )
        model_uri = f"runs:/{run.info.run_id}/rfdetr_model"
        
        # Register the model
        registered_model = mlflow.register_model(model_uri, model_name)
        print(f"RF-DETR {model_variant} model registered: {registered_model.name} version {registered_model.version}")
        
        # Add description if provided
        if version_description:
            client = mlflow.tracking.MlflowClient()
            client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=version_description
            )
    
    return registered_model

def transition_model_stage(model_name, version, stage):
    """Transition a RF-DETR model version to a different stage.
    
    Args:
        model_name (str): Name of the registered model
        version (int): Version number of the model
        stage (str): Target stage ('None', 'Staging', 'Production', 'Archived')
        
    Returns:
        ModelVersion: The updated model version
    """
    client = mlflow.tracking.MlflowClient()
    model_version = client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    print(f"RF-DETR model {model_name} version {version} transitioned to {stage}")
    return model_version

def load_production_model(model_name):
    """Load the production version of an RF-DETR model from the registry.
    
    Args:
        model_name (str): Name of the registered model
        
    Returns:
        dict: Model metadata (since actual RF-DETR models are loaded via Roboflow Inference SDK)
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("Roboflow Inference SDK not available. Install with: pip install inference")
    
    # Get model info from registry
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(model_name, stages=["Production"])[0]
    run_id = model_version.run_id
    
    # Get run info to extract model variant
    run = client.get_run(run_id)
    model_variant = run.data.params.get("model_variant", "base")
    model_id = run.data.params.get("model_id", "rfdetr-base/1")
    
    # Load the actual model using Roboflow Inference SDK
    print(f"Loading RF-DETR {model_variant} model from production stage")
    model = get_model(model_id)
    
    # Return model metadata
    return {
        "model": model,
        "variant": model_variant,
        "model_id": model_id,
        "run_id": run_id
    }

# Try to import utils module for enhanced output
try:
    from utils import print_section, print_subsection, print_env_info
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    # Define minimal versions of utility functions
    def print_section(title):
        """Print a section title."""
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    
    def print_subsection(title):
        """Print a subsection title."""
        print(f"\n{'-' * 40}\n{title}\n{'-' * 40}")
    
    def print_env_info():
        """Print basic environment information."""
        print("Environment Information:")
        print(f"- Python: {os.sys.version.split()[0]}")
        print(f"- MLflow: {mlflow.__version__}")
        print(f"- Working Directory: {os.getcwd()}")


def run_inference_on_image(model_info, image_path="test_image.jpg", confidence=0.5):
    """Run inference on an image using the loaded RF-DETR model.
    
    Args:
        model_info (dict): Model metadata from load_production_model
        image_path (str): Path to the image for inference
        confidence (float): Confidence threshold for detections
        
    Returns:
        dict: Inference results
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image {image_path} not found. Skipping inference.")
        return None
    
    model = model_info["model"]
    variant = model_info["variant"]
    
    print(f"Running inference with RF-DETR {variant} model on {image_path}")
    
    # Time the inference
    t0 = time.perf_counter()
    results = model.infer(image_path, confidence=confidence)
    inference_time = (time.perf_counter() - t0) * 1000.0  # ms
    
    # Print results
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Detected {len(results['predictions'])} objects")
    
    # Print detection details
    for i, pred in enumerate(results['predictions']):
        print(f"  {i+1}. {pred['class']} ({pred['confidence']:.2f}): "
              f"[{pred['x']:.1f}, {pred['y']:.1f}, {pred['width']:.1f}, {pred['height']:.1f}]")
    
    return results


if __name__ == "__main__":
    # Print environment information
    if UTILS_AVAILABLE:
        print_env_info()
    else:
        print_env_info()
    
    print_section("MLflow Model Registry with RF-DETR Object Detection Models")
    
    # Check if Roboflow Inference SDK is available
    if not RFDETR_AVAILABLE:
        print("Error: Roboflow Inference SDK not available. Install with: pip install inference")
        print("Exiting example...")
        exit(1)
    
    # Register base model
    print_subsection("Registering RF-DETR Base Model")
    model_name = "RF-DETR-ObjectDetection"
    registered_model = train_and_register_model(
        model_name, 
        model_variant="base",
        version_description="Initial RF-DETR base model"
    )
    
    # Transition to staging
    print_subsection("Transitioning Base Model to Staging")
    transition_model_stage(model_name, registered_model.version, "Staging")
    
    # Register a new version (small variant)
    print_subsection("Registering RF-DETR Small Model")
    new_version = train_and_register_model(
        model_name, 
        model_variant="small",
        version_description="RF-DETR small variant for faster inference"
    )
    
    # Transition new version to production
    print_subsection("Transitioning Small Model to Production")
    transition_model_stage(model_name, new_version.version, "Production")
    
    # Load the production model
    print_subsection("Loading Production Model")
    production_model_info = load_production_model(model_name)
    
    # Run inference with the production model
    print_subsection("Running Inference with Production Model")
    test_image = "test_image.jpg"  # Replace with your test image
    if os.path.exists(test_image):
        inference_results = run_inference_on_image(production_model_info, test_image)
    else:
        print(f"Warning: Test image {test_image} not found. Create or download a test image to run inference.")
    
    # Archive the old version
    print_subsection("Archiving Base Model")
    transition_model_stage(model_name, registered_model.version, "Archived")
    
    # Register nano and large variants for comparison
    print_subsection("Registering Additional RF-DETR Variants for Comparison")
    nano_version = train_and_register_model(
        model_name, 
        model_variant="nano",
        version_description="RF-DETR nano variant for edge devices"
    )
    
    large_version = train_and_register_model(
        model_name, 
        model_variant="large",
        version_description="RF-DETR large variant for highest accuracy"
    )
    
    print_section("Model Registry Example Complete")
    print("\nView the registered models in the MLflow UI:")
    print("  mlflow ui")
    print("\nOr use the MLflow API to query the Model Registry programmatically.")
    print("\nFor more information, visit: https://www.mlflow.org/docs/latest/model-registry.html")