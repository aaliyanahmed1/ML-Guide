"""MLflow Model Versioning Example with RF-DETR Object Detection Models.

This module demonstrates how to create and manage different versions of RF-DETR object detection
models using MLflow. It shows how to log different variants (nano, small, base, large) as separate
model versions, compare their performance metrics, and identify the best model version based on
specified criteria.
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

def create_and_log_model_version(experiment_name, run_name, model_variant, confidence_threshold=0.5):
    """Create and log a new version of an RF-DETR model.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name for this particular run
        model_variant (str): RF-DETR variant ('nano', 'small', 'base', 'large')
        confidence_threshold (float, optional): Detection confidence threshold. Defaults to 0.5.
        
    Returns:
        str: The run ID of the logged model
    """
    if not RFDETR_AVAILABLE:
        raise ImportError("Roboflow Inference SDK not available. Install with: pip install inference")
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
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
    
    # Start a new run
    with mlflow.start_run(run_name=run_name):
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
        
        # Run inference if test image exists
        if os.path.exists(image_path):
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # Run inference and time it
            t0 = time.perf_counter()
            outputs = model.infer(image_path, confidence=confidence_threshold)
            infer_time = (time.perf_counter() - t0) * 1000.0  # ms
            
            # Process predictions
            if isinstance(outputs, list):
                data = outputs[0]
            else:
                data = outputs
            
            preds = data.get("predictions", data)
            if isinstance(preds, dict):
                preds = [preds]
            
            # Extract detection metrics
            num_detections = len(preds)
            confidences = [float(p.get("confidence", p.get("score", 0.0))) for p in preds]
            avg_confidence = sum(confidences) / max(1, len(confidences))
            
            # Log metrics
            mlflow.log_metrics({
                "load_time_ms": load_time,
                "inference_time_ms": infer_time,
                "num_detections": num_detections,
                "avg_confidence": avg_confidence
            })
            
            # Calculate a performance score (lower is better)
            # This combines speed and detection confidence
            performance_score = infer_time / (avg_confidence + 0.1)  # Avoid division by zero
            mlflow.log_metric("performance_score", performance_score)
            
            print(f"Model version logged - Variant: {model_variant}, Detections: {num_detections}, Inference time: {infer_time:.2f}ms")
        else:
            # Log only load time if no test image
            mlflow.log_metric("load_time_ms", load_time)
            print(f"Model version logged - Variant: {model_variant}, Load time: {load_time:.2f}ms (No test image found)")
        
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
        
        return mlflow.active_run().info.run_id

def compare_model_versions(run_ids):
    """Compare different versions of RF-DETR models based on their metrics.
    
    Args:
        run_ids (list): List of run IDs to compare
    """
    client = mlflow.tracking.MlflowClient()
    
    print("\nRF-DETR Model Version Comparison:")
    print("-" * 80)
    print(f"{'Run ID':<36} {'Variant':<10} {'Size (MB)':<10} {'Infer Time (ms)':<15} {'Detections':<10} {'Avg Conf':<10}")
    print("-" * 80)
    
    for run_id in run_ids:
        run = client.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics
        
        # Get model variant and size
        variant = params.get('model_variant', 'N/A')
        size_mb = params.get('model_size_mb', 'N/A')
        
        # Get metrics
        infer_time = metrics.get("inference_time_ms", 0.0)
        num_detections = metrics.get("num_detections", 0)
        avg_confidence = metrics.get("avg_confidence", 0.0)
        
        print(f"{run_id:<36} {variant:<10} {size_mb:<10} {infer_time:<15.2f} {num_detections:<10} {avg_confidence:<10.4f}")

def get_best_model_version(run_ids, metric="performance_score", higher_is_better=False):
    """Get the best RF-DETR model version based on a metric.
    
    Args:
        run_ids (list): List of run IDs to compare
        metric (str, optional): Metric to use for comparison. Defaults to "performance_score".
        higher_is_better (bool, optional): Whether higher metric values are better.
            Defaults to False (for performance_score, lower is better).
        
    Returns:
        tuple: Best run ID and its metric value
    """
    client = mlflow.tracking.MlflowClient()
    
    best_metric_value = float("inf") if not higher_is_better else -float("inf")
    best_run_id = None
    
    for run_id in run_ids:
        run = client.get_run(run_id)
        metric_value = run.data.metrics.get(metric, float("inf") if not higher_is_better else -float("inf"))
        
        if (not higher_is_better and metric_value < best_metric_value) or \
           (higher_is_better and metric_value > best_metric_value):
            best_metric_value = metric_value
            best_run_id = run_id
    
    return best_run_id, best_metric_value

if __name__ == "__main__":
    # Import utils for enhanced output
    try:
        from utils import print_section, print_subsection, print_environment_info, visualize_metrics
        
        # Print environment information
        print_section("MLflow Model Versioning Example with RF-DETR")
        print_environment_info()
        has_utils = True
    except ImportError:
        # If utils.py is not available, continue without enhanced output
        print("\n===== MLflow Model Versioning Example with RF-DETR =====")
        has_utils = False
    
    # Check if Roboflow Inference SDK is available
    if not RFDETR_AVAILABLE:
        print("\nError: Roboflow Inference SDK not available.")
        print("Install with: pip install inference")
        print("Exiting example.")
        import sys
        sys.exit(1)
    
    # Create experiment
    experiment_name = "rfdetr_model_versioning_example"
    
    # Print section header
    if has_utils:
        print_subsection("Creating RF-DETR Model Versions")
    else:
        print("\n--- Creating RF-DETR Model Versions ---")
    
    print("This example will create and compare different RF-DETR model variants.")
    print("Models to compare: RF-DETR nano, small, base, and large")
    print("Note: This requires the Roboflow Inference SDK and a test image.")
    
    try:
        # Create and log four model versions with different RF-DETR variants
        print("\nCreating and logging RF-DETR model versions...")
        run_id_1 = create_and_log_model_version(experiment_name, "nano_version", model_variant="nano")
        run_id_2 = create_and_log_model_version(experiment_name, "small_version", model_variant="small")
        run_id_3 = create_and_log_model_version(experiment_name, "base_version", model_variant="base")
        run_id_4 = create_and_log_model_version(experiment_name, "large_version", model_variant="large")
        
        # Compare model versions
        run_ids = [run_id_1, run_id_2, run_id_3, run_id_4]
        compare_model_versions(run_ids)
        
        # Get best model version based on performance score (lower is better)
        best_run_id, best_metric_value = get_best_model_version(run_ids, metric="performance_score", higher_is_better=False)
        
        # Get client to fetch best run details
        client = mlflow.tracking.MlflowClient()
        best_run = client.get_run(best_run_id)
        best_variant = best_run.data.params.get("model_variant", "unknown")
        
        print(f"\nBest RF-DETR model version: {best_variant} (Run ID: {best_run_id}) with performance score: {best_metric_value:.4f}")
        
        # Visualize metrics if utils is available
        if has_utils:
            # Collect metrics for visualization
            metrics_data = {}
            for run_id in run_ids:
                run = client.get_run(run_id)
                variant = run.data.params.get("model_variant", "unknown")
                infer_time = run.data.metrics.get("inference_time_ms", 0)
                metrics_data[variant] = infer_time
            
            # Visualize metrics
            print_subsection("Metric Visualization")
            print("Displaying inference time comparison chart...")
            visualize_metrics(metrics_data, "RF-DETR Inference Time Comparison")
    
    except Exception as e:
        print(f"\nError running RF-DETR model versioning example: {str(e)}")
    
    print("\nModel versioning example completed. All models are logged to MLflow.")
    print("To view results in the MLflow UI, run: mlflow ui")