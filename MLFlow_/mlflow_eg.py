"""MLflow example for RF-DETR object detection model training and logging.

This module demonstrates how to use MLflow to track experiments with RF-DETR models,
including loading different variants (base, nano, small, large), running inference,
and comparing their performance metrics.
"""

import os
import time
import mlflow
import mlflow.data
import mlflow.pyfunc
from PIL import Image

# Import Roboflow Inference SDK for RF-DETR models
try:
    from inference import get_model
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Roboflow Inference SDK not found. Install with: pip install inference")

# Try to import the RF-DETR training API (optional)
try:
    from rfdetr import RFDETRBase  # type: ignore
    RFDETR_TRAIN_AVAILABLE = True
except ImportError:
    RFDETR_TRAIN_AVAILABLE = False

def load_rfdetr_model(model_variant="base"):
    """Load an RF-DETR model variant using Roboflow Inference SDK.
    
    Args:
        model_variant (str): RF-DETR variant to load ('nano', 'small', 'base', 'large')
        
    Returns:
        model: Loaded RF-DETR model
        dict: Model metadata including size and variant
    """
    variant_map = {
        "nano": "rfdetr-nano/1",
        "small": "rfdetr-small/1",
        "base": "rfdetr-base/1",
        "large": "rfdetr-large/1"
    }
    
    model_id = variant_map.get(model_variant, "rfdetr-base/1")
    
    t0 = time.perf_counter()
    model = get_model(model_id)
    load_time = (time.perf_counter() - t0) * 1000.0  # ms
    
    # Approximate model sizes in MB
    model_sizes = {
        "nano": 15,
        "small": 35,
        "base": 45,
        "large": 300
    }
    
    metadata = {
        "variant": model_variant,
        "model_id": model_id,
        "size_mb": model_sizes.get(model_variant, 0),
        "load_time_ms": load_time
    }
    
    return model, metadata


def evaluate_rfdetr_model(model, image_path, confidence_threshold=0.5):
    """Run inference with RF-DETR model and calculate metrics.
    
    Args:
        model: RF-DETR model from Roboflow Inference SDK
        image_path (str): Path to test image
        confidence_threshold (float): Detection confidence threshold
        
    Returns:
        dict: Metrics including inference time and detection counts
    """
    # Ensure image exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
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
    
    metrics = {
        "inference_time_ms": infer_time,
        "num_detections": num_detections,
        "avg_confidence": avg_confidence,
        "image_width": image.width,
        "image_height": image.height
    }
    
    return metrics


def main():
    """Run RF-DETR model variants and log results to MLflow."""
    if not RFDETR_AVAILABLE:
        print("Error: Roboflow Inference SDK not available. Install with: pip install inference")
        return
    
    # Path to test image (update with your image path)
    image_path = "test_image.jpg"  # Replace with your test image
    confidence_threshold = 0.5
    
    # RF-DETR variants to compare
    variants = ["base", "nano", "small", "large"]
    
    for variant in variants:
        # Start MLflow run for this variant
        with mlflow.start_run(run_name=f"rfdetr_{variant}_evaluation"):
            try:
                # Load model
                model, metadata = load_rfdetr_model(variant)
                
                # Log model parameters
                mlflow.log_params({
                    "model_variant": variant,
                    "model_id": metadata["model_id"],
                    "model_size_mb": metadata["size_mb"],
                    "confidence_threshold": confidence_threshold
                })
                
                # Evaluate model
                metrics = evaluate_rfdetr_model(model, image_path, confidence_threshold)
                
                # Log metrics
                mlflow.log_metrics({
                    "load_time_ms": metadata["load_time_ms"],
                    "inference_time_ms": metrics["inference_time_ms"],
                    "num_detections": metrics["num_detections"],
                    "avg_confidence": metrics["avg_confidence"]
                })
                
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
                
                print(f"RF-DETR {variant} evaluation logged to MLflow ✅")
                
            except Exception as e:
                print(f"Error evaluating RF-DETR {variant}: {str(e)}")
    
    print("\nRF-DETR model comparison complete. View results in MLflow UI.")

def train_rfdetr_model():
    """Train RF-DETR model if rfdetr package is available.
    
    This function demonstrates how to fine-tune an RF-DETR model and log the training
    process to MLflow. It requires the optional rfdetr package to be installed.
    
    Returns:
        bool: True if training was successful, False otherwise
    """
    if not RFDETR_TRAIN_AVAILABLE:
        print("RF-DETR training package not available. Install with: pip install rfdetr")
        return False
    
    # Dataset directory (update with your COCO-format dataset path)
    dataset_dir = "dataset"  # expects train/, valid/, test/ with *_annotations.coco.json
    
    # Training hyperparameters
    epochs = 5
    batch_size = 4
    grad_accum_steps = 4
    learning_rate = 1e-4
    output_dir = "rf_detr_finetuned"
    
    with mlflow.start_run(run_name="rfdetr_training"):
        try:
            # Log training parameters
            mlflow.log_params({
                "epochs": epochs,
                "batch_size": batch_size,
                "grad_accum_steps": grad_accum_steps,
                "learning_rate": learning_rate,
                "dataset_dir": dataset_dir
            })
            
            # Create base RF-DETR model
            model = RFDETRBase()
            
            # Start training with timing
            t0 = time.perf_counter()
            
            # Train model
            model.train(
                dataset_dir=dataset_dir,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum_steps=grad_accum_steps,
                lr=learning_rate,
                output_dir=output_dir
            )
            
            train_time = time.perf_counter() - t0
            
            # Log training time
            mlflow.log_metric("training_time_seconds", train_time)
            
            # Log model artifacts
            mlflow.log_artifact(output_dir)
            
            print(f"RF-DETR training completed in {train_time:.2f} seconds")
            print(f"Model and checkpoints saved to '{output_dir}'")
            return True
            
        except Exception as e:
            print(f"Error during RF-DETR training: {str(e)}")
            return False


if __name__ == "__main__":
    # Import utils for enhanced output
    try:
        from utils import print_section, print_subsection, print_environment_info, visualize_metrics
        
        # Print environment information
        print_section("MLflow RF-DETR Example")
        print_environment_info()
        has_utils = True
    except ImportError:
        # If utils.py is not available, continue without enhanced output
        print("\n===== MLflow RF-DETR Example =====")
        has_utils = False
    
    # Print section header
    if has_utils:
        print_subsection("RF-DETR Model Comparison")
    else:
        print("\n--- RF-DETR Model Comparison ---")
    
    print("This example will compare different RF-DETR model variants using MLflow.")
    print("Models to compare: RF-DETR nano, small, base, and large")
    print("Note: This requires the Roboflow Inference SDK and a test image.")
    
    try:
        main()
        print("\nRF-DETR model comparison completed successfully.")
        
        # Optionally run training if available
        if RFDETR_TRAIN_AVAILABLE:
            if has_utils:
                print_subsection("RF-DETR Model Training")
            else:
                print("\n--- RF-DETR Model Training ---")
            
            print("Attempting to train RF-DETR model...")
            train_rfdetr_model()
        
    except Exception as e:
        print(f"\nError running RF-DETR example: {str(e)}")
        print("This example requires the Roboflow Inference SDK and a test image.")
        print("To run this example, install the SDK: pip install inference")
    
    print("\nTo view results in the MLflow UI, run: mlflow ui")
