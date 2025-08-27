import os
import mlflow
import mlflow.data
import mlflow.sklearn  # For consistency, though not used directly
from ultralytics import YOLO

def main():
    """Train YOLO model and log results to MLflow."""
    # Enable autologging for PyTorch (works for YOLO models too)
    mlflow.pytorch.autolog()

    # Paths (update with your dataset + pretrained YOLO model)
    data_yaml = "datasets/custom_dataset/data.yaml"  # YOLO dataset config file
    pretrained_model = "yolov8n.pt"  # Pretrained YOLO checkpoint
    epochs = 20
    img_size = 640
    batch_size = 16

    # Start MLflow run
    with mlflow.start_run(run_name="yolo_object_detection_training"):
        # Initialize YOLO model
        model = YOLO(pretrained_model)

        # Train model
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            name="yolo-mlflow-exp"
        )

        # Log model parameters to MLflow
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("image_size", img_size)
        mlflow.log_param("batch_size", batch_size)

        # Evaluate model
        metrics = model.val()
        results_dict = metrics.results_dict
        mlflow.log_metrics({
            "mAP50": results_dict.get("metrics/mAP50(B)", 0.0),
            "mAP50-95": results_dict.get("metrics/mAP50-95(B)", 0.0),
            "precision": results_dict.get("metrics/precision(B)", 0.0),
            "recall": results_dict.get("metrics/recall(B)", 0.0),
        })

        # Log trained model checkpoint
        mlflow.pytorch.log_model(model=model.model, artifact_path="yolo_model")

    print("YOLO training & evaluation logged to MLflow ✅")

if __name__ == "__main__":
    main()
