"""MLflow Model Deployment Example.

This module demonstrates how to deploy MLflow models using Flask and export to different formats.
"""

import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_and_log_model():
    """Train a model and log it to MLflow.
    
    Returns:
        str: The run ID of the logged model
    """
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Log model to MLflow
    with mlflow.start_run(run_name="deployment_model") as run:
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", model.score(X_test, y_test))
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="DeploymentModel"
        )
        
        run_id = run.info.run_id
        print(f"Model logged with run_id: {run_id}")
        
        # Save feature names for inference
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        pd.DataFrame({"feature_name": feature_names}).to_csv("feature_names.csv", index=False)
        mlflow.log_artifact("feature_names.csv")
        
        return run_id

def deploy_model_locally(run_id, port=5000):
    """Deploy a model locally using Flask.
    
    Args:
        run_id (str): The run ID of the model to deploy
        port (int, optional): Port to run the Flask app on. Defaults to 5000.
    """
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Create Flask app
    app = Flask(__name__)
    
    @app.route("/predict", methods=["POST"])
    def predict():
        # Get input data from request
        data = request.json
        
        if not data or "features" not in data:
            return jsonify({"error": "Invalid input. Expected 'features' key with array of values."}), 400
        
        # Convert input to DataFrame
        features = data["features"]
        if not isinstance(features, list):
            return jsonify({"error": "Features must be a list of values."}), 400
        
        # Make prediction
        try:
            # Convert to numpy array and reshape for single sample
            features_array = np.array(features).reshape(1, -1)
            prediction = model.predict(features_array)
            probability = model.predict_proba(features_array).tolist()
            
            return jsonify({
                "prediction": int(prediction[0]),
                "probability": probability[0],
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
    """Export a model to different formats (ONNX, TensorFlow, etc.).
    
    Args:
        run_id (str): The run ID of the model to export
    """
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    
    # Export as ONNX format
    # Note: This requires the mlflow-onnx extra dependency
    try:
        import mlflow.onnx
        onnx_output_path = "model.onnx"
        mlflow.onnx.save_model(mlflow.sklearn.load_model(model_uri), onnx_output_path)
        print(f"Model exported to ONNX format at {onnx_output_path}")
    except ImportError:
        print("ONNX export requires mlflow-onnx extra dependency")
    
    # Export as TensorFlow SavedModel format
    # Note: This requires the mlflow-tensorflow extra dependency
    try:
        import mlflow.tensorflow
        tf_output_path = "tf_model"
        mlflow.tensorflow.save_model(mlflow.sklearn.load_model(model_uri), tf_output_path)
        print(f"Model exported to TensorFlow format at {tf_output_path}")
    except ImportError:
        print("TensorFlow export requires mlflow-tensorflow extra dependency")

if __name__ == "__main__":
    # Train and log model
    run_id = train_and_log_model()
    
    # Export model to different formats
    # export_model_to_different_formats(run_id)
    
    # Deploy model locally
    # deploy_model_locally(run_id, port=5000)
    
    print("\nTo deploy the model locally, uncomment the deploy_model_locally line")
    print("To export the model to different formats, uncomment the export_model_to_different_formats line")