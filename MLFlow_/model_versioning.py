"""MLflow Model Versioning Example.

This module demonstrates how to create and manage different versions of ML models using MLflow.
"""

import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def create_and_log_model_version(experiment_name, run_name, n_estimators, max_depth=None):
    """Create and log a new version of a model with different hyperparameters.
    
    Args:
        experiment_name (str): Name of the MLflow experiment
        run_name (str): Name for this particular run
        n_estimators (int): Number of trees in the forest
        max_depth (int, optional): Maximum depth of the trees. Defaults to None.
        
    Returns:
        str: The run ID of the logged model
    """
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start a new run
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model version logged with accuracy: {accuracy:.4f}")
        return mlflow.active_run().info.run_id

def compare_model_versions(run_ids):
    """Compare different versions of a model.
    
    Args:
        run_ids (list): List of run IDs to compare
    """
    client = mlflow.tracking.MlflowClient()
    
    print("\nModel Version Comparison:")
    print("-" * 60)
    print(f"{'Run ID':<36} {'n_estimators':<12} {'max_depth':<10} {'Accuracy':<10}")
    print("-" * 60)
    
    for run_id in run_ids:
        run = client.get_run(run_id)
        params = run.data.params
        metrics = run.data.metrics
        
        n_estimators = params.get("n_estimators", "N/A")
        max_depth = params.get("max_depth", "None")
        accuracy = metrics.get("accuracy", 0.0)
        
        print(f"{run_id:<36} {n_estimators:<12} {max_depth:<10} {accuracy:<10.4f}")

def get_best_model_version(run_ids, metric="accuracy"):
    """Get the best model version based on a metric.
    
    Args:
        run_ids (list): List of run IDs to compare
        metric (str, optional): Metric to use for comparison. Defaults to "accuracy".
        
    Returns:
        tuple: Best run ID and its metric value
    """
    client = mlflow.tracking.MlflowClient()
    
    best_metric_value = -float("inf")
    best_run_id = None
    
    for run_id in run_ids:
        run = client.get_run(run_id)
        metric_value = run.data.metrics.get(metric, -float("inf"))
        
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_run_id = run_id
    
    return best_run_id, best_metric_value

if __name__ == "__main__":
    # Example usage
    experiment_name = "RandomForest_Versions"
    
    # Create different model versions
    run_ids = [
        create_and_log_model_version(experiment_name, "version_1", n_estimators=100, max_depth=10),
        create_and_log_model_version(experiment_name, "version_2", n_estimators=200, max_depth=15),
        create_and_log_model_version(experiment_name, "version_3", n_estimators=300, max_depth=None),
    ]
    
    # Compare model versions
    compare_model_versions(run_ids)
    
    # Get best model version
    best_run_id, best_accuracy = get_best_model_version(run_ids)
    print(f"\nBest model version: {best_run_id} with accuracy: {best_accuracy:.4f}")