"""MLflow Reproducibility Example.

This module demonstrates how to ensure reproducibility in machine learning experiments using MLflow.
"""

import os
import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def log_environment_info():
    """Log environment information to ensure reproducibility.
    
    Returns:
        str: The run ID of the logged environment information
    """
    import sys
    import platform
    import sklearn
    import pandas as pd
    
    with mlflow.start_run(run_name="environment_info") as run:
        # Log Python version
        mlflow.log_param("python_version", platform.python_version())
        
        # Log OS information
        mlflow.log_param("os", platform.system())
        mlflow.log_param("os_release", platform.release())
        
        # Log package versions
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("pandas_version", pd.__version__)
        mlflow.log_param("numpy_version", np.__version__)
        mlflow.log_param("mlflow_version", mlflow.__version__)
        
        # Log command line arguments
        mlflow.log_param("cmd_args", str(sys.argv))
        
        print(f"Environment info logged with run_id: {run.info.run_id}")
        return run.info.run_id

def train_with_seed(seed=42, n_estimators=100):
    """Train a model with a specific random seed for reproducibility.
    
    Args:
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        n_estimators (int, optional): Number of estimators for the RandomForest. Defaults to 100.
        
    Returns:
        tuple: (run_id, accuracy) - The run ID and test accuracy of the model
    """
    # Set random seeds
    np.random.seed(seed)
    
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        random_state=seed
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    # Train model
    with mlflow.start_run(run_name=f"reproducible_run_seed_{seed}") as run:
        # Log seed and parameters
        mlflow.log_param("random_seed", seed)
        mlflow.log_param("n_estimators", n_estimators)
        
        # Train model with fixed random state
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=seed
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Log dataset snapshot
        dataset_info = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "test_size": 0.2,
            "class_distribution": np.bincount(y).tolist()
        }
        mlflow.log_dict(dataset_info, "dataset_info.json")
        
        print(f"Model trained with seed {seed}, accuracy: {accuracy:.4f}, run_id: {run.info.run_id}")
        return run.info.run_id, accuracy

def verify_reproducibility():
    """Verify that training with the same seed produces the same results.
    
    This function trains models with the same and different seeds to demonstrate
    that using the same seed produces consistent results while different seeds
    produce different results.
    
    Returns:
        tuple: (run_ids, accuracies) - Lists of run IDs and their corresponding accuracies
    """
    print("\nVerifying reproducibility...")
    
    # Train model twice with the same seed
    run_id1, accuracy1 = train_with_seed(seed=42)
    run_id2, accuracy2 = train_with_seed(seed=42)
    
    # Train model with a different seed
    run_id3, accuracy3 = train_with_seed(seed=123)
    
    # Compare results
    print("\nReproducibility Results:")
    print(f"Run 1 (seed=42): Accuracy = {accuracy1:.6f}")
    print(f"Run 2 (seed=42): Accuracy = {accuracy2:.6f}")
    print(f"Run 3 (seed=123): Accuracy = {accuracy3:.6f}")
    
    # Check if runs with same seed have identical accuracy
    is_reproducible = abs(accuracy1 - accuracy2) < 1e-10
    print(f"\nRuns with same seed have identical results: {is_reproducible}")
    
    # Check difference between different seeds
    seed_difference = abs(accuracy1 - accuracy3)
    print(f"Difference between different seeds: {seed_difference:.6f}")
    
    return is_reproducible

if __name__ == "__main__":
    # Log environment information
    log_environment_info()
    
    # Verify reproducibility
    is_reproducible = verify_reproducibility()
    
    if is_reproducible:
        print("\nSuccess! The model training is reproducible.")
    else:
        print("\nWarning! The model training is not perfectly reproducible.")
        print("This could be due to non-deterministic operations in the libraries.")
        print("Consider using deterministic algorithms or fixed seeds for all operations.")