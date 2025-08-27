"""Experiment tracking using MLflow.

This module demonstrates how to track machine learning experiments using MLflow.
"""

import os
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def prepare_data():
    """Prepare synthetic data for demonstration.
    
    Returns:
        tuple: Train and test data splits (X_train, X_test, y_train, y_test)
    """
    # For demonstration, we'll create synthetic data
    X = np.random.rand(1000, 10)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Simple binary classification
    return train_test_split(X, y, test_size=0.2, random_state=42)

def run_experiment(n_estimators=100, max_depth=None, min_samples_split=2):
    """Run a machine learning experiment with MLflow tracking.
    
    Args:
        n_estimators (int, optional): Number of trees in the forest. Defaults to 100.
        max_depth (int, optional): Maximum depth of the tree. Defaults to None.
        min_samples_split (int, optional): Minimum samples required to split. Defaults to 2.
        
    Returns:
        tuple: Trained model and accuracy score
    """
    # Start MLflow run
    with mlflow.start_run(run_name="rf_classifier"):
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data()
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        print(f"Run completed with accuracy: {accuracy:.4f}")
        return model, accuracy

def run_multiple_experiments(param_variations):
    """Run multiple experiments with different parameter configurations.
    
    Args:
        param_variations (list): List of parameter dictionaries for experiments
        
    Returns:
        list: Results of all experiments with parameters and accuracy
    """
    results = []
    
    for i, params in enumerate(param_variations):
        print(f"Running experiment {i+1}/{len(param_variations)}")
        _, accuracy = run_experiment(**params)
        results.append((params, accuracy))
    
    # Find best parameters
    best_params, best_accuracy = max(results, key=lambda x: x[1])
    print(f"Best parameters: {best_params}, accuracy: {best_accuracy:.4f}")
    return results

if __name__ == "__main__":
    # Example usage
    param_variations = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 300, "max_depth": None},
    ]
    run_multiple_experiments(param_variations)








