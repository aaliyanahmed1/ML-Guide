"""MLflow Model Registry Example.

This module demonstrates how to use the MLflow Model Registry for model management.
"""

import os
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def train_and_register_model(model_name, version_description=None):
    """Train a model and register it to the MLflow Model Registry.
    
    Args:
        model_name (str): Name to register the model under
        version_description (str, optional): Description for the model version. Defaults to None.
        
    Returns:
        RegisteredModel: The registered model object
    """
    # Create a simple model
    X = np.random.rand(100, 5)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    model = RandomForestClassifier(n_estimators=10)
    model.fit(X, y)
    
    # Log the model
    with mlflow.start_run() as run:
        mlflow.log_param("n_estimators", 10)
        mlflow.log_metric("accuracy", model.score(X, y))
        
        # Log the model to MLflow
        mlflow.sklearn.log_model(model, "model")
        model_uri = f"runs:/{run.info.run_id}/model"
        
        # Register the model
        registered_model = mlflow.register_model(model_uri, model_name)
        print(f"Model registered: {registered_model.name} version {registered_model.version}")
        
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
    """Transition a model version to a different stage.
    
    Args:
        model_name (str): Name of the registered model
        version (str): Version of the model to transition
        stage (str): Target stage (None, Staging, Production, Archived)
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    print(f"Model {model_name} version {version} transitioned to {stage}")

def load_production_model(model_name):
    """Load the production version of a registered model.
    
    Args:
        model_name (str): Name of the registered model
        
    Returns:
        mlflow.pyfunc.PyFuncModel: The loaded production model
    """
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    return model

if __name__ == "__main__":
    # Example usage
    model_name = "RandomForestClassifier"
    
    # Register a new model version
    registered_model = train_and_register_model(
        model_name, 
        "Initial model version with basic parameters"
    )
    
    # Transition to staging
    transition_model_stage(model_name, registered_model.version, "Staging")
    
    # After validation, transition to production
    # transition_model_stage(model_name, registered_model.version, "Production")
    
    # Load the production model (would work after transitioning to production)
    # production_model = load_production_model(model_name)