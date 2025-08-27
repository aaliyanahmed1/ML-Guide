"""Utility functions for MLflow examples.

This module provides helper functions for enhancing output and visualization in MLflow examples.
"""

import os
import sys
import platform
import mlflow
import pandas as pd
import matplotlib.pyplot as plt


def print_section(title):
    """Print a formatted section header.
    
    Args:
        title (str): The section title to print
    """
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header.
    
    Args:
        title (str): The subsection title to print
    """
    print("\n" + "-" * 60)
    print(f" {title} ")
    print("-" * 60)


def print_environment_info():
    """Print information about the current environment."""
    print_subsection("Environment Information")
    print(f"Python version: {platform.python_version()}")
    print(f"MLflow version: {mlflow.__version__}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


def print_run_info(run):
    """Print detailed information about an MLflow run.
    
    Args:
        run: MLflow run object
    """
    print_subsection("Run Information")
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    print(f"Status: {run.info.status}")
    
    print("\nParameters:")
    for key, value in run.data.params.items():
        print(f"  {key}: {value}")
    
    print("\nMetrics:")
    for key, value in run.data.metrics.items():
        print(f"  {key}: {value}")
    
    print("\nArtifacts:")
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run.info.run_id)
    for artifact in artifacts:
        print(f"  {artifact.path} ({artifact.file_size} bytes)")


def visualize_metrics(metrics_dict, title="Metrics Comparison"):
    """Visualize metrics in a bar chart.
    
    Args:
        metrics_dict (dict): Dictionary of metric names and values
        title (str, optional): Title for the chart. Defaults to "Metrics Comparison".
    """
    plt.figure(figsize=(10, 6))
    plt.bar(metrics_dict.keys(), metrics_dict.values())
    plt.title(title)
    plt.ylabel("Value")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def display_model_info(model_name, version=None):
    """Display detailed information about a registered model.
    
    Args:
        model_name (str): Name of the registered model
        version (str, optional): Specific version to display. Defaults to None.
    """
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Get model details
        if version:
            model_version = client.get_model_version(model_name, version)
            print_subsection(f"Model: {model_name} (Version {version})")
            print(f"Status: {model_version.status}")
            print(f"Stage: {model_version.current_stage}")
            print(f"Description: {model_version.description}")
            print(f"Run ID: {model_version.run_id}")
        else:
            # List all versions
            versions = client.search_model_versions(f"name='{model_name}'")
            print_subsection(f"Model: {model_name} (All Versions)")
            
            if not versions:
                print("No versions found for this model.")
                return
                
            data = []
            for v in versions:
                data.append({
                    "Version": v.version,
                    "Stage": v.current_stage,
                    "Status": v.status,
                    "Run ID": v.run_id
                })
            
            # Display as table
            df = pd.DataFrame(data)
            print(df.to_string(index=False))
    except Exception as e:
        print(f"Error retrieving model information: {str(e)}")