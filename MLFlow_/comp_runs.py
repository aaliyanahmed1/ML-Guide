"""MLflow Run Comparison Example.

This module provides utilities for comparing different MLflow runs.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI (optional)
# mlflow.set_tracking_uri("sqlite:///mlflow.db")

def get_experiment_runs(experiment_name):
    """Get all runs for a specific experiment.
    
    Args:
        experiment_name (str): Name of the experiment
        
    Returns:
        list: List of experiment runs
    """
    # Get experiment by name
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found")
        return []
    
    # Get all runs for the experiment
    client = MlflowClient()
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    return runs

def compare_runs_table(runs, param_keys=None, metric_keys=None):
    """Compare runs in a tabular format.
    
    Args:
        runs (list): List of runs to compare
        param_keys (list, optional): Parameters to include. Defaults to None (all parameters).
        metric_keys (list, optional): Metrics to include. Defaults to None (all metrics).
        
    Returns:
        DataFrame: Table comparing the runs
    """
    if not runs:
        print("No runs to compare")
        return None
    
    # Extract run data
    run_data = []
    
    for run in runs:
        run_info = run.info
        params = run.data.params
        metrics = run.data.metrics
        
        # Create a dictionary for this run
        run_dict = {
            "run_id": run_info.run_id,
            "start_time": pd.to_datetime(run_info.start_time, unit="ms"),
            "status": run_info.status,
        }
        
        # Add parameters
        if param_keys:
            for key in param_keys:
                run_dict[f"param.{key}"] = params.get(key, None)
        else:
            for key, value in params.items():
                run_dict[f"param.{key}"] = value
        
        # Add metrics
        if metric_keys:
            for key in metric_keys:
                run_dict[f"metric.{key}"] = metrics.get(key, None)
        else:
            for key, value in metrics.items():
                run_dict[f"metric.{key}"] = value
        
        run_data.append(run_dict)
    
    # Create DataFrame
    runs_df = pd.DataFrame(run_data)
    
    return runs_df

def plot_metric_comparison(runs_df, metric_name):
    """Plot a comparison of a specific metric across runs.
    
    Args:
        runs_df (DataFrame): DataFrame containing run information
        metric_name (str): Name of the metric to plot
    """
    if f"metric.{metric_name}" not in runs_df.columns:
        print(f"Metric '{metric_name}' not found in runs")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort by metric value
    sorted_df = runs_df.sort_values(f"metric.{metric_name}", ascending=False)
    
    # Create bar chart
    plt.bar(range(len(sorted_df)), sorted_df[f"metric.{metric_name}"])
    plt.xticks(range(len(sorted_df)), sorted_df["run_id"].str[:8], rotation=45)
    plt.xlabel("Run ID")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name} across runs")
    plt.tight_layout()
    
    # Save the plot as an artifact
    plt.savefig(f"{metric_name}_comparison.png")
    print(f"Plot saved as {metric_name}_comparison.png")
    
    plt.show()

def find_best_run(runs_df, metric_name, higher_is_better=True):
    """Find the best run based on a specific metric.
    
    Args:
        runs_df (DataFrame): DataFrame containing run information
        metric_name (str): Name of the metric to use for comparison
        higher_is_better (bool, optional): Whether higher metric values are better. Defaults to True.
        
    Returns:
        Series: The best run
    """
    if f"metric.{metric_name}" not in runs_df.columns:
        print(f"Metric '{metric_name}' not found in runs")
        return None
    
    if higher_is_better:
        best_idx = runs_df[f"metric.{metric_name}"].idxmax()
    else:
        best_idx = runs_df[f"metric.{metric_name}"].idxmin()
    
    best_run = runs_df.loc[best_idx]
    
    return best_run

if __name__ == "__main__":
    # Example usage
    experiment_name = "RandomForest_Versions"
    
    # Get runs for the experiment
    runs = get_experiment_runs(experiment_name)
    
    # Compare runs in a table
    runs_df = compare_runs_table(
        runs,
        param_keys=["n_estimators", "max_depth"],
        metric_keys=["accuracy"]
    )
    
    if runs_df is not None:
        print("\nRuns Comparison:")
        print(runs_df[["run_id", "param.n_estimators", "param.max_depth", "metric.accuracy"]])
        
        # Plot metric comparison
        plot_metric_comparison(runs_df, "accuracy")
        
        # Find best run
        best_run = find_best_run(runs_df, "accuracy")
        if best_run is not None:
            print("\nBest Run:")
            print(f"Run ID: {best_run['run_id']}")
            print(f"n_estimators: {best_run['param.n_estimators']}")
            print(f"max_depth: {best_run['param.max_depth']}")
            print(f"Accuracy: {best_run['metric.accuracy']:.4f}")