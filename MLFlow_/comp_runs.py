"""MLflow Run Comparison Example for RF-DETR Object Detection Models.

This module provides utilities for comparing different MLflow runs of RF-DETR
object detection models, focusing on metrics like model size, inference time,
and detection performance across different model variants (nano, small, base, large).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

# Import Roboflow Inference SDK for RF-DETR models
try:
    from inference import get_model
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    print("Roboflow Inference SDK not found. Install with: pip install inference")

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
    """Compare RF-DETR model runs in a tabular format.
    
    Args:
        runs (list): List of runs to compare
        param_keys (list, optional): Parameters to include. Defaults to None (all parameters).
            For RF-DETR models, useful parameters include: model_variant, model_id, model_size_mb, confidence_threshold
        metric_keys (list, optional): Metrics to include. Defaults to None (all metrics).
            For RF-DETR models, useful metrics include: load_time_ms, inference_time_ms, num_detections, avg_confidence
        
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

def plot_metric_comparison(runs_df, metric_name, lower_is_better=False):
    """Plot a comparison of a specific metric across RF-DETR model variants.
    
    Args:
        runs_df (DataFrame): DataFrame containing run information
        metric_name (str): Name of the metric to plot
        lower_is_better (bool, optional): Whether lower values are better for this metric. 
            Defaults to False. Set to True for metrics like inference_time_ms.
    """
    if f"metric.{metric_name}" not in runs_df.columns:
        print(f"Metric '{metric_name}' not found in runs")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Sort by metric value
    ascending = lower_is_better
    sorted_df = runs_df.sort_values(f"metric.{metric_name}", ascending=ascending)
    
    # Get model variants if available
    if "param.model_variant" in sorted_df.columns:
        labels = sorted_df["param.model_variant"]
    else:
        labels = sorted_df["run_id"].str[:8]
    
    # Create bar chart
    bars = plt.bar(range(len(sorted_df)), sorted_df[f"metric.{metric_name}"])
    plt.xticks(range(len(sorted_df)), labels, rotation=45)
    plt.xlabel("RF-DETR Model Variant")
    plt.ylabel(metric_name)
    plt.title(f"Comparison of {metric_name} across RF-DETR variants")
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    # Save the plot as an artifact
    plt.savefig(f"{metric_name}_comparison.png")
    print(f"Plot saved as {metric_name}_comparison.png")
    
    plt.show()


def plot_model_variant_comparison(runs_df):
    """Create a comprehensive comparison of RF-DETR model variants.
    
    Args:
        runs_df (DataFrame): DataFrame containing run information for different RF-DETR variants
    """
    if "param.model_variant" not in runs_df.columns:
        print("Model variant information not found in runs")
        return
    
    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RF-DETR Model Variant Comparison", fontsize=16)
    
    # Sort by model size (nano, small, base, large)
    variant_order = {"nano": 0, "small": 1, "base": 2, "large": 3}
    if "param.model_variant" in runs_df.columns:
        runs_df["variant_order"] = runs_df["param.model_variant"].map(variant_order)
        sorted_df = runs_df.sort_values("variant_order")
        labels = sorted_df["param.model_variant"]
    else:
        sorted_df = runs_df
        labels = sorted_df["run_id"].str[:8]
    
    # Plot 1: Model Size
    if "param.model_size_mb" in sorted_df.columns:
        ax = axs[0, 0]
        bars = ax.bar(labels, sorted_df["param.model_size_mb"])
        ax.set_title("Model Size (MB)")
        ax.set_ylabel("Size (MB)")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
    
    # Plot 2: Load Time
    if "metric.load_time_ms" in sorted_df.columns:
        ax = axs[0, 1]
        bars = ax.bar(labels, sorted_df["metric.load_time_ms"])
        ax.set_title("Model Load Time (ms)")
        ax.set_ylabel("Time (ms)")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
    
    # Plot 3: Inference Time
    if "metric.inference_time_ms" in sorted_df.columns:
        ax = axs[1, 0]
        bars = ax.bar(labels, sorted_df["metric.inference_time_ms"])
        ax.set_title("Inference Time (ms)")
        ax.set_ylabel("Time (ms)")
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom')
    
    # Plot 4: Average Confidence or Number of Detections
    metric_to_plot = "metric.avg_confidence" if "metric.avg_confidence" in sorted_df.columns else "metric.num_detections"
    if metric_to_plot in sorted_df.columns:
        ax = axs[1, 1]
        bars = ax.bar(labels, sorted_df[metric_to_plot])
        title = "Average Confidence" if metric_to_plot == "metric.avg_confidence" else "Number of Detections"
        ax.set_title(title)
        ax.set_ylabel(title)
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}' if metric_to_plot == "metric.avg_confidence" else f'{height:.0f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot as an artifact
    plt.savefig("rfdetr_variant_comparison.png")
    print("Plot saved as rfdetr_variant_comparison.png")
    
    plt.show()

def find_best_run(runs_df, metric_name, higher_is_better=True):
    """Find the best RF-DETR model run based on a specific metric.
    
    Args:
        runs_df (DataFrame): DataFrame containing run information
        metric_name (str): Name of the metric to use for comparison
            For RF-DETR models, useful metrics include:
            - inference_time_ms (lower is better)
            - avg_confidence (higher is better)
            - num_detections (depends on use case)
        higher_is_better (bool, optional): Whether higher metric values are better. 
            Defaults to True. Set to False for metrics like inference_time_ms.
        
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

# Try to import utils module for enhanced output
try:
    from utils import print_section, print_subsection, print_env_info, print_run_info
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    # Define minimal versions of utility functions
    def print_section(title):
        """Print a section title."""
        print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")
    
    def print_subsection(title):
        """Print a subsection title."""
        print(f"\n{'-' * 40}\n{title}\n{'-' * 40}")
    
    def print_env_info():
        """Print basic environment information."""
        print("Environment Information:")
        print(f"- Python: {os.sys.version.split()[0]}")
        print(f"- MLflow: {mlflow.__version__}")
        print(f"- Working Directory: {os.getcwd()}")
    
    def print_run_info(run):
        """Print detailed information about a run."""
        print(f"Run ID: {run['run_id']}")
        print("Parameters:")
        for key, value in run.items():
            if key.startswith("param."):
                print(f"  {key[6:]}: {value}")
        print("Metrics:")
        for key, value in run.items():
            if key.startswith("metric."):
                print(f"  {key[7:]}: {value}")


if __name__ == "__main__":
    # Print environment information
    if UTILS_AVAILABLE:
        print_section("MLflow Run Comparison Example for RF-DETR Models")
        print_env_info()
    else:
        print_section("MLflow Run Comparison Example for RF-DETR Models")
        print_env_info()
    
    # Check if Roboflow Inference SDK is available
    if not RFDETR_AVAILABLE:
        print("Error: Roboflow Inference SDK not available. Install with: pip install inference")
        print("Exiting example...")
        exit(1)
    
    # Example usage
    print_subsection("Retrieving Experiment Runs")
    experiment_name = "RF-DETR-ObjectDetection"
    print(f"Fetching runs for experiment: {experiment_name}")
    
    # Get runs for the experiment
    runs = get_experiment_runs(experiment_name)
    
    # Compare runs in a table
    print_subsection("RF-DETR Model Variant Comparison")
    runs_df = compare_runs_table(
        runs,
        param_keys=["model_variant", "model_id", "model_size_mb", "confidence_threshold"],
        metric_keys=["load_time_ms", "inference_time_ms", "num_detections", "avg_confidence"]
    )
    
    if runs_df is not None and not runs_df.empty:
        print(f"Found {len(runs_df)} RF-DETR model variants in experiment {experiment_name}")
        print("\nRF-DETR Model Comparison:")
        display_columns = ["run_id", "param.model_variant", "param.model_size_mb", 
                          "metric.load_time_ms", "metric.inference_time_ms"]
        # Add optional metrics if they exist
        if "metric.num_detections" in runs_df.columns:
            display_columns.append("metric.num_detections")
        if "metric.avg_confidence" in runs_df.columns:
            display_columns.append("metric.avg_confidence")
        
        print(runs_df[display_columns])
        
        # Plot comprehensive model variant comparison
        print_subsection("RF-DETR Model Variant Visualization")
        print("Generating comprehensive model comparison visualization...")
        plot_model_variant_comparison(runs_df)
        
        # Plot individual metric comparisons
        print_subsection("Inference Time Comparison")
        print("Generating inference time comparison plot...")
        plot_metric_comparison(runs_df, "inference_time_ms", lower_is_better=True)
        
        if "metric.avg_confidence" in runs_df.columns:
            print_subsection("Average Confidence Comparison")
            print("Generating average confidence comparison plot...")
            plot_metric_comparison(runs_df, "avg_confidence")
        
        # Find best run for inference speed
        print_subsection("Finding Fastest Model")
        fastest_run = find_best_run(runs_df, "inference_time_ms", higher_is_better=False)
        if fastest_run is not None:
            print("\nFastest RF-DETR Model:")
            print(f"Model Variant: {fastest_run.get('param.model_variant', 'Unknown')}")
            print(f"Model Size: {fastest_run.get('param.model_size_mb', 'Unknown')} MB")
            print(f"Inference Time: {fastest_run.get('metric.inference_time_ms', 'Unknown'):.2f} ms")
            
            # Print detailed information about best run
            print_run_info(fastest_run)
        
        # Find best run for confidence (if available)
        if "metric.avg_confidence" in runs_df.columns:
            print_subsection("Finding Most Confident Model")
            most_confident_run = find_best_run(runs_df, "avg_confidence")
            if most_confident_run is not None:
                print("\nMost Confident RF-DETR Model:")
                print(f"Model Variant: {most_confident_run.get('param.model_variant', 'Unknown')}")
                print(f"Model Size: {most_confident_run.get('param.model_size_mb', 'Unknown')} MB")
                print(f"Average Confidence: {most_confident_run.get('metric.avg_confidence', 'Unknown'):.4f}")
                
                # Print detailed information about most confident run
                print_run_info(most_confident_run)
    else:
        print(f"No runs found for experiment {experiment_name}")
        print("Please run RF-DETR model experiments first or check the experiment name")
        print("You can use the model_registry.py or mlflow_eg.py scripts to register RF-DETR models")
    
    print_section("Run Comparison Example Complete")
    print("\nView the RF-DETR model comparisons in the MLflow UI:")
    print("  mlflow ui")
    print("\nOr use the MLflow API to query the Model Registry programmatically.")
    print("\nFor more information, visit: https://www.mlflow.org/docs/latest/model-registry.html")