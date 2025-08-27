"""MLflow Examples Runner.

This script runs all MLflow examples in the directory with thorough output.
"""

import os
import sys
import time
import importlib.util
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80)


def print_subheader(title):
    """Print a formatted subheader."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80)


def run_module(module_path):
    """Run a Python module and capture its output.
    
    Args:
        module_path (str): Path to the Python module
    """
    module_name = Path(module_path).stem
    print_header(f"Running {module_name}")
    
    # Print module description
    try:
        with open(module_path, 'r') as f:
            content = f.read()
            docstring = content.split('"""')[1] if '"""' in content else ""
            if docstring:
                print(f"\nDescription: {docstring.strip()}\n")
    except Exception:
        pass
    
    print("Output:")
    print("-" * 40)
    
    # Load the module
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        # Execute the module
        spec.loader.exec_module(module)
        print("-" * 40)
        print(f"\n✅ {module_name} completed successfully")
    except Exception as e:
        print("-" * 40)
        print(f"\n❌ Error running {module_name}: {str(e)}")
    
    print("\n" + "-" * 80)
    time.sleep(2)  # Small pause between modules


def check_mlflow_installation():
    """Check if MLflow is properly installed and configured."""
    try:
        import mlflow
        print(f"✅ MLflow version {mlflow.__version__} is installed")
        
        # Check if MLflow tracking server is accessible
        try:
            mlflow.tracking.get_tracking_uri()
            print("✅ MLflow tracking URI is configured")
        except Exception as e:
            print(f"⚠️ MLflow tracking URI issue: {str(e)}")
            print("   Using default local tracking")
        
        return True
    except ImportError:
        print("❌ MLflow is not installed. Please install it using:")
        print("    pip install -r requirements.txt")
        return False

def main():
    """Run all MLflow examples."""
    print_header("MLflow Examples Runner")
    print("This script will run all MLflow examples in sequence.")
    print("Each example demonstrates different MLflow capabilities.")
    
    # Check MLflow installation
    print_subheader("Checking Environment")
    if not check_mlflow_installation():
        return
    
    print("\nPress Ctrl+C at any time to stop execution.")
    
    # Get the directory of this script
    base_dir = Path(__file__).parent
    
    # Define the order of execution
    modules = [
        base_dir / "Exp_tracking.py",
        base_dir / "model_registry.py",
        base_dir / "model_versioning.py",
        base_dir / "comp_runs.py",
        base_dir / "deployment_.py",
        base_dir / "Reproducibility_" / "Reproducibility__.py",
        base_dir / "mlflow_eg.py"
    ]
    
    # Run each module
    for module_path in modules:
        if module_path.exists():
            run_module(str(module_path))
        else:
            print(f"\n⚠️ Module {module_path} not found, skipping...")
    
    print_header("All MLflow Examples Completed")
    print("\nTo view the MLflow UI, run:")
    print("    mlflow ui")
    print("\nThen open http://localhost:5000 in your browser")
    print("\nMLflow UI shows all experiments, runs, parameters, metrics, and artifacts")
    print("You can compare runs, view model details, and manage model versions")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution stopped by user.")
        sys.exit(0)