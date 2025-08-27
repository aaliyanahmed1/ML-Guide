# MLflow Guide



## Contents

- **Exp_tracking.py**: Demonstrates experiment tracking with MLflow, including logging parameters, metrics, and models. Shows how to run multiple experiments and compare results.

- **model_registry.py**: Shows how to register models in the MLflow Model Registry, transition models between stages (None, Staging, Production, Archived), and load production models.

- **model_versioning.py**: Illustrates model versioning in MLflow, including creating different versions of a model, comparing versions, and selecting the best version.

- **comp_runs.py**: Provides utilities for comparing different MLflow runs, including tabular comparisons, metric visualizations, and finding the best run.

- **deployment_.py**: Demonstrates how to deploy MLflow models, including local deployment with Flask and exporting models to different formats (ONNX, TensorFlow).

- **Reproducibility_/Reproducibility__.py**: Shows how to ensure reproducibility in machine learning experiments using MLflow, including logging environment information and using fixed random seeds.

## Getting Started

### Prerequisites

Install all required dependencies using the provided requirements.txt file:

```bash
pip install -r requirements.txt
```

### Running Examples

You can run all examples at once with thorough output using the main.py script:

```bash
python main.py
```

Or run each Python file independently:

```bash
python Exp_tracking.py
python model_registry.py
python model_versioning.py
python comp_runs.py
python deployment_.py
python Reproducibility_/Reproducibility__.py
python mlflow_eg.py
```

### Viewing MLflow UI

After running the examples, you can view the MLflow UI by running:

```bash
mlflow ui
```

Then open your browser to http://localhost:5000

## Key MLflow Concepts

1. **Tracking**: Record and query experiments: code, data, config, and results
2. **Projects**: Package data science code in a format to reproduce runs on any platform
3. **Models**: Deploy machine learning models in diverse serving environments
4. **Model Registry**: Store, annotate, discover, and manage models in a central repository
5. **Model Serving**: Host MLflow Models as REST endpoints

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow GitHub Repository](https://github.com/mlflow/mlflow)
- [MLflow Tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html)