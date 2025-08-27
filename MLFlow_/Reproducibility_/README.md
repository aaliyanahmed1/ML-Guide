# MLflow Reproducibility

This directory contains examples and utilities for ensuring reproducibility in machine learning experiments using MLflow.

## Contents

- **Reproducibility__.py**: Demonstrates how to ensure reproducibility in machine learning experiments by logging environment information, using fixed random seeds, and verifying that runs with the same seed produce identical results.

## Key Concepts

1. **Environment Logging**: Capturing system and library versions to ensure reproducibility across different environments.

2. **Fixed Random Seeds**: Using consistent random seeds for all randomized operations (data splitting, model initialization, etc.).

3. **Verification**: Testing that runs with the same seed produce identical results, while different seeds produce different results.

## Running the Example

To run the reproducibility example:

```bash
python Reproducibility__.py
```

Or run it from the parent directory using the main script:

```bash
python main.py
```

## Expected Output

The example will:

1. Log environment information (Python version, OS, library versions)
2. Train models with the same and different random seeds
3. Compare the results to verify reproducibility
4. Report whether the training process is reproducible

## Best Practices for Reproducibility

- Always set random seeds for all libraries (numpy, tensorflow, pytorch, etc.)
- Log all hyperparameters and environment information
- Version control your code and data
- Use fixed versions of dependencies
- Document your hardware specifications