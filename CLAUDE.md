# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Environment Setup

This project uses `uv` for Python environment management with a `pyproject.toml` configuration:

```bash
# Initial setup
uv venv --python 3.9
source .venv/bin/activate
uv pip install -e .
uv sync --group dev
pre-commit install

# Testing (must activate venv first)
source .venv/bin/activate && python run_pytest.py

# Code quality
uv run ruff check .
uv run ruff format .
```

**Critical**: Always use `source .venv/bin/activate && python run_pytest.py` for testing. Never create custom tests - this wrapper handles all testing needs and outputs logs to `_test_results/pytest.log`.

## Key Development Commands

- **Install dependencies**: `uv add [--dev] <package>` (prefer over direct `pyproject.toml` editing)
- **Test pipeline**: `source .venv/bin/activate && python run_pytest.py`
- **Run experiments**: `python scripts/run_pipelines.py`
- **Download data**: `python scripts/download_catalogs.py`
- **Format code**: `uv run ruff format .`
- **Check linting**: `uv run ruff check .`
- **Visual interface**: `python launch_gradio.py`
- **Standalone regression**: `python standalone_utils/run_standalone_regression.py --config config.yaml`

## Architecture Overview

### Core Pipeline Flow
1. **Configuration**: YAML configs in `_experiment_config_stack/` → processed by `scripts/run_pipelines.py`
2. **Data Processing**: ATLAS PHYSLITE ROOT files → HDF5 datasets in `_processed_datasets/`
3. **Model Training**: Foundation models (VAE/Autoencoder) + task-specific models (DNN)
4. **Evaluation**: Regression, classification, and anomaly detection comparisons
5. **Results**: Complete experiment folders in `_foundation_experiments/`

### Key Architecture Components

**Foundation Model Pipeline** (`src/hep_foundation/pipeline/foundation_model_pipeline.py`):
- Central orchestrator that runs the complete sequence: foundation training → regression → classification → anomaly detection
- Manages experiment folder creation, logging, and result organization

**PhysLite Data System** (`src/hep_foundation/data/`):
- Processes ATLAS particle physics ROOT files from CERN OpenData
- Features dynamic branch loading, derived feature calculation, and track aggregation
- Uses homemade indexes (`atlas_index.json`, `physlite_branch_index.json`) for file/branch discovery
- Handles event filtering and track-level filtering with configurable aggregators

**Model Factory System** (`src/hep_foundation/models/`):
- Dynamic model creation with `ModelFactory` supporting VAE, Autoencoder, and DNN architectures
- Model-specific classes contain training logic and visualization utilities
- `ModelRegistry` manages experiment metadata and folder organization

**Configuration Management** (`src/hep_foundation/config/`):
- Type-safe config loading with validation for dataset, training, task, and evaluation settings
- Supports complex aggregator configurations for multi-track/object physics data

### Data Processing Details

**Event Processing Flow**:
1. Downloads ROOT files from CERN OpenData to `atlas_data/`
2. `physlite_catalog_processor.py` reads branches and calculates derived features (eta from theta, pt from momentum, etc.)
3. Applies event-level filters and track-level filters from config
4. Creates aggregated arrays: sorts tracks, pads/truncates to fixed lengths, stacks features
5. Saves as HDF5 datasets in `_processed_datasets/` for reuse

**Key Physics Features**:
- Supports any PhysLite branch names in config (InDetTrackParticles, AnalysisElectrons, etc.)
- Automatic derived feature calculation defined in `physlite_derived_features.py`
- Separates background and signal datasets using signal keys from `atlas_index.json`
- HistogramManager creates overlaid background/signal distributions

## Testing and Quality

- **Comprehensive test**: `tests/test_pipeline.py` runs a complete mini-pipeline in ~60 seconds
- **Pre-commit hooks**: Automated formatting (ruff), linting, spell-check, and dead code detection (vulture)
- **Test outputs**: Results saved to `_test_results/` (cleaned each run) for verification
- **Logging**: Centralized configuration in `config/logging_config.py`, use `logger.templog()` for debugging

## NERSC/HPC Usage

```bash
# Download data on login nodes first (I/O bottleneck)
python scripts/download_catalogs.py

# Submit cluster job
sbatch jobs/submit_pipeline_simple.sh
```

## Common Patterns

- **Config creation**: Copy `tests/_test_pipeline_config.yaml` as template, modify values for experiments
- **Dataset reuse**: Pipeline automatically reuses matching datasets from `_processed_datasets/`
- **Experiment organization**: Each config produces a complete results folder with training plots, evaluation metrics, and reproducible config copies
- **Model comparison**: Standard pattern compares fine-tuned vs fixed vs from-scratch models across varying data sizes

## Standalone Model Training

The project supports standalone DNN regression training that operates independently from foundation models. This approach trains task-specific models from scratch without relying on pre-trained foundation model embeddings.

### Standalone Model Architecture

**Key Components** (`src/hep_foundation/standalone_models/`):
- **`StandaloneRegressionPipeline`**: Main orchestrator for standalone training and evaluation
- **`StandaloneDNNRegressor`**: Deep neural network architecture for regression tasks
- **`StandaloneTrainer`**: Handles model training, validation, and early stopping
- **`StandalonePlotManager`**: Comprehensive visualization suite with advanced plots
- **`run_standalone_regression.py`**: Command-line interface for executing standalone experiments

**Architecture Flow**:
1. **Data Processing**: Uses same PhysLite data system as foundation pipeline
2. **Model Training**: Direct from-scratch training on aggregated physics features
3. **Evaluation**: K-fold cross-validation with data size efficiency studies
4. **Visualization**: Advanced plots including 2D histograms, relative errors, and data size comparisons

### Core Standalone Training Commands

```bash
# Run single configuration
python standalone_utils/run_standalone_regression.py --config path/to/config.yaml

# Process all configs in stack (batch mode)
python standalone_utils/run_standalone_regression.py --config-stack

# Create example configuration
python standalone_utils/run_standalone_regression.py --create-example example_config.yaml

# Run with custom directories and verbose logging
python standalone_utils/run_standalone_regression.py \
  --config my_config.yaml \
  --experiments-dir custom_experiments \
  --datasets-dir custom_datasets \
  --verbose

# NERSC cluster submission
sbatch standalone_utils/submit_leading_jet_pt_standalone.sh
sbatch standalone_utils/submit_test_standalone.sh  # For testing
```

### Standalone Configuration Structure

**Essential Configuration Sections**:
```yaml
name: "experiment_name"
description: "Brief description"

# Dataset configuration - use latest run numbers
dataset:
  run_numbers: !python get_run_numbers()[-5:]  # Latest 5 ATLAS runs
  signal_keys: ["zprime_tt", "wprime_taunu", "DMA"]
  catalog_limit: null  # Use all catalogs
  validation_fraction: 0.15
  test_fraction: 0.15

# Physics task definition
task:
  input_array_aggregators:
    - input_branches: ["InDetTrackParticlesAuxDyn.d0", "InDetTrackParticlesAuxDyn.pt"]
      filter_branches: [{"branch": "InDetTrackParticlesAuxDyn.pt", "min": 1.0}]
      sort_by_branch: {"branch": "InDetTrackParticlesAuxDyn.pt"}
      max_length: 50
  
  label_array_aggregators:  # Leading jet pt regression
    - - input_branches: ["AnalysisJetsAuxDyn.pt"]
        filter_branches: [{"branch": "AnalysisJetsAuxDyn.pt", "min": 25.0}]
        max_length: 1

# Model architecture
models:
  standalone_dnn:
    model_type: "standalone_dnn_regressor"
    architecture:
      hidden_layers: [128, 128, 64, 32]
      activation: "relu"
      output_activation: "linear"

# Training configuration
training:
  standalone_dnn:
    batch_size: 128
    learning_rate: 0.001
    epochs: 100
    early_stopping: {patience: 15, min_delta: 0.0001}

# Data efficiency evaluation
evaluation:
  regression_data_sizes: [1000, 10000, 50000, 100000, 500000]
  fixed_epochs: 50
  use_k_fold_cv: true  # Enable k-fold cross-validation
  k_fold: 3  # Number of folds
  create_detailed_plots: true
```

### Advanced Visualization Features

**StandalonePlotManager** provides comprehensive plotting capabilities:

- **Training History**: Loss curves, learning rate schedules, early stopping visualization
- **Prediction Analysis**: 
  - 2D density histograms (predictions vs true values)
  - Scatter plots with correlation statistics
  - Residual analysis and error distributions
- **Relative Error Analysis**: 
  - Histograms of `(pred-true)/true * 100%`
  - Full range and zoomed views
  - Statistical summaries and percentiles
- **Data Size Efficiency Studies**:
  - Test loss vs training data size with k-fold error bars
  - R² score progression across data sizes
  - Comprehensive summary tables

### NERSC Cluster Integration

**SLURM Submission Scripts**:
- **`submit_leading_jet_pt_standalone.sh`**: Full training with comprehensive data sizes (10-hour limit)
- **`submit_test_standalone.sh`**: Quick testing with minimal settings (2-hour limit)

**Key NERSC Features**:
- GPU acceleration with TensorFlow 2.12.0
- Proper module loading and environment setup
- Memory management for large datasets
- Timestamped output logging
- Automatic dependency installation

**Usage Pattern**:
```bash
# Copy config to stack directory (processed and removed automatically)
cp my_config.yaml _experiment_config_stack/

# Submit job
sbatch standalone_utils/submit_leading_jet_pt_standalone.sh

# Monitor progress
tail -f logs/slurm-leading_jet_pt_standalone-*.out
```

### Best Practices for Standalone Training

1. **Configuration Management**:
   - Use `!python get_run_numbers()[-5:]` for latest data
   - Test with small `catalog_limit` before full runs
   - Enable k-fold CV for stable statistics

2. **Data Size Studies**:
   - Start with smaller sizes: `[1000, 5000, 10000]`
   - Use `fixed_epochs` for fair comparison
   - Include error bars with `use_k_fold_cv: true`

3. **Model Architecture**:
   - Start with moderate depth: `[128, 64, 32]`
   - Use batch normalization and dropout for regularization
   - Linear output activation for regression tasks

4. **Performance Optimization**:
   - Use appropriate `batch_size` for GPU memory
   - Enable `hdf5_compression` for storage efficiency
   - Set reasonable `shuffle_buffer` size

5. **Result Analysis**:
   - Check training plots for overfitting/underfitting
   - Analyze relative error distributions
   - Compare data efficiency curves across experiments