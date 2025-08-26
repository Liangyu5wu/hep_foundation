#!/bin/bash
#SBATCH --job-name=hep_leading_jet_pt_standalone
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32         # Required: NERSC gpu_shared_ss11 queue mandates 32 cores per GPU
#SBATCH --gpus-per-task=1
#SBATCH --time=10:00:00  # Extended time for comprehensive analysis
#SBATCH --output=logs/slurm-leading_jet_pt_standalone-%j.out
#SBATCH --error=logs/slurm-leading_jet_pt_standalone-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=liangyu5@stanford.edu

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Ensure we're in SCRATCH space
cd $SCRATCH/hep_foundation || {
    echo "ERROR: Could not change to $SCRATCH/hep_foundation"
    exit 1
}

# Create logs directory if it doesn't exist
mkdir -p logs

# Load required NERSC modules
echo "Loading NERSC modules..."
module load craype
module load tensorflow/2.12.0

# Set environment variables for optimal GPU performance
export SLURM_CPU_BIND="cores"
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

# Memory management for large datasets
export TF_GPU_MEMORY_GROWTH=1
export TF_FORCE_GPU_ALLOW_GROWTH=true

# Verify GPU access
echo "Checking GPU access..."
nvidia-smi || {
    echo "WARNING: nvidia-smi failed, but continuing..."
}

# Install project dependencies if not already installed
echo "Ensuring project dependencies are installed..."
pip install --user --upgrade pip
pip install --user -e .

# Remove any conflicting tensorflow installations
echo "Cleaning up any pip-installed tensorflow..."
pip uninstall --user -y tensorflow tensorflow-gpu tensorflow-cpu 2>/dev/null || true

# Verify environment
echo "Python version: $(python --version)"
echo "Python location: $(which python)"
echo "Pip user directory: $(python -m site --user-base)"

# Test TensorFlow functionality
echo "Testing TensorFlow functionality..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
print(f'CUDA available: {tf.test.is_built_with_cuda()}')
" || {
    echo "WARNING: TensorFlow test failed, but continuing..."
}

# Copy the config file to experiment stack
echo "Setting up experiment configuration..."
CONFIG_FILE="standalone_utils/leading_jet_pt_standalone_config.yaml"
STACK_DIR="_experiment_config_stack"

# Create stack directory if it doesn't exist
mkdir -p $STACK_DIR

# Copy config to stack (this will be processed and removed by the pipeline)
cp $CONFIG_FILE $STACK_DIR/

echo "Configuration copied to stack: $STACK_DIR/$(basename $CONFIG_FILE)"

# Run the standalone regression pipeline
echo "Starting standalone regression pipeline execution..."
echo "Command: python standalone_utils/run_standalone_regression.py --config-stack"
echo "Time started: $(date)"

# Run with output capture and timestamping
python standalone_utils/run_standalone_regression.py --config-stack 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done

# Capture exit code
EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End time: $(date)"

# Show disk usage of results
if [ -d "_standalone_experiments" ]; then
    echo "Results directory size:"
    du -sh _standalone_experiments
fi

if [ -d "_processed_datasets" ]; then
    echo "Processed datasets size:"
    du -sh _processed_datasets
fi

echo "=========================================="

# Exit with the pipeline's exit code
exit $EXIT_CODE