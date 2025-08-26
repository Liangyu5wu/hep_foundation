#!/bin/bash
#SBATCH --job-name=test_leading_jet_pt_standalone
#SBATCH --account=m2616
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --nodes=1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00  # Short time for testing
#SBATCH --output=logs/slurm-test_standalone-%j.out
#SBATCH --error=logs/slurm-test_standalone-%j.err
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

# Create logs directory
mkdir -p logs

# Load required NERSC modules
echo "Loading NERSC modules..."
module load craype
module load tensorflow/2.12.0

# Set environment variables
export SLURM_CPU_BIND="cores"
export NUMEXPR_MAX_THREADS=128
export CUDA_VISIBLE_DEVICES=0
export TF_NUM_INTEROP_THREADS=8
export TF_NUM_INTRAOP_THREADS=16
export OMP_NUM_THREADS=16

# Install dependencies
echo "Installing dependencies..."
pip install --user --upgrade pip
pip install --user -e .

# Clean up conflicting tensorflow
pip uninstall --user -y tensorflow tensorflow-gpu tensorflow-cpu 2>/dev/null || true

# Test TensorFlow
echo "Testing TensorFlow..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')
"

# Run the test directly with config file
echo "Running test standalone regression..."
echo "Time started: $(date)"

python standalone_utils/run_standalone_regression.py \
    --config standalone_utils/test_leading_jet_pt_standalone.yaml \
    --experiments-dir _test_standalone_experiments \
    --verbose 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done

EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
echo "Test completed with exit code: $EXIT_CODE"
echo "End time: $(date)"

# Show results
if [ -d "_test_standalone_experiments" ]; then
    echo "Test results:"
    ls -la _test_standalone_experiments/
    du -sh _test_standalone_experiments
fi

echo "=========================================="

exit $EXIT_CODE