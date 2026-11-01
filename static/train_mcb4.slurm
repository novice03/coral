#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=30:00:00
#SBATCH --job-name=coral_mcb
#SBATCH --output=logs/coral_mcb_%j_%a.out
#SBATCH --error=logs/coral_mcb_%j_%a.err
#SBATCH --array=0-4

# Create logs directory if it doesn't exist
mkdir -p logs

# Force unbuffered output for live printing
export PYTHONUNBUFFERED=1

# ==========================================
# Define dataset configurations
# ==========================================
# Format: "name|data_root|train_file|test_file"

BASE_DATA="/fs/nexus-scratch/pulijala"
BASE_SPLITS="/fs/nexus-scratch/pulijala/data_list_paths/mcb_b"

DATASETS=(
    "multishape|${BASE_DATA}/data_poisson_multishape_20260126_125426|${BASE_SPLITS}/fitting/known_shape_known_prob.txt|${BASE_SPLITS}/fitting/unknown_shape_unknown_prob.txt"
    "gear|${BASE_DATA}/data_poisson_gear_20260128_222045|${BASE_SPLITS}/gear/known_shape_known_prob.txt|${BASE_SPLITS}/gear/unknown_shape_unknown_prob.txt"
    "motor|${BASE_DATA}/data_poisson_motor_20260129_123736|${BASE_SPLITS}/motor/known_shape_known_prob.txt|${BASE_SPLITS}/motor/unknown_shape_unknown_prob.txt"
    "nut|${BASE_DATA}/data_poisson_nut_20260130_170014|${BASE_SPLITS}/nut/known_shape_known_prob.txt|${BASE_SPLITS}/nut/unknown_shape_unknown_prob.txt"
    "screwsbolts|${BASE_DATA}/data_poisson_screwsbolts_20260130_171523|${BASE_SPLITS}/screws_and_bolts/known_shape_known_prob.txt|${BASE_SPLITS}/screws_and_bolts/unknown_shape_unknown_prob.txt"
)

# Get current dataset config based on array task ID
DATASET_CONFIG="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# Parse the config string
IFS='|' read -r DATASET_NAME DATA_ROOT TRAIN_FILE TEST_FILE <<< "$DATASET_CONFIG"

# Print job info
echo "=========================================="
echo "CORAL Training for MCB Poisson Problem"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo "Dataset: $DATASET_NAME"
echo "Data root: $DATA_ROOT"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate coral

# Change to working directory
cd /fs/nexus-scratch/pulijala/coral/static

# ==========================================
# Stage 1: Train INRs
# ==========================================
echo ""
echo "=========================================="
echo "Stage 1: Training INRs ($DATASET_NAME)"
echo "=========================================="

python -u static_inr_mcb.py \
    data.data_root="$DATA_ROOT" \
    data.train_split_file="$TRAIN_FILE" \
    data.test_split_file="$TEST_FILE" \
    wandb.name="inr_${DATASET_NAME}"

# Get the most recent run directory for THIS dataset
# Use the wandb name pattern to find the correct checkpoint
INR_RUN_NAME=$(ls -t mcb/inr/inr_${DATASET_NAME}*.pt 2>/dev/null | grep -v "_final" | head -1 | xargs basename | sed 's/.pt$//')

# Fallback: if naming pattern doesn't match, get most recent
if [ -z "$INR_RUN_NAME" ]; then
    INR_RUN_NAME=$(ls -t mcb/inr/*.pt 2>/dev/null | grep -v "_final" | head -1 | xargs basename | sed 's/.pt$//')
fi

if [ -z "$INR_RUN_NAME" ]; then
    echo "ERROR: No INR checkpoint found for $DATASET_NAME!"
    exit 1
fi

echo ""
echo "=========================================="
echo "INR training complete. Run name: $INR_RUN_NAME"
echo "=========================================="

# ==========================================
# Stage 2: Train Mapping Network
# ==========================================
echo ""
echo "=========================================="
echo "Stage 2: Training Mapping Network ($DATASET_NAME)"
echo "=========================================="

python -u static_regr_mcb.py \
    data.data_root="$DATA_ROOT" \
    data.train_split_file="$TRAIN_FILE" \
    data.test_split_file="$TEST_FILE" \
    inr.run_name="$INR_RUN_NAME" \
    wandb.name="regr_${DATASET_NAME}"

echo ""
echo "=========================================="
echo "Training complete for $DATASET_NAME!"
echo "End time: $(date)"
echo "=========================================="