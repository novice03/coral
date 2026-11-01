#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --job-name=coral_gear_regr
#SBATCH --output=logs/coral_gear_regr_%j.out
#SBATCH --error=logs/coral_gear_regr_%j.err

mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ~/miniconda3/bin/activate
conda activate coral
cd /fs/nexus-scratch/pulijala/coral/static

echo "=========================================="
echo "CORAL Regression Only - Gear Dataset"
echo "=========================================="
echo "Start time: $(date)"

# Find the existing INR checkpoint
INR_CKPT=$(ls -t mcb/inr/inr_gear*.pt 2>/dev/null | grep -v "_final" | head -1)
if [ -z "$INR_CKPT" ]; then
    echo "ERROR: No INR checkpoint found for gear!"
    exit 1
fi
INR_RUN_NAME=$(basename "$INR_CKPT" .pt)
echo "Using INR checkpoint: $INR_RUN_NAME"

# Stage 2: Train Regression
python -u static_regr_mcb.py \
    data.data_root="/fs/nexus-scratch/pulijala/data_poisson_gear_20260128_222045" \
    data.train_split_file="/fs/nexus-scratch/pulijala/data_list_paths/mcb_b/gear/known_shape_known_prob.txt" \
    data.test_split_file="/fs/nexus-scratch/pulijala/data_list_paths/mcb_b/gear/unknown_shape_unknown_prob.txt" \
    optim.batch_size=2 \
    inr.run_name="$INR_RUN_NAME" \
    wandb.name="regr_gear"

echo "=========================================="
echo "Complete! End time: $(date)"
echo "=========================================="