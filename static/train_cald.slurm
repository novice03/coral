#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=8:00:00
#SBATCH --job-name=coral_calderon
#SBATCH --output=logs/coral_%j.out
#SBATCH --error=logs/coral_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "=========================================="
echo "CORAL Training for Calderón Problem"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
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
echo "Stage 1: Training INRs"
echo "=========================================="

python static_inr_cald.py 

# Capture the INR run name from wandb
# The run name is printed during training, we need to extract it
# For now, we'll use a fixed approach - you may need to adjust this

# Get the most recent run directory
INR_RUN_NAME=$(ls -t calderon/inr/*.pt 2>/dev/null | head -1 | xargs basename | sed 's/.pt$//')

if [ -z "$INR_RUN_NAME" ]; then
    echo "ERROR: No INR checkpoint found!"
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
echo "Stage 2: Training Mapping Network"
echo "=========================================="

python static_regr_cald.py \
    "inr.run_name=$INR_RUN_NAME" 

echo ""
echo "=========================================="
echo "Training complete!"
echo "End time: $(date)"
echo "=========================================="