#!/bin/bash
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --time=00:20:00
#SBATCH --job-name=coral_mcb_pred
#SBATCH --output=logs/coral_mcb_pred_%j.out
#SBATCH --error=logs/coral_mcb_pred_%j.err

mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"

source ~/miniconda3/bin/activate
conda activate coral

cd /fs/nexus-scratch/pulijala/coral/static

for DATASET in gear multishape nut screwsbolts; do
    echo ""
    echo "=========================================="
    echo "Running inference for: $DATASET"
    echo "=========================================="
    python save_mcb_preds.py \
        --dataset "$DATASET" \
        --output_dir ~/model_predictions
done

echo "Done: $(date)"
