#!/bin/bash
#SBATCH --partition=tron
#SBATCH --account=nexus
#SBATCH --qos=default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=48:00:00
#SBATCH --job-name=coral_mcb_seeds
#SBATCH --output=logs/coral_mcb_seeds_%A_%a.out
#SBATCH --error=logs/coral_mcb_seeds_%A_%a.err
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --array=0-15

# 4x seed replicate of the CORAL MCB runs (train_mcb4.slurm, Jan 31 - Feb 2; seed 42),
# physical median rel-L2: multishape 18.80, gear 13.57, nut 23.25, screwsbolts 20.76.
# Same code, configs (config/mcb/*.yaml), data roots and split files; only data.seed changes.
# data.seed only drives set_seed (torch/numpy/random): the train/test split comes from the
# split files and subsample_interior is unset in config/mcb (all mesh nodes are used), so the
# data is identical for every seed. Stage 2 re-reads the seed from the INR checkpoint.
#
# Array index -> shape = index / 4, seed = index % 4:
#   0-3 multishape, 4-7 gear, 8-11 nut, 12-15 screwsbolts
# (e.g. --array=4-7 submits gear only).
#
# Differences from train_mcb4.slurm that do not touch the training itself:
#   - Run names are fixed per shape/seed and stage 2 is given its INR explicitly, instead of
#     "newest .pt", which would race between concurrent array tasks.
#   - tron instead of scavenger, rtxa6000 (48 GB) instead of rtxa5000: motor OOMed on a 24 GB
#     card. 48h because gear took 29.3h (INR) + 3.7h (regression) in the original.
#   - Requeue: a .done marker is written after stage 1; a requeued job skips a finished
#     stage 1 and otherwise reruns the unfinished stage from scratch (same seed).
# Gear stage 2 uses optim.batch_size=2, as in cont_gear.slurm (job 6227932), which produced
# the reported gear number; the other shapes use the config default (4).

mkdir -p logs
export PYTHONUNBUFFERED=1

BASE_DATA="/fs/nexus-scratch/pulijala"
BASE_SPLITS="/fs/nexus-scratch/pulijala/data_list_paths/mcb_b"
# Format: "name|data_root|split_dir"
DATASETS=(
    "multishape|${BASE_DATA}/data_poisson_multishape_20260126_125426|fitting"
    "gear|${BASE_DATA}/data_poisson_gear_20260128_222045|gear"
    "nut|${BASE_DATA}/data_poisson_nut_20260130_170014|nut"
    "screwsbolts|${BASE_DATA}/data_poisson_screwsbolts_20260130_171523|screws_and_bolts"
)

IFS='|' read -r DATASET_NAME DATA_ROOT SPLIT_DIR <<< "${DATASETS[$((SLURM_ARRAY_TASK_ID / 4))]}"
SEED=$((SLURM_ARRAY_TASK_ID % 4))
TRAIN_FILE="${BASE_SPLITS}/${SPLIT_DIR}/known_shape_known_prob.txt"
TEST_FILE="${BASE_SPLITS}/${SPLIT_DIR}/unknown_shape_unknown_prob.txt"
INR_NAME=inr_${DATASET_NAME}_seed${SEED}
REGR_NAME=regr_${DATASET_NAME}_seed${SEED}

REGR_EXTRA=()
if [ "$DATASET_NAME" = "gear" ]; then
    REGR_EXTRA=("optim.batch_size=2")
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

echo "=========================================="
echo "CORAL MCB seed run  dataset=${DATASET_NAME}  seed=${SEED}"
echo "Array job ${SLURM_ARRAY_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}  (restart count ${SLURM_RESTART_COUNT:-0})"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Data root: $DATA_ROOT"
echo "Train file: $TRAIN_FILE"
echo "Test file: $TEST_FILE"
echo "=========================================="

source ~/miniconda3/bin/activate
conda activate coral

cd /fs/nexus-scratch/pulijala/coral/static

# ==========================================
# Stage 1: Train INRs
# ==========================================
if [ -f "mcb/inr/${INR_NAME}.done" ]; then
    echo "[INFO] Stage 1 already finished for ${INR_NAME}; skipping."
else
    echo ""
    echo "=========================================="
    echo "Stage 1: Training INRs (${INR_NAME})"
    echo "=========================================="
    python -u static_inr_mcb.py \
        data.data_root="$DATA_ROOT" \
        data.train_split_file="$TRAIN_FILE" \
        data.test_split_file="$TEST_FILE" \
        "data.seed=${SEED}" \
        "wandb.name=${INR_NAME}"
    rc=$?
    if [ $rc -ne 0 ] || [ ! -f "mcb/inr/${INR_NAME}.pt" ]; then
        echo "ERROR: stage 1 failed (rc=${rc}) or mcb/inr/${INR_NAME}.pt missing" >&2
        exit 1
    fi
    touch "mcb/inr/${INR_NAME}.done"
fi

# ==========================================
# Stage 2: Train Mapping Network
# ==========================================
echo ""
echo "=========================================="
echo "Stage 2: Training Mapping Network (${REGR_NAME})"
echo "=========================================="
python -u static_regr_mcb.py \
    data.data_root="$DATA_ROOT" \
    data.train_split_file="$TRAIN_FILE" \
    data.test_split_file="$TEST_FILE" \
    "inr.run_name=${INR_NAME}" \
    "data.seed=${SEED}" \
    "wandb.name=${REGR_NAME}" \
    "${REGR_EXTRA[@]}"
rc=$?

echo ""
echo "=========================================="
echo "Training finished (rc=${rc})  metrics: mcb/regression/${REGR_NAME}_metrics.txt"
echo "End time: $(date)"
echo "=========================================="
exit $rc
