#!/bin/bash

#SBATCH --job-name=date_logo
#SBATCH --partition=gpu
#SBATCH --array=0-14%3
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x_%A_%a_%N.log

cd ~/model-training
source .venv/bin/activate

echo "Fold: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python testing_validation.py fold=$SLURM_ARRAY_TASK_ID
