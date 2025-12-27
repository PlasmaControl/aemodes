#!/bin/bash
#SBATCH --job-name=train_mask_rcnn
#SBATCH --output=logs/%j_train_mask_rcnn.out
#SBATCH --error=logs/%j_train_mask_rcnn.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12GB
#SBATCH --gres=gpu:1

source .venv/bin/activate
python -m aemodes.pipeline.step_3_train.mask_rcnn
