#!/bin/bash
#SBATCH --job-name=train_yolo
#SBATCH --output=logs/train_yolo_%j.out
#SBATCH --error=logs/train_yolo_%j.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Run training with uv
uv run python -m aemodes.models.detection.train_yolo
