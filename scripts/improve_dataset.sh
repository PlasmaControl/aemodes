#!/bin/bash
#SBATCH --job-name=improve_dataset
#SBATCH --output=logs/%j_improve_dataset.out
#SBATCH --error=logs/%j_improve_dataset.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

uv run python scripts/improve_dataset.py