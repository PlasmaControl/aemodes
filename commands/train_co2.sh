#!/bin/bash
#SBATCH --job-name=train_co2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=24G
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

module purge
source /scratch/gpfs/nc1514/specseg/.venv/bin/activate

srun python /scratch/gpfs/nc1514/specseg/projects/co2modes/main.py
# srun python /scratch/gpfs/nc1514/specseg/projects/co2modes/evaluate.py