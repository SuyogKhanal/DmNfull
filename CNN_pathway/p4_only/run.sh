#!/bin/bash
#SBATCH --job-name=cnn_p4_only
#SBATCH --partition=gpu
#SBATCH --qos=batch-long
#SBATCH --gpus=1
#SBATCH --mem=40G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -eo pipefail
module purge
module load Anaconda3
source /home/s226137394/.bashrc
eval "$(conda shell.bash hook)"
conda activate maze
mkdir -p slurm_logs
python -u -m CNN_pathway.p4_only.pipeline --force_restart "$@"
