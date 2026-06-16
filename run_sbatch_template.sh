#!/bin/bash
#SBATCH -A uppmax2025-2-413
#SBATCH --partition=gpu
#SBATCH --job-name=fl_cifar
#SBATCH --output=fl_cifar_%j.out
#SBATCH --error=fl_cifar_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate <your_env_name>

cd <your_project_path>

nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
           --format=csv -l 5 >> gpu_mem_log_$SLURM_JOB_ID.csv &
NVPID=$!

python fl_sequential_cifar.py

kill $NVPID

echo "End: $(date)"
