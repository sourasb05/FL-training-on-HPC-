#!/bin/bash
#SBATCH -A uppmax2025-2-413
#SBATCH --partition=gpu
#SBATCH --job-name=fl_cifar_vram_flat
#SBATCH --output=<filename>_%j.out
#SBATCH --error=<filename>_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

WORKDIR=</path/to/your/workdir>
SIF=$WORKDIR/fl_cifar.sif

nvidia-smi --query-gpu=timestamp,index,memory.used,memory.free,memory.total \
           --format=csv -l 5 >> gpu_mem_log_$SLURM_JOB_ID.csv &
NVPID=$!

apptainer exec --nv \
    --bind $WORKDIR:/workspace \
    $SIF python3 /workspace/fl_sequential_cifar_vram_flat.py

kill $NVPID
echo "End: $(date)"

