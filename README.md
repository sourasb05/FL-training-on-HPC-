# Federated Learning Demo — CIFAR-10 / ResNet-18

Federated Averaging (FedAvg) on CIFAR-10 with ResNet-18, progressing from a naive single-GPU baseline to a VRAM-optimised single-GPU version, then to a multi-GPU parallel implementation.

---

## Overview

| Script | Mode | GPUs | VRAM strategy |
|---|---|---|---|
| `fl_sequential_cifar.py` | Single-GPU naive | 1 | All client models stay on GPU simultaneously |
| `fl_sequential_cifar_vram_opt.py` | Single-GPU optimised | 1 | One client on GPU at a time; offload after each |
| `fl_multigpu_vram_opt.py` | Multi-GPU optimised | N | Clients distributed across GPUs; offload after each |

---

## Prerequisites

```bash
conda activate myenv
# packages required: torch, torchvision, numpy, matplotlib
```

CIFAR-10 (~170 MB) is downloaded automatically on first run into `./data/`.

---

## Step 1 — Naive Single-GPU (`fl_sequential_cifar.py`)

### What it does

- All clients train sequentially on one GPU
- After each client finishes, its model **stays on GPU** until all clients complete the round
- FedAvg aggregation happens on GPU with all client models loaded simultaneously
- **Problem:** VRAM grows linearly with number of clients — OOM at scale (400+ clients on a 46 GB GPU)

### Key parameters

```python
NUM_CLIENTS  = 400
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 5
BATCH_SIZE   = 128
```

### Run on cluster (SLURM)

```bash
sbatch run_fl_sequential_cifar.sh
```

SLURM settings in `run_fl_sequential_cifar.sh`:

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
```

Monitor the job:

```bash
tail -f fl_cifar_<job_id>.out
```

---

## Step 2 — Optimised Single-GPU (`fl_sequential_cifar_vram_opt.py`)

### What it does

- Clients still train sequentially on one GPU
- After each client finishes:
  1. Weights are copied to CPU (`deepcopy(model.state_dict())`)
  2. Model is moved off GPU (`model.cpu()`)
  3. VRAM is freed immediately (`torch.cuda.empty_cache()`)
- Only **one client model** lives on GPU at any time
- FedAvg aggregation runs entirely on CPU
- **Result:** Peak VRAM = global model + 1 local model, regardless of client count — scales to 600+ clients

### Key parameters

```python
NUM_CLIENTS  = 600
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 5
BATCH_SIZE   = 128
```

### Run on cluster (SLURM)

```bash
sbatch run_fl_sequential_cifar_vram_opt.sh
```

SLURM settings in `run_fl_sequential_cifar_vram_opt.sh`:

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
```

Monitor the job:

```bash
tail -f fl_cifar_vram_opt_<job_id>.out
```

---

## Step 3 — Multi-GPU Optimised (`fl_multigpu_vram_opt.py`)

### What it does

- Spawns one worker process per GPU using `torch.multiprocessing`
- Clients are distributed **round-robin** across all GPUs (e.g., 10 clients across 5 GPUs = 2 clients/GPU)
- Each GPU worker trains its assigned clients **sequentially with VRAM offload** (same as Step 2)
- All GPU workers run **in parallel**, so wall time per round = slowest GPU worker
- Global model and FedAvg aggregation live entirely on CPU
- Communication via `mp.Queue`: server sends model bytes to workers; workers return CPU state_dicts

### Architecture

```
Main process (server)
  ├── global model on CPU
  ├── FedAvg aggregation on CPU
  ├── evaluation on GPU 0
  └── spawns N GPU worker processes
        ├── GPU 0 worker → trains clients 0, 5  (sequential + VRAM offload)
        ├── GPU 1 worker → trains clients 1, 6
        ├── GPU 2 worker → trains clients 2, 7
        ├── GPU 3 worker → trains clients 3, 8
        └── GPU 4 worker → trains clients 4, 9
```

### Key parameters

```python
NUM_CLIENTS  = 10
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 5
BATCH_SIZE   = 128
```

### Run on cluster (SLURM)

```bash
sbatch run_fl_multigpu_vram_opt.sh
```

SLURM settings in `run_fl_multigpu_vram_opt.sh`:

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12   # 5 GPU workers × 2 DataLoader workers + server + headroom
#SBATCH --gres=gpu:5
#SBATCH --mem=16G
#SBATCH --time=4:00:00
```

Monitor the job:

```bash
tail -f fl_multigpu_vram_opt_<job_id>.out
```

---

## GPU Memory Monitoring

All SLURM scripts log GPU memory via `nvidia-smi` every 5 seconds into a CSV:

```
gpu_mem_log_<job_id>.csv
```

### Plot memory usage per GPU

```bash
python plot_gpu_mem.py <job_id>
```

Produces `gpu_mem_<job_id>.png` — one subplot per GPU showing used VRAM over time, peak, and total.

Example:

```bash
python plot_gpu_mem.py 5920546
```

---

## SLURM commands

### Submit a job

```bash
sbatch run_fl_sequential_cifar.sh
```

### Check job status

```bash
squeue -u $USER               # all your jobs
squeue -j <job_id>            # specific job
watch -n 5 squeue -u $USER    # refresh every 5 seconds
```

### Cancel a job

```bash
scancel <job_id>              # cancel one job
scancel -u $USER              # cancel all your jobs
```

### View job output live

```bash
tail -f fl_cifar_<job_id>.out
tail -f fl_cifar_<job_id>.err
```

### Job details and resource usage

```bash
scontrol show job <job_id>    # full job info (nodes, CPUs, GPUs, state)
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocCPUS,AllocGRES
                              # resource usage after job completes
```

### Check available GPU nodes

```bash
sinfo -p gpu                  # partition state and availability
sinfo -p gpu -o "%N %G %C %m" # nodes with GPU, CPU, memory info
```

### Interactive GPU session (for debugging)

```bash
srun --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
     --mem=16G --time=1:00:00 --account=uppmax2025-2-413 --pty bash
```
