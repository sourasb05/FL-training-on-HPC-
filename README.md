# Federated Learning Demo — CIFAR-10 / ResNet-18

Federated Averaging (FedAvg) on CIFAR-10 with ResNet-18, progressing from a naive single-GPU baseline to a VRAM-flat single-GPU version, then to a multi-GPU parallel implementation with aggressive VRAM teardown.

---

## Overview

| Script | Mode | GPUs | VRAM Strategy |
|---|---|---|---|
| `fl_sequential_cifar.py` | Naive single-GPU | 1 | All client models stay on GPU simultaneously — OOM at scale |
| `fl_sequential_cifar_vram_flat.py` | VRAM-flat single-GPU | 1 | Aggressive 6-step teardown — flat nvidia-smi baseline between clients |
| `fl_multigpu_vram_flat.py` | VRAM-flat multi-GPU | N | Clients distributed across GPUs; 6-step teardown after each client |

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
- **Problem:** VRAM grows linearly with number of clients — OOM beyond ~5 clients on a 46 GB GPU

### Why it fails at scale

- All N client models resident on GPU simultaneously → peak VRAM = global model + N × local model
- Gradient buffers and optimizer momentum never freed between clients
- No `empty_cache()` → CUDA driver never releases blocks during the round

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

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
```

---

## Step 2 — VRAM-Flat Single-GPU (`fl_sequential_cifar_vram_flat.py`)

### What it does

- Same sequential training as Step 2 but with an aggressive 6-step VRAM teardown after every client
- Ensures `nvidia-smi` shows a **flat baseline** between clients, not just a reset at round boundaries

### 6-step teardown after each client

```python
opt.zero_grad(set_to_none=True)                               # 1. free .grad buffers on GPU
weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}  # 2. explicit CPU snapshot
del opt, criterion                                            # 3. free SGD momentum buffers
model.cpu()                                                   # 4. move model off GPU
torch.cuda.synchronize()                                      # 5. wait for async D2H transfers
torch.cuda.empty_cache()                                      # 6. release blocks to CUDA driver
```

### Why each step matters

| Step | What it frees |
|------|--------------|
| `zero_grad(set_to_none=True)` | Deallocates `.grad` buffers — not just zeroed in-place |
| `{k: v.cpu().clone()}` | Avoids temporary GPU copies that linger during `deepcopy` |
| `del opt, criterion` | SGD momentum buffers freed before `model.cpu()` |
| `model.cpu()` | Model parameters and buffers moved off GPU |
| `synchronize()` | Waits for async D2H transfers — without this, `empty_cache()` sees in-flight blocks |
| `empty_cache()` | Releases all freed blocks back to the CUDA driver immediately |

### Key parameters

```python
NUM_CLIENTS  = 600
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 5
BATCH_SIZE   = 128
```

### Run on cluster (SLURM)

```bash
sbatch run_fl_sequential_cifar_vram_flat.sh
```

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
```

---

## Step 3 — VRAM-Flat Multi-GPU (`fl_multigpu_vram_flat.py`)

### What it does

- Spawns one worker process per GPU using `torch.multiprocessing`
- Clients distributed **round-robin** across GPUs
- Each GPU worker trains its assigned clients **sequentially** with the same 6-step VRAM teardown as Step 3
- All GPU workers run **in parallel** — wall time per round = slowest GPU worker
- Global model and FedAvg aggregation live entirely on CPU
- Communication via `mp.Queue`: server serializes model to bytes → workers deserialize and train → return CPU weights as bytes

### Architecture

```
Main process (server — CPU)
  ├── global model on CPU
  ├── FedAvg aggregation on CPU
  ├── evaluation on GPU 0 (then moved back to CPU)
  └── spawns M GPU worker processes
        ├── GPU 0 worker → trains clients [0, 2, 4, ...] sequentially + 6-step teardown
        └── GPU 1 worker → trains clients [1, 3, 5, ...] sequentially + 6-step teardown
```

### Key implementation details

- **`mp.Process`** — true parallelism across GPUs (not threads, which are blocked by Python's GIL)
- **`state_to_bytes()` / `bytes_to_state()`** — GPU tensors cannot cross process boundaries directly; serialization is required
- **`torch.cuda.set_device(device)`** — pins each worker to its assigned GPU
- **Round-robin assignment** — static; GPU with more/heavier clients determines round time

### Key parameters

```python
NUM_CLIENTS  = 10
NUM_ROUNDS   = 5
LOCAL_EPOCHS = 5
BATCH_SIZE   = 128
```

### Run on cluster (SLURM)

```bash
sbatch run_fl_multigpu_vram_flat.sh
```

```bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6    # 2 GPUs × 2 DataLoader workers + server overhead
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=1:00:00
```

### Known limitation

Multi-GPU adds serialization overhead and inter-process communication cost. With small client counts (< ~20), sequential single-GPU is faster. Multi-GPU pays off when `num_clients >> num_gpus` and each client trains long enough to amortize the communication cost.

---

## GPU Memory Monitoring

All SLURM scripts log GPU memory via `nvidia-smi` every 5 seconds:

```
gpu_mem_log_<job_id>.csv
```

### Plot memory usage

```bash
python plot_gpu_mem.py <job_id>
```

Produces `gpu_mem_<job_id>.png` — one subplot per GPU showing used VRAM over time, peak, and total.

---

## SLURM Commands

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
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocCPUS,AllocGRES
```

### Check available GPU nodes

```bash
sinfo -p gpu
sinfo -p gpu -o "%N %G %C %m"
```

### Interactive GPU session (for debugging)

```bash
srun --partition=gpu --gres=gpu:1 --ntasks=1 --cpus-per-task=4 \
     --mem=16G --time=1:00:00 --account=uppmax2025-2-413 --pty bash
```
