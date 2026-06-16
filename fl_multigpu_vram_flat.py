"""
FedAvg - Multi-GPU + VRAM-flat  (N clients, M GPUs, N >= M)
=================================================================
Architecture:
  - 1 server process  : global model on CPU, aggregates weights, evaluates
  - M GPU worker processes: each owns one GPU, trains clients assigned to it
                            sequentially with aggressive VRAM teardown between clients

Key difference vs fl_multigpu_vram_opt.py
------------------------------------------
After each client finishes training, a 6-step aggressive VRAM teardown is
applied so that nvidia-smi shows a flat baseline between clients:

  1. opt.zero_grad(set_to_none=True)
       → deallocates .grad buffers on GPU (vs zeroing in-place)
  2. weights = {k: v.cpu().clone() ...}
       → moves each tensor to CPU explicitly — avoids temporary GPU copies
         that linger during deepcopy
  3. del opt, criterion
       → SGD momentum buffers live in optimizer state on GPU; deleting
         before model.cpu() frees those buffers first
  4. model.cpu()
       → moves parameters/buffers off GPU
  5. torch.cuda.synchronize(device)
       → waits for all async D2H copy ops to complete so empty_cache()
         sees no in-flight work holding block references
  6. torch.cuda.empty_cache()
       → releases all freed blocks back to the CUDA driver immediately

Communication via multiprocessing Queues:
  server → gpu_worker : (round, global state_dict bytes, [(cid, indices), ...])
  gpu_worker → server : [(cid, elapsed, mem_peak, weights_bytes), ...]
"""

import copy
import io
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ── Model ─────────────────────────────────────────────────────────────────────
def build_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ── Non-IID data split ────────────────────────────────────────────────────────
def non_iid_split(dataset, num_clients=10, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    class_indices = [np.where(labels == c)[0] for c in range(len(np.unique(labels)))]
    client_indices = [[] for _ in range(num_clients)]
    for c_idx in class_indices:
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (proportions * len(c_idx)).astype(int)
        splits[-1] = len(c_idx) - splits[:-1].sum()
        start = 0
        for k, n in enumerate(splits):
            client_indices[k].extend(c_idx[start:start + n].tolist())
            start += n
    return client_indices

# ── Safe queue serialization ──────────────────────────────────────────────────
def state_to_bytes(state_dict):
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in state_dict.items()}, buf)
    return buf.getvalue()

def bytes_to_state(b):
    return torch.load(io.BytesIO(b), weights_only=True)

# ── FedAvg aggregation (CPU only) ─────────────────────────────────────────────
def fedavg(state_dicts):
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        avg[key] = torch.stack([sd[key].cpu().float() for sd in state_dicts]).mean(0)
    return avg

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model = model.to(device)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    model.cpu()
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    return 100.0 * correct / total

# ── Train one client on a GPU, aggressive teardown after ─────────────────────
def train_one_client(global_state_bytes, client_indices, device,
                     batch_size, local_epochs, data_path, client_id):
    """
    Loads global weights onto GPU, trains, then performs 6-step aggressive
    VRAM teardown so nvidia-smi shows a flat baseline between clients.
    Returns: (cpu_state_dict, elapsed, mem_peak_MB)
    """
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_path, train=True,
                                             download=False, transform=transform)
    loader = DataLoader(Subset(train_set, client_indices),
                        batch_size=batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    model = build_resnet18()
    model.load_state_dict(bytes_to_state(global_state_bytes))
    model = model.to(device)
    model.train()

    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    for ep in range(local_epochs):
        running_loss = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"  [client {client_id} / {device}] "
              f"epoch {ep+1}/{local_epochs}  loss={avg_loss:.4f}", flush=True)

    elapsed  = time.time() - t0
    mem_peak = torch.cuda.max_memory_allocated(device) / 1024**2

    # ── 6-step aggressive VRAM teardown ──────────────────────────────────────
    # 1. Free .grad buffers — set_to_none deallocates instead of zeroing
    opt.zero_grad(set_to_none=True)
    # 2. Snapshot weights to CPU explicitly — avoids temp GPU copies from deepcopy
    cpu_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # 3. Delete optimizer — frees SGD momentum buffers on GPU before model.cpu()
    del opt, criterion
    # 4. Move model parameters/buffers off GPU
    model.cpu()
    # 5. Wait for all async D2H transfers to complete
    torch.cuda.synchronize(device)
    # 6. Release all freed blocks back to CUDA driver
    torch.cuda.empty_cache()

    vram_after = torch.cuda.memory_allocated(device) / 1024**2
    print(f"  [client {client_id} / {device}] done in {elapsed:.2f}s  "
          f"peak={mem_peak:.0f}MB  VRAM after offload={vram_after:.0f}MB", flush=True)

    return cpu_weights, elapsed, mem_peak

# ── GPU worker process ────────────────────────────────────────────────────────
def gpu_worker(gpu_id, assigned_clients, num_rounds, local_epochs,
               batch_size, data_path, task_queue, result_queue):
    """
    One process per GPU. Each round:
      1. Receives global model bytes + list of (cid, indices) to train
      2. Trains each assigned client SEQUENTIALLY with aggressive VRAM teardown
      3. Returns all CPU state_dicts to server
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    for rnd in range(1, num_rounds + 1):
        global_bytes, client_list = task_queue.get()

        round_results = []
        for cid, indices in client_list:
            print(f"  [GPU {gpu_id}] starting client {cid}  "
                  f"({len(indices)} samples)", flush=True)
            cpu_weights, elapsed, mem_peak = train_one_client(
                global_bytes, indices, device,
                batch_size, local_epochs, data_path, cid
            )
            round_results.append((cid, elapsed, mem_peak,
                                   state_to_bytes(cpu_weights)))

        result_queue.put((gpu_id, round_results))

# ── Server (main process) ─────────────────────────────────────────────────────
def server(num_clients, num_gpus, num_rounds, local_epochs,
           batch_size, data_path, client_indices_list, round_log_file=None):
    """
    Global model lives on CPU throughout.
    Distributes clients round-robin across GPUs each round.
    Aggregates CPU state_dicts via FedAvg on CPU.
    Evaluates on GPU 0, then moves global model back to CPU.
    """
    eval_device = torch.device("cuda:0")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_set = torchvision.datasets.CIFAR10(root=data_path, train=False,
                                            download=False, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=False,
                             num_workers=2, pin_memory=True)

    global_model = build_resnet18()
    print(f"Server: {sum(p.numel() for p in global_model.parameters()):,} parameters")
    print(f"Server: {num_clients} clients distributed across {num_gpus} GPUs "
          f"({num_clients // num_gpus}–{(num_clients + num_gpus - 1) // num_gpus} clients/GPU)\n")

    gpu_client_map = [[] for _ in range(num_gpus)]
    for cid in range(num_clients):
        gpu_client_map[cid % num_gpus].append(cid)

    print("Client → GPU assignment:")
    for gid, cids in enumerate(gpu_client_map):
        print(f"  GPU {gid}: clients {cids}")
    print()

    task_queues  = [mp.Queue() for _ in range(num_gpus)]
    result_queue = mp.Queue()

    processes = []
    for gid in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gid, gpu_client_map[gid], num_rounds, local_epochs,
                  batch_size, data_path, task_queues[gid], result_queue)
        )
        p.start()
        processes.append(p)

    round_stats = []

    def log_round_marker(tag, rnd):
        if round_log_file:
            ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
            with open(round_log_file, "a") as f:
                f.write(f"{ts}, ROUND_{rnd}_{tag}\n")

    for rnd in range(1, num_rounds + 1):
        print(f"{'='*60}", flush=True)
        print(f"Round {rnd}/{num_rounds}", flush=True)
        round_start = time.time()
        log_round_marker("START", rnd)

        global_bytes = state_to_bytes(global_model.state_dict())
        for gid in range(num_gpus):
            client_list = [(cid, client_indices_list[cid])
                           for cid in gpu_client_map[gid]]
            task_queues[gid].put((global_bytes, client_list))
        print(f"  [server] dispatched {num_clients} clients to {num_gpus} GPUs", flush=True)

        all_weights  = {}
        all_peaks    = []
        gpus_done    = 0
        while gpus_done < num_gpus:
            gid, round_results = result_queue.get()
            for cid, elapsed, mem_peak, state_bytes in round_results:
                all_weights[cid] = bytes_to_state(state_bytes)
                all_peaks.append(mem_peak)
                print(f"  [server] client {cid} (GPU {gid}): "
                      f"{elapsed:.2f}s  peak={mem_peak:.0f}MB", flush=True)
            gpus_done += 1

        print(f"  [server] FedAvg over {num_clients} client weights (CPU)...", flush=True)
        ordered = [all_weights[cid] for cid in range(num_clients)]
        global_model.load_state_dict(fedavg(ordered))

        # evaluate() now includes synchronize + empty_cache internally
        acc = evaluate(global_model, test_loader, eval_device)

        round_time = time.time() - round_start
        log_round_marker("END", rnd)
        avg_peak   = float(np.mean(all_peaks))
        max_peak   = float(np.max(all_peaks))
        print(f"  [server] >> Round {rnd} done in {round_time:.2f}s  |  "
              f"peak VRAM avg={avg_peak:.0f}MB  max={max_peak:.0f}MB  |  "
              f"Test acc: {acc:.2f}%", flush=True)
        round_stats.append(dict(round=rnd, time=round_time, acc=acc,
                                avg_peak=avg_peak, max_peak=max_peak))

    for p in processes:
        p.join()

    print(f"\n{'='*60}")
    print(f"SUMMARY (CIFAR-10 / ResNet-18 / {num_clients} clients / {num_gpus} GPUs / VRAM-flat)")
    print(f"{'='*60}")
    total_time = sum(s["time"] for s in round_stats)
    print(f"Total wall time  : {total_time:.2f}s")
    print(f"Clients per GPU  : ~{num_clients // num_gpus}")
    print()
    print(f"  {'Round':<8} {'Avg peak/client (MB)':<24} {'Max peak (MB)':<18} {'Test Acc (%)':<12}")
    print(f"  {'-'*8} {'-'*24} {'-'*18} {'-'*12}")
    for s in round_stats:
        print(f"  {s['round']:<8} {s['avg_peak']:<24.0f} {s['max_peak']:<18.0f} {s['acc']:<12.2f}")
    print()
    print("Key observations:")
    print(f"  GPU workers train {num_clients // num_gpus}–"
          f"{(num_clients + num_gpus - 1) // num_gpus} clients sequentially per GPU.")
    print("  6-step VRAM teardown after each client: flat nvidia-smi baseline.")
    print("  Global model and FedAvg aggregation run entirely on CPU.")
    print("  Wall time per round = slowest GPU worker (parallel across GPUs).")

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    NUM_CLIENTS  = 5
    NUM_ROUNDS   = 5
    LOCAL_EPOCHS = 5
    BATCH_SIZE   = 128
    DATA_PATH    = "./data"

    job_id    = os.environ.get("SLURM_JOB_ID", "local")
    round_log = f"round_log_{job_id}.csv"

    num_gpus = torch.cuda.device_count()
    print(f"GPUs available : {num_gpus}")
    if num_gpus < 1:
        raise SystemExit("Need at least 1 GPU.")

    print(f"Clients        : {NUM_CLIENTS}  ({NUM_CLIENTS // num_gpus}–"
          f"{(NUM_CLIENTS + num_gpus - 1) // num_gpus} per GPU, sequential + VRAM-flat)")
    print(f"Rounds         : {NUM_ROUNDS}")
    print(f"Local epochs   : {LOCAL_EPOCHS} per round")
    print(f"Dataset        : CIFAR-10  |  Model: ResNet-18")
    print(f"Mode           : Multi-GPU + VRAM-flat\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    print("Loading CIFAR-10...", flush=True)
    train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                             download=True, transform=transform)
    torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                 download=True, transform=transform)

    client_indices_list = non_iid_split(train_set, NUM_CLIENTS)

    print("Non-IID class distribution per client:")
    all_labels = np.array(train_set.targets)
    for i, idxs in enumerate(client_indices_list):
        counts = np.bincount(all_labels[idxs], minlength=10)
        print(f"  Client {i:>3}: {counts}  (total={len(idxs)})")
    print()

    mp.set_start_method("spawn", force=True)
    server(NUM_CLIENTS, num_gpus, NUM_ROUNDS, LOCAL_EPOCHS,
           BATCH_SIZE, DATA_PATH, client_indices_list,
           round_log_file=round_log)

if __name__ == "__main__":
    main()
