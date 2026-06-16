"""
Federated Averaging (FedAvg) - Sequential Training  [VRAM-flat variant]
Multiple clients, non-IID CIFAR-10 data, one GPU, Model: ResNet-18.

Key difference vs fl_sequential_cifar_vram_opt.py
---------------------------------------------------
After each client finishes training we do a more aggressive VRAM teardown
so that nvidia-smi shows a flat baseline between clients (not just a reset
at round boundaries):

  1. opt.zero_grad(set_to_none=True)
       → deallocates .grad buffers on GPU (vs zeroing in-place)
  2. weights = {k: v.cpu().clone() ...}
       → moves each tensor to CPU explicitly before deepcopy so no temporary
         GPU copies linger during the snapshot
  3. del opt, criterion
       → SGD momentum buffers live in the optimizer state on GPU; deleting
         the optimizer before model.cpu() frees those buffers first
  4. model.cpu()
       → moves parameters/buffers off GPU
  5. torch.cuda.synchronize()
       → waits for all async CUDA D2H copy ops to complete so empty_cache()
         sees no in-flight work holding block references
  6. torch.cuda.empty_cache()
       → releases freed blocks back to the CUDA driver immediately

Together these ensure every GPU tensor from the client (weights, gradients,
optimizer state, activations) is provably dead before the cache flush, so
nvidia-smi should show a flat baseline between clients.

FedAvg aggregation is done on CPU to keep GPU pressure low.
"""

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

# ── Model: ResNet-18 adapted for CIFAR-10 ────────────────────────────────────
def build_resnet18(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ── Non-IID data split (Dirichlet distribution) ───────────────────────────────
def non_iid_split(dataset, num_clients=5, alpha=0.5, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    client_indices = [[] for _ in range(num_clients)]
    for c_idx in class_indices:
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        splits = (proportions * len(c_idx)).astype(int)
        splits[-1] = len(c_idx) - splits[:-1].sum()
        start = 0
        for k, n in enumerate(splits):
            client_indices[k].extend(c_idx[start:start + n].tolist())
            start += n

    return [Subset(dataset, idxs) for idxs in client_indices]

# ── FedAvg aggregation (runs on CPU) ─────────────────────────────────────────
def fedavg(weight_list):
    """Average a list of state_dicts (CPU tensors). Returns a new state_dict."""
    avg = copy.deepcopy(weight_list[0])
    for key in avg:
        avg[key] = torch.stack([w[key].float() for w in weight_list]).mean(0)
    return avg

# ── Client local training ─────────────────────────────────────────────────────
def train_client(model, loader, device, epochs=2, lr=0.01, client_id=None):
    """
    Train model in-place on `device`.
    Returns a CPU state_dict, then aggressively frees all GPU memory so that
    nvidia-smi shows a flat baseline between clients.
    """
    model = model.to(device)
    model.train()
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    prefix = f"  [client {client_id}]" if client_id is not None else "  "
    num_batches = len(loader)

    for ep in range(epochs):
        ep_start = time.time()
        running_loss = 0.0
        for bid, (x, y) in enumerate(loader):
            batch_start = time.time()
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            batch_time = time.time() - batch_start
            running_loss += loss.item()

            if bid == 0 or (bid + 1) % max(1, num_batches // 4) == 0 or bid + 1 == num_batches:
                avg_loss = running_loss / (bid + 1)
                print(f"{prefix} epoch {ep+1}/{epochs}  "
                      f"batch {bid+1:>4}/{num_batches}  "
                      f"loss={avg_loss:.4f}  "
                      f"batch_time={batch_time*1000:.1f}ms", flush=True)

        ep_time = time.time() - ep_start
        print(f"{prefix} epoch {ep+1}/{epochs} done in {ep_time:.2f}s  "
              f"avg_loss={running_loss/num_batches:.4f}", flush=True)

    # ── Aggressive VRAM teardown for flat nvidia-smi baseline ────────────────
    # 1. Deallocate .grad buffers (set_to_none frees GPU memory, not just zeros)
    opt.zero_grad(set_to_none=True)
    # 2. Snapshot weights to CPU explicitly — avoids temporary GPU copies from deepcopy
    weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # 3. Delete optimizer before model.cpu() — frees SGD momentum buffers on GPU
    del opt, criterion
    # 4. Move model parameters/buffers off GPU
    model.cpu()
    # 5. Wait for all async CUDA D2H copy ops to finish before flushing cache
    torch.cuda.synchronize()
    # 6. Release all freed blocks back to the CUDA driver
    torch.cuda.empty_cache()
    return weights

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
    return 100.0 * correct / total

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    def _cuda_usable():
        if not torch.cuda.is_available():
            return False
        try:
            torch.zeros(1).cuda()
            return True
        except Exception:
            return False

    DEVICE = torch.device("cuda:0" if _cuda_usable() else "cpu")
    NUM_CLIENTS  = 5
    NUM_ROUNDS   = 5
    LOCAL_EPOCHS = 5
    BATCH_SIZE   = 128

    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}  |  Rounds: {NUM_ROUNDS}  |  Dataset: CIFAR-10  |  "
          f"Model: ResNet-18  |  Mode: SEQUENTIAL (VRAM-flat)\n")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print("Loading CIFAR-10 (downloading if needed ~170MB)...", flush=True)
    t_data = time.time()
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=transform_train)
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                             download=True, transform=transform_test)
    print(f"Dataset ready in {time.time()-t_data:.1f}s  "
          f"(train={len(train_set)}, test={len(test_set)})", flush=True)

    client_datasets = non_iid_split(train_set, NUM_CLIENTS)
    client_loaders  = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True)
                       for ds in client_datasets]
    test_loader     = DataLoader(test_set, batch_size=256, shuffle=False,
                                 num_workers=2, pin_memory=True)

    print("Class distribution per client:")
    all_labels = np.array(train_set.targets)
    for i, ds in enumerate(client_datasets):
        counts = np.bincount(all_labels[ds.indices], minlength=10)
        print(f"  Client {i}: {counts}  (total={len(ds.indices)})")
    print()

    # Global model lives on CPU between rounds; moved to GPU only for eval.
    global_model = build_resnet18()
    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}\n")

    round_stats = []

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"{'='*60}")
        print(f"Round {rnd}/{NUM_ROUNDS}")
        round_start = time.time()

        local_weights = []   # list of CPU state_dicts — no GPU memory held here
        client_peaks  = []

        for cid in range(NUM_CLIENTS):
            t0 = time.time()
            # deepcopy on CPU, then train_client moves it to GPU
            local_model = copy.deepcopy(global_model)
            print(f"  [client {cid}] starting training  "
                  f"({len(client_loaders[cid])} batches x {LOCAL_EPOCHS} epochs)", flush=True)

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(None)

            # Returns CPU state_dict; GPU is freed inside train_client
            weights = train_client(local_model, client_loaders[cid], DEVICE,
                                   LOCAL_EPOCHS, client_id=cid)

            if DEVICE.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(None) / 1024**2
                mem_peak  = torch.cuda.max_memory_allocated(None) / 1024**2
            else:
                mem_alloc = mem_peak = 0.0

            local_weights.append(weights)
            client_peaks.append(mem_peak)
            elapsed = time.time() - t0
            print(f"  Client {cid} trained in {elapsed:.2f}s  |  "
                  f"Mem after offload: alloc={mem_alloc:.0f}MB  peak={mem_peak:.0f}MB",
                  flush=True)

        # All N client models already offloaded to CPU right here — GPU is free
        vram_after_offload = torch.cuda.memory_allocated(None) / 1024**2 if DEVICE.type == "cuda" else 0.0
        print(f"  [round {rnd}] VRAM after all {NUM_CLIENTS} clients offloaded to CPU: {vram_after_offload:.0f}MB", flush=True)

        print(f"  [round {rnd}] aggregating {len(local_weights)} client models (CPU)", flush=True)
        # FedAvg entirely on CPU — no GPU needed for aggregation
        avg_weights = fedavg(local_weights)
        global_model.load_state_dict(avg_weights)

        round_time = time.time() - round_start
        print(f"  [round {rnd}] evaluating on test set...", flush=True)
        acc = evaluate(global_model, test_loader, DEVICE)
        # Move global model back to CPU after eval
        global_model.cpu()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if DEVICE.type == "cuda":
            round_mem_alloc = torch.cuda.memory_allocated(None) / 1024**2
        else:
            round_mem_alloc = 0.0

        print(f"  >> Round {rnd} done in {round_time:.2f}s  |  "
              f"Mem: alloc={round_mem_alloc:.0f}MB  |  "
              f"Test acc: {acc:.2f}%")
        round_stats.append(dict(round=rnd, time=round_time,
                                mem_alloc=round_mem_alloc,
                                vram_after_offload=vram_after_offload,
                                avg_peak=np.mean(client_peaks),
                                acc=acc))

    print(f"\n{'='*60}")
    print("SEQUENTIAL SUMMARY (CIFAR-10 / ResNet-18 / VRAM-flat)")
    print(f"{'='*60}")
    total_time   = sum(s["time"] for s in round_stats)
    overall_peak = np.mean([s["avg_peak"] for s in round_stats])
    print(f"Total wall time          : {total_time:.2f}s")
    print(f"Avg peak VRAM per client : {overall_peak:.0f}MB")
    print(f"Final accuracy           : {round_stats[-1]['acc']:.2f}%")
    print()
    print(f"  {'Round':<8} {'VRAM all clients offloaded (MB)':<32} {'Avg peak/client (MB)':<24} {'Test Acc (%)':<12}")
    print(f"  {'-'*8} {'-'*32} {'-'*24} {'-'*12}")
    for s in round_stats:
        print(f"  {s['round']:<8} {s['vram_after_offload']:<32.0f} {s['avg_peak']:<24.0f} {s['acc']:<12.2f}")
    print()

if __name__ == "__main__":
    main()
