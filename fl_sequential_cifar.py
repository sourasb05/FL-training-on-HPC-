"""
Federated Averaging (FedAvg) - Sequential Training
Multiple clients
non-IID CIFAR-10 data
one GPU
Model: ResNet-18 (pretrained=False).
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

# ── FedAvg aggregation ────────────────────────────────────────────────────────
def fedavg(global_model, client_models):
    global_state = global_model.state_dict()
    for key in global_state:
        global_state[key] = torch.stack(
            [cm.state_dict()[key].float() for cm in client_models]
        ).mean(0)
    global_model.load_state_dict(global_state)

# ── Client local training ─────────────────────────────────────────────────────
def train_client(model, loader, device, epochs=2, lr=0.01, client_id=None):
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

# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
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
    DEVICE       = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLIENTS  = 400
    NUM_ROUNDS   = 5
    LOCAL_EPOCHS = 5
    BATCH_SIZE   = 128

    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}  |  Rounds: {NUM_ROUNDS}  |  Dataset: CIFAR-10  |  Model: ResNet-18  |  Mode: SEQUENTIAL\n")

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

    global_model = build_resnet18().to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}\n")

    round_stats = []

    for rnd in range(1, NUM_ROUNDS + 1):
        print(f"{'='*60}")
        print(f"Round {rnd}/{NUM_ROUNDS}")
        round_start = time.time()

        client_models = []
        client_peaks  = []
        for cid in range(NUM_CLIENTS):
            t0 = time.time()
            local_model = copy.deepcopy(global_model)
            print(f"  [client {cid}] starting training  "
                  f"({len(client_loaders[cid])} batches x {LOCAL_EPOCHS} epochs)", flush=True)

            torch.cuda.reset_peak_memory_stats(DEVICE)
            train_client(local_model, client_loaders[cid], DEVICE, LOCAL_EPOCHS, client_id=cid)

            mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2
            mem_peak  = torch.cuda.max_memory_allocated(DEVICE) / 1024**2

            client_models.append(local_model)
            client_peaks.append(mem_peak)
            elapsed = time.time() - t0
            print(f"  Client {cid} trained in {elapsed:.2f}s  |  "
                  f"Mem: alloc={mem_alloc:.0f}MB  peak={mem_peak:.0f}MB", flush=True)

        # All N client models are sitting idle on GPU right here — worst-case VRAM
        vram_all_clients = torch.cuda.memory_allocated(DEVICE) / 1024**2
        print(f"  [round {rnd}] VRAM with all {NUM_CLIENTS} client models idle on GPU: {vram_all_clients:.0f}MB", flush=True)

        print(f"  [round {rnd}] aggregating {len(client_models)} client models", flush=True)
        fedavg(global_model, client_models)

        round_time = time.time() - round_start
        print(f"  [round {rnd}] evaluating on test set...", flush=True)
        acc = evaluate(global_model, test_loader, DEVICE)

        round_mem_alloc = torch.cuda.memory_allocated(DEVICE) / 1024**2

        print(f"  >> Round {rnd} done in {round_time:.2f}s  |  "
              f"Mem: alloc={round_mem_alloc:.0f}MB  |  "
              f"Test acc: {acc:.2f}%")
        round_stats.append(dict(round=rnd, time=round_time,
                                mem_alloc=round_mem_alloc,
                                vram_all_clients=vram_all_clients,
                                avg_peak=np.mean(client_peaks),
                                acc=acc))

    print(f"\n{'='*60}")
    print("SEQUENTIAL SUMMARY (CIFAR-10 / ResNet-18)")
    print(f"{'='*60}")
    total_time   = sum(s["time"] for s in round_stats)
    overall_peak = np.mean([s["avg_peak"] for s in round_stats])
    print(f"Total wall time          : {total_time:.2f}s")
    print(f"Avg peak VRAM per client : {overall_peak:.0f}MB")
    print(f"Final accuracy           : {round_stats[-1]['acc']:.2f}%")
    print()
    print(f"  {'Round':<8} {'VRAM all 5 idle (MB)':<24} {'Avg peak/client (MB)':<24} {'Test Acc (%)':<12}")
    print(f"  {'-'*8} {'-'*24} {'-'*24} {'-'*12}")
    for s in round_stats:
        print(f"  {s['round']:<8} {s['vram_all_clients']:<24.0f} {s['avg_peak']:<24.0f} {s['acc']:<12.2f}")
    print()

if __name__ == "__main__":
    main()



# problems: Memory overflow for more number of clients. 
# client_models.append(local_model) keeps every trained model 
# on GPU until all clients finish — so at the end of a round 
# you're holding NUM_CLIENTS full ResNet-18 copies in VRAM 
# simultaneously. That's fine for 5 clients on a big GPU, 
# but it grows linearly with client count and blows up at scale.