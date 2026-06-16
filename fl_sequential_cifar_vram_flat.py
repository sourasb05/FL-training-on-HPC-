"""
Federated Averaging (FedAvg) - Sequential Training
Multiple clients, non-IID CIFAR-10 data, one GPU, Model: ResNet-18.
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
    model = torchvision.models.resnet18(weights=None) # no pretrained weights
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # adapt for 32x32 input
    model.maxpool = nn.Identity() # remove maxpool for CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes) # adapt final layer for 10 classes
    return model

# ── Non-IID data split (Dirichlet distribution) ───────────────────────────────
def non_iid_split(dataset, num_clients=5, alpha=0.5, seed=42): # alpha < 1.0 => more skewed, alpha > 1.0 => more balanced
    rng = np.random.default_rng(seed)   # for reproducibility
    labels = np.array(dataset.targets) # assumes dataset.targets is a list/array of class labels
    num_classes = len(np.unique(labels)) # e.g. 10 for CIFAR-10
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)] # list of arrays of indices for each class

    client_indices = [[] for _ in range(num_clients)] # list of lists to hold indices for each client
    for c_idx in class_indices: # shuffle indices of this class and split among clients
        proportions = rng.dirichlet(alpha * np.ones(num_clients)) # sample proportions for this class
        splits = (proportions * len(c_idx)).astype(int) # number of samples for each client from this class
        splits[-1] = len(c_idx) - splits[:-1].sum() # adjust last split to ensure all samples are assigned
        start = 0 # assign samples to clients according to splits
        for k, n in enumerate(splits): # assign n samples to client k
            client_indices[k].extend(c_idx[start:start + n].tolist()) # add these indices to client k
            start += n # move start pointer for next client

    return [Subset(dataset, idxs) for idxs in client_indices] # return list of Subset datasets for each client

# ── FedAvg aggregation (runs on CPU) ─────────────────────────────────────────
def fedavg(weight_list):  # weight_list is a list of state_dicts (all on CPU) from different clients
    """Average a list of state_dicts (CPU tensors). Returns a new state_dict."""  
    avg = copy.deepcopy(weight_list[0]) # start with a copy of the first client's weights
    for key in avg: # iterate over all parameter keys
        avg[key] = torch.stack([w[key].float() for w in weight_list]).mean(0) # stack the same parameter from all clients and take mean
    return avg # return the averaged state_dict
 
# ── Client local training ─────────────────────────────────────────────────────
def train_client(model, loader, device, epochs=2, lr=0.01, client_id=None): # client_id is just for logging purposes
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
        for bid, (x, y) in enumerate(loader): # iterate over batches
            batch_start = time.time() 
            x, y = x.to(device), y.to(device) # move batch to GPU
            opt.zero_grad() # zero gradients
            loss = criterion(model(x), y) # forward pass and compute loss
            loss.backward() # backward pass
            opt.step() # update parameters
            batch_time = time.time() - batch_start
            running_loss += loss.item() # accumulate loss for logging

            if bid == 0 or (bid + 1) % max(1, num_batches // 4) == 0 or bid + 1 == num_batches: # log at 25%, 50%, 75%, and 100% of batches
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
    weights = {k: v.cpu().clone() for k, v in model.state_dict().items()} # move each parameter to CPU and clone to ensure it's a separate copy; this way we avoid any GPU memory usage from the returned state_dict, and we don't rely on the caller to move it to CPU later, which could lead to accidental GPU memory usage if they forget. By doing this here, we ensure that once we delete the model and optimizer, there are no lingering GPU tensors.
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
        for x, y in loader: # iterate over test batches
            x, y = x.to(device), y.to(device) # move batch to GPU
            correct += (model(x).argmax(1) == y).sum().item() # count correct predictions
            total += y.size(0) # count total samples
    return 100.0 * correct / total

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    def _cuda_usable(): # check if CUDA is available and can allocate memory (handles cases where CUDA is present but not usable)
        if not torch.cuda.is_available(): # check if CUDA is available at all
            return False
        try:
            torch.zeros(1).cuda() # try to allocate a small tensor on GPU to confirm it's usable
            return True
        except Exception: # if allocation fails, CUDA is not usable (e.g. out of memory, driver issues)
            return False

    DEVICE = torch.device("cuda:0" if _cuda_usable() else "cpu") # use GPU if available and usable, otherwise fallback to CPU
    NUM_CLIENTS  = 5  
    NUM_ROUNDS   = 5
    LOCAL_EPOCHS = 5
    BATCH_SIZE   = 128

    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}  |  Rounds: {NUM_ROUNDS}  |  Dataset: CIFAR-10  |  "
          f"Model: ResNet-18  |  Mode: SEQUENTIAL (VRAM-flat)\n") 

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # data augmentation for training set
        transforms.RandomHorizontalFlip(), # data augmentation for training set
        transforms.ToTensor(), # convert PIL image to tensor and scale pixel values to [0, 1]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # normalization for CIFAR-10 training set
    ]) # standard data augmentation for CIFAR-10 training set
    transform_test = transforms.Compose([
        transforms.ToTensor(), # convert PIL image to tensor and scale pixel values to [0, 1]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # same normalization as training set, but no augmentation for test set
    ]) # normalization for CIFAR-10 test set (no augmentation)
 
    print("Loading CIFAR-10 (downloading if needed ~170MB)...", flush=True)
    t_data = time.time()
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True,
                                             download=True, transform=transform_train) # load CIFAR-10 dataset with specified transforms; will download if not already present in ./data
    test_set  = torchvision.datasets.CIFAR10(root="./data", train=False,
                                             download=True, transform=transform_test) # load CIFAR-10 dataset with specified transforms; will download if not already present in ./data
    print(f"Dataset ready in {time.time()-t_data:.1f}s  "
          f"(train={len(train_set)}, test={len(test_set)})", flush=True) # print dataset loading time and number of samples in train and test sets

    client_datasets = non_iid_split(train_set, NUM_CLIENTS)
    client_loaders  = [DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True) # create DataLoader for each client's dataset with specified batch size and shuffling; num_workers=2 for parallel data loading, pin_memory=True for faster GPU transfers
                       for ds in client_datasets] # create a DataLoader for each client's dataset with specified batch size and shuffling; num_workers=2 for parallel data loading, pin_memory=True for faster GPU transfers
    test_loader     = DataLoader(test_set, batch_size=256, shuffle=False,
                                 num_workers=2, pin_memory=True) # create DataLoader for test set with larger batch size (256) since we don't need to backpropagate, and no shuffling; num_workers=2 for parallel data loading, pin_memory=True for faster GPU transfers

    print("Class distribution per client:") 
    all_labels = np.array(train_set.targets) # get all labels from the training set as a numpy array for easy indexing
    for i, ds in enumerate(client_datasets): # print class distribution for this client's dataset by counting occurrences of each class label among the indices assigned to this client
        counts = np.bincount(all_labels[ds.indices], minlength=10) # count occurrences of each class label (0-9) in this client's dataset; minlength=10 ensures we get counts for all classes even if some are zero
        print(f"  Client {i}: {counts}  (total={len(ds.indices)})") # print the class distribution for this client along with the total number of samples assigned to this client
    print() # print a newline for better readability

    # Global model lives on CPU between rounds; moved to GPU only for eval.
    global_model = build_resnet18() # initialize the global model (ResNet-18 adapted for CIFAR-10) on CPU; it will be moved to GPU during evaluation and back to CPU after each round
    print(f"Model parameters: {sum(p.numel() for p in global_model.parameters()):,}\n") # print the total number of parameters in the model for reference

    round_stats = [] # list to hold statistics for each round, such as time taken, memory usage, and accuracy, for later summary and analysis

    for rnd in range(1, NUM_ROUNDS + 1): # loop over federated learning rounds
        print(f"{'='*60}")
        print(f"Round {rnd}/{NUM_ROUNDS}")
        round_start = time.time()

        local_weights = []   # list of CPU state_dicts — no GPU memory held here
        client_peaks  = []

        for cid in range(NUM_CLIENTS):
            t0 = time.time()
            # deepcopy on CPU, then train_client moves it to GPU
            local_model = copy.deepcopy(global_model) # create a local copy of the global model for this client; this is done on CPU to avoid unnecessary GPU memory usage, and train_client will handle moving it to GPU for training
            print(f"  [client {cid}] starting training  "
                  f"({len(client_loaders[cid])} batches x {LOCAL_EPOCHS} epochs)", flush=True)

            if DEVICE.type == "cuda":
                torch.cuda.reset_peak_memory_stats(None) # reset peak memory stats at the start of this client's training so we can measure the peak VRAM usage for this client alone; this does not free any memory, it just resets the counters that track peak usage
 
            # Returns CPU state_dict; GPU is freed inside train_client
            weights = train_client(local_model, client_loaders[cid], DEVICE,
                                   LOCAL_EPOCHS, client_id=cid)

            if DEVICE.type == "cuda":
                mem_alloc = torch.cuda.memory_allocated(None) / 1024**2. # current VRAM allocated after training and offloading this client. converting to MB for easier reading. This should be close to zero if train_client successfully freed GPU memory.
                mem_peak  = torch.cuda.max_memory_allocated(None) / 1024**2 # peak VRAM allocated during this client's training (resets at the start of each client) converting to MB for easier reading. This gives us an idea of how much VRAM this client's training consumed at its peak, which is useful for understanding the memory requirements of the model and training process.
            else:
                mem_alloc = mem_peak = 0.0

            local_weights.append(weights)
            client_peaks.append(mem_peak)
            elapsed = time.time() - t0
            print(f"  Client {cid} trained in {elapsed:.2f}s  |  "
                  f"Mem after offload: alloc={mem_alloc:.0f}MB  peak={mem_peak:.0f}MB",
                  flush=True)

        # All N client models already offloaded to CPU right here — GPU is free
        vram_after_offload = torch.cuda.memory_allocated(None) / 1024**2 if DEVICE.type == "cuda" else 0.0 # check VRAM usage after all clients have offloaded their models to CPU; this should be close to zero if all GPU memory was successfully freed, confirming that we have a flat VRAM baseline before aggregation
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
            torch.cuda.synchronize() # ensure all GPU operations are finished before measuring memory and printing results
            torch.cuda.empty_cache() # free any cached memory that can be released back to the system, ensuring that our VRAM measurements reflect the actual usage after cleanup

        if DEVICE.type == "cuda":
            round_mem_alloc = torch.cuda.memory_allocated(None) / 1024**2
        else:
            round_mem_alloc = 0.0

        print(f"  >> Round {rnd} done in {round_time:.2f}s  |  "
              f"Mem: alloc={round_mem_alloc:.0f}MB  |  "
              f"Test acc: {acc:.2f}%")
        round_stats.append(dict(round=rnd, time=round_time, 
                                mem_alloc=round_mem_alloc, # VRAM allocated at the end of the round (should be low if cleanup was successful)
                                vram_after_offload=vram_after_offload, # VRAM after all clients offloaded to CPU (should be close to zero if all GPU memory was successfully freed, confirming that we have a flat VRAM baseline before aggregation)
                                avg_peak=np.mean(client_peaks), # average of the peak VRAM usage across all clients for this round, which gives us an idea of the typical memory requirements during client training
                                acc=acc)) # store statistics for this round, including time taken, memory usage, and accuracy, for later summary and analysis

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
