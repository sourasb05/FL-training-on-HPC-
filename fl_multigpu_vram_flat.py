"""
FedAvg - Multi-GPU + VRAM-flat  (N clients, M GPUs, N >= M)
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
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # modify the first conv layer to fit CIFAR-10's 32x32 images (original ResNet-18 is designed for ImageNet's 224x224); this change allows us to use ResNet-18 on CIFAR-10 without resizing the images, which is common practice for this dataset and helps keep the model size and training time manageable for our federated learning demo
    model.maxpool = nn.Identity() # remove the maxpool layer to preserve spatial dimensions after the first conv; this is a common modification when using ResNet architectures on smaller images like CIFAR-10, as the original maxpool would reduce the feature map size too much and hurt performance; by replacing it with Identity, we allow the model to retain more spatial information early on, which can improve accuracy on CIFAR-10
    model.fc = nn.Linear(model.fc.in_features, num_classes) # replace the final fully connected layer to output the correct number of classes for CIFAR-10 (10 classes); this is necessary because the original ResNet-18 is designed for ImageNet with 1000 classes, and we need to adapt it for our specific dataset in this federated learning setup
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
def state_to_bytes(state_dict): # serialize a state_dict to bytes, ensuring all tensors are moved to CPU first to avoid GPU memory usage during serialization; this is important for our VRAM-flat approach where we want to minimize GPU memory usage and avoid accidental GPU allocations from the returned state_dict
    buf = io.BytesIO() # create an in-memory bytes buffer to hold the serialized state_dict; this avoids writing to disk and allows us to easily get the byte content after saving
    torch.save({k: v.cpu() for k, v in state_dict.items()}, buf) # save the state_dict to the buffer, moving each tensor to CPU to ensure that the resulting bytes do not contain any GPU tensors, which could lead to GPU memory usage when deserialized if not handled carefully
    return buf.getvalue() # return the byte content of the buffer, which is the serialized state_dict; this can be sent to GPU workers without worrying about GPU memory usage from the serialization process

def bytes_to_state(b):
    return torch.load(io.BytesIO(b), weights_only=True) # deserialize bytes back to a state_dict; since we saved with tensors on CPU, this will load the tensors onto CPU by default, which is what we want for our VRAM-flat approach where the global model and aggregation run on CPU

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
        global_bytes, client_list = task_queue.get() # receive global model and client assignments for this round from server; this will block until the server puts the data in the queue, ensuring synchronization between server and GPU workers at the start of each round

        round_results = [] 
        for cid, indices in client_list:
            print(f"  [GPU {gpu_id}] starting client {cid}  "
                  f"({len(indices)} samples)", flush=True)
            cpu_weights, elapsed, mem_peak = train_one_client(
                global_bytes, indices, device,
                batch_size, local_epochs, data_path, cid
            ) # train the assigned client on this GPU and get the updated weights, elapsed time, and peak memory usage; this will run sequentially for all clients assigned to this GPU for the current round, with aggressive VRAM teardown after each client to minimize GPU memory usage and ensure a flat baseline for nvidia-smi monitoring
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
                             num_workers=2, pin_memory=True) # load the test set and create a DataLoader for evaluation; we use a batch size equal to the entire test set to evaluate in one go, which is feasible for CIFAR-10 and allows us to get the accuracy without needing to loop over multiple batches; this will be used by the server after each round of training to evaluate the global model's performance on the test set

    global_model = build_resnet18() # initialize the global model on CPU; this model will be updated each round with the aggregated client weights and will be evaluated on GPU 0 before being moved back to CPU for the next round; keeping the global model on CPU throughout allows us to manage GPU memory more effectively with our VRAM-flat approach, as the GPU workers will only load the model for their assigned clients and then offload it back to CPU after training
    print(f"Server: {sum(p.numel() for p in global_model.parameters()):,} parameters") # print the total number of parameters in the global model for informational purposes; this helps give a sense of the model size and complexity, which can be relevant for understanding the training time and memory usage on the GPU workers during the federated learning rounds
    print(f"Server: {num_clients} clients distributed across {num_gpus} GPUs " # print the number of clients and GPUs, and how they will be distributed; this sets the stage for the multi-GPU federated learning setup and helps clarify how many clients each GPU worker will be responsible for training sequentially in each round, which is important for understanding the workload and memory management strategy of the VRAM-flat approach
          f"({num_clients // num_gpus}–{(num_clients + num_gpus - 1) // num_gpus} clients/GPU)\n")

    gpu_client_map = [[] for _ in range(num_gpus)] # create a mapping of GPU IDs to the list of client IDs that will be assigned to each GPU; this will be used to distribute the clients in a round-robin fashion across the available GPUs for each round of training, ensuring that the workload is balanced and that we can effectively utilize all GPUs while managing memory with the VRAM-flat approach
    for cid in range(num_clients):
        gpu_client_map[cid % num_gpus].append(cid) # assign clients to GPUs in a round-robin manner; this ensures that the clients are distributed as evenly as possible across the available GPUs, which helps balance the training workload and allows us to take advantage of multiple GPUs while keeping the memory usage manageable with our VRAM-flat strategy

    print("Client → GPU assignment:")
    for gid, cids in enumerate(gpu_client_map): 
        print(f"  GPU {gid}: clients {cids}") # print the client-to-GPU assignment for informational purposes; this helps verify that the clients have been distributed correctly across the GPUs and gives insight into which clients will be trained on which GPUs during each round, which is useful for understanding the training dynamics and for debugging if needed
    print()

    task_queues  = [mp.Queue() for _ in range(num_gpus)] # create a list of multiprocessing queues, one for each GPU worker, to send tasks (global model and client assignments) from the server to the GPU workers; this allows for communication between the main process (server) and the worker processes (GPUs) in a way that is safe for multiprocessing and helps coordinate the training rounds effectively
    result_queue = mp.Queue() # create a multiprocessing queue for GPU workers to send results (updated client weights, elapsed time, and memory usage) back to the server after training; this allows the GPU workers to communicate their results back to the main process in a thread-safe manner and enables the server to aggregate the results from all GPUs after each round of training

    processes = [] # create a list to hold the GPU worker processes; this will allow us to keep track of the processes we spawn for each GPU and ensure that we can join them properly at the end of the training rounds to clean up resources
    for gid in range(num_gpus): # spawn a separate process for each GPU worker, passing the GPU ID, assigned clients, number of rounds, local epochs, batch size, data path, task queue, and result queue as arguments; this will allow each GPU worker to run independently and manage its own GPU memory while training the assigned clients sequentially for each round, following the VRAM-flat approach to minimize memory usage and ensure a flat baseline for monitoring with nvidia-smi
        p = mp.Process(
            target=gpu_worker,
            args=(gid, gpu_client_map[gid], num_rounds, local_epochs,
                  batch_size, data_path, task_queues[gid], result_queue)
        ) # create a new process for the GPU worker with the specified arguments; this will start the training loop for that GPU worker, which will wait for tasks from the server, train the assigned clients, and send results back to the server for each round
        p.start() # start the GPU worker process; this will run the gpu_worker function in a separate process, allowing it to manage its own GPU memory and perform training independently of the main server process, which is essential for our multi-GPU federated learning setup with the VRAM-flat approach
        processes.append(p) # add the process to the list of processes so we can keep track of it and join it later; this is important for ensuring that we can clean up all worker processes properly at the end of the training rounds

    round_stats = []

    def log_round_marker(tag, rnd):
        if round_log_file:
            ts = datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]
            with open(round_log_file, "a") as f: 
                f.write(f"{ts}, ROUND_{rnd}_{tag}\n") # helper function to log the start and end of each round to a CSV file with timestamps; this can be useful for analyzing the timing of each round and correlating it with GPU memory usage from nvidia-smi logs, especially when using the VRAM-flat approach where we want to see the memory usage before and after each round clearly in the logs

    for rnd in range(1, num_rounds + 1): # main training loop for the specified number of rounds; in each round, the server will distribute the global model and client assignments to the GPU workers, wait for their results, perform FedAvg aggregation on CPU, evaluate the global model on GPU 0, and log the results; this loop coordinates the entire federated learning process across multiple GPUs while managing memory with the VRAM-flat strategy
        print(f"{'='*60}", flush=True)
        print(f"Round {rnd}/{num_rounds}", flush=True)
        round_start = time.time()
        log_round_marker("START", rnd)

        global_bytes = state_to_bytes(global_model.state_dict()) # serialize the global model's state_dict to bytes so it can be sent to the GPU workers without worrying about GPU memory usage from the serialization process; this is important for our VRAM-flat approach where we want to minimize GPU memory usage and ensure that the workers can load the global model onto their GPUs without unintended GPU allocations from the server's side
        for gid in range(num_gpus):
            client_list = [(cid, client_indices_list[cid])
                           for cid in gpu_client_map[gid]]
            task_queues[gid].put((global_bytes, client_list)) # send the global model and the list of assigned clients (with their data indices) to each GPU worker for this round; this will trigger the GPU workers to start training their assigned clients sequentially with aggressive VRAM teardown, and they will return the updated weights and stats back to the server through the result queue when done
        print(f"  [server] dispatched {num_clients} clients to {num_gpus} GPUs", flush=True) # print a log message indicating that the server has dispatched the clients to the GPU workers for this round; this helps track the progress of the training rounds and confirms that the tasks have been sent to the workers

        all_weights  = {}
        all_peaks    = []
        gpus_done    = 0
        while gpus_done < num_gpus: # wait for results from all GPU workers; this loop will block until it receives results from each GPU worker through the result queue, ensuring that the server waits for all clients to finish training on their respective GPUs before proceeding with aggregation and evaluation for the round
            gid, round_results = result_queue.get()
            for cid, elapsed, mem_peak, state_bytes in round_results:
                all_weights[cid] = bytes_to_state(state_bytes) # store the updated weights for each client in a dictionary keyed by client ID; this will be used later for FedAvg aggregation on CPU, and since we serialized the state_dicts to bytes, we need to deserialize them back to state_dict format before storing
                all_peaks.append(mem_peak) # keep track of the peak memory usage for each client trained on the GPU workers; this will allow us to calculate the average and maximum peak VRAM usage across all clients for this round, which is important for analyzing the memory efficiency of our VRAM-flat approach and understanding the GPU memory dynamics during training
                print(f"  [server] client {cid} (GPU {gid}): "
                      f"{elapsed:.2f}s  peak={mem_peak:.0f}MB", flush=True)
            gpus_done += 1 # increment the count of completed GPU workers; once all GPU workers have sent their results, the server can proceed with the FedAvg aggregation and evaluation for this round

        print(f"  [server] FedAvg over {num_clients} client weights (CPU)...", flush=True)
        ordered = [all_weights[cid] for cid in range(num_clients)]
        global_model.load_state_dict(fedavg(ordered)) # perform FedAvg aggregation on CPU using the collected client weights; this will update the global model's state_dict with the averaged weights from all clients, and since the global model is kept on CPU, this operation will be memory efficient and will not involve GPU memory usage, which is a key aspect of our VRAM-flat approach  

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
                                avg_peak=avg_peak, max_peak=max_peak)) # log the results for this round, including the round number, total time taken, test accuracy, average peak VRAM usage across clients, and maximum peak VRAM usage; this will allow us to analyze the performance and memory efficiency of our multi-GPU federated learning setup with the VRAM-flat approach after all rounds are completed

    for p in processes:
        p.join() # join all GPU worker processes to ensure they have finished before the server process exits; this is important for cleaning up resources properly and ensuring that all training and communication with the GPU workers has completed before the main process terminates

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

    job_id    = os.environ.get("SLURM_JOB_ID", "local") # use SLURM_JOB_ID if available, otherwise "local"
    round_log = f"round_log_{job_id}.csv"

    num_gpus = torch.cuda.device_count() # detect number of available GPUs; this will determine how many parallel GPU worker processes we spawn and how clients are distributed across GPUs each round
    print(f"GPUs available : {num_gpus}") # print the number of GPUs detected for informational purposes; this helps confirm that the script is correctly recognizing the GPU resources in the environment and will be useful for debugging if the expected number of GPUs is not found
    if num_gpus < 1:
        raise SystemExit("Need at least 1 GPU.") #

    print(f"Clients        : {NUM_CLIENTS}  ({NUM_CLIENTS // num_gpus}–"
          f"{(NUM_CLIENTS + num_gpus - 1) // num_gpus} per GPU, sequential + VRAM-flat)")
    print(f"Rounds         : {NUM_ROUNDS}")
    print(f"Local epochs   : {LOCAL_EPOCHS} per round")
    print(f"Dataset        : CIFAR-10  |  Model: ResNet-18")
    print(f"Mode           : Multi-GPU + VRAM-flat\n")

    print("Loading CIFAR-10...", flush=True)
    train_set = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                             download=True, transform=None)
    torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                 download=True, transform=None)

    client_indices_list = non_iid_split(train_set, NUM_CLIENTS)

    print("Class distribution per client:")
    all_labels = np.array(train_set.targets)
    for i, idxs in enumerate(client_indices_list):
        counts = np.bincount(all_labels[idxs], minlength=10) # print the class distribution for each client after the non-IID split; this helps verify that the data has been split in a non-IID manner and gives insight into the diversity of data each client will be training on, which is important for understanding the federated learning dynamics and the challenges of aggregating heterogeneous client updates on the server
        print(f"  Client {i:>3}: {counts}  (total={len(idxs)})")
    print()

    mp.set_start_method("spawn", force=True) # set multiprocessing start method to "spawn" for better compatibility with CUDA and to avoid issues with forked processes inheriting GPU memory; this is important for our multi-GPU setup where each worker process will manage its own GPU memory and we want to ensure a clean state for each worker without unintended sharing of GPU resources that can happen with the default "fork" method on Unix systems
    server(NUM_CLIENTS, num_gpus, NUM_ROUNDS, LOCAL_EPOCHS,
           BATCH_SIZE, DATA_PATH, client_indices_list,
           round_log_file=round_log) # start the server (main process) which will coordinate the federated learning rounds, distribute tasks to GPU workers, aggregate results, and evaluate the global model; this function will run the entire federated learning simulation and print out detailed logs and a final summary of results

if __name__ == "__main__":
    main()
