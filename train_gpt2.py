import os
import math
import time
import inspect
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from hellaswag import render_example, iterate_examples, get_most_likely_row

### Codebook Usage 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_codebook_usage(codebook_indices, num_elements):
    # Calculate Codebook Usage
    codebook_usage = {}
    for i in range(codebook_indices.shape[-1]):
        indices = codebook_indices[...,i].view(-1)
        codebook_usage[i] = torch.bincount(indices, minlength=num_elements)/ indices.shape[0]

    return codebook_usage


def plot_codebook_usage(codebook_usage, num_elements, ddp=False, dist=None):
    # Plot Codebook Usage
    fig, axes = plt.subplots(len(codebook_usage), 1, figsize=(12, 4*len(codebook_usage)))
    if len(codebook_usage) == 1:
        axes = [axes]

    for i, usage in codebook_usage.items():
        usage_array = usage.detach().cpu().numpy()

        sns.barplot(x=np.arange(num_elements), y=usage_array, ax=axes[i])
        axes[i].set_title(f'Codebook Usage for Block {i+1}')
        axes[i].set_xlabel('Codebook Index')
        axes[i].set_ylabel('Usage Count')

    plt.tight_layout()
    return fig

dictloss2str = lambda loss_dict: " | ".join([f"{key}: {value:.4f}" for key, value in loss_dict.items()])

# -----------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model_VQVAE import GPT, GPTConfig
from dataset import DataLoaderLite

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 32 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
model_config = GPTConfig(vocab_size=50304)

# init WanB
if master_process:
    import wandb
    wandb.init(project="gpt2-training", config={
        "total_batch_size": total_batch_size,
        "micro_batch_size": B,
        "sequence_length": T,
        "max_lr": max_lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "max_steps": max_steps,
        "model": asdict(model_config),
        "dataset": DataLoaderLite.name
    })

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train", print_log=master_process)
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val", print_log=master_process)

torch.set_float32_matmul_precision('high')

# create model
model = GPT(model_config)
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = {}
            val_loss_steps = 20
            codebook_usage = {}
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, vq_indices, loss = model(x, y)
                    
                micro_codebook_usage = get_codebook_usage(vq_indices, model_config.cb_num_elements)
                for block in micro_codebook_usage:
                    codebook_usage[block] = codebook_usage.get(block,0) + micro_codebook_usage[block]/ val_loss_steps

                for loss_type in loss:
                    val_loss_accum[loss_type] = val_loss_accum.get(loss_type,0.0) + (loss[loss_type]/val_loss_steps).detach()

        if ddp:
            for l in val_loss_accum:
                dist.all_reduce(val_loss_accum[l], op=dist.ReduceOp.AVG)
            for block in codebook_usage:
                dist.all_reduce(codebook_usage[block], op=dist.ReduceOp.AVG)

        if master_process:
            val_loss_accum = {k:v.item() for k,v in val_loss_accum.items()}
            nonzero_codebook = [(codebook_usage[b]>0).sum().item() for b in codebook_usage]
            print(f"validation loss: {dictloss2str(val_loss_accum)}")
            print(f"Codebook Usage: {nonzero_codebook}")

            codebook_usage_plot = plot_codebook_usage(codebook_usage, model_config.cb_num_elements)
            wandb.log({"val_"+k:v for k,v in val_loss_accum.items()}, step=step)
            wandb.log({'val_codebook_usage': wandb.Image(codebook_usage_plot), 
                       "val_nonzero_codebook": sum(nonzero_codebook)/len(nonzero_codebook)}, step=step)
            
            with open(log_file, "a") as f:
                f.write(f"{step} val - {dictloss2str(val_loss_accum)}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        val_t0 = time.time()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits,_, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        val_dt = time.time() - val_t0
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            print(f"HellaSwag time: {val_dt:.2f}s")
            wandb.log({"hellaswag_accuracy": acc_norm}, step=step)
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits,_, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    losses_accum = {} 
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits,_, losses = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = losses["loss"] / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

        for l in losses:
            losses_accum[l] = losses_accum.get(l,0)+losses[l]/grad_accum_steps
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        for l in losses_accum:
            dist.all_reduce(losses_accum[l], op=dist.ReduceOp.AVG)
    

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f}| dist_temp: {model.cb_decoder.dist_temp.item():.3f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        print(dictloss2str(losses_accum))
        wandb.log({
            "train_loss": loss_accum.item(),
            "learning_rate": lr,
            "grad_norm": norm,
            "step_time": dt,
            "tokens_per_second": tokens_per_sec,
            "dist_temperature": model.cb_decoder.dist_temp.item()
        }, step=step) 
        wandb.log({k:v.item() for k,v in losses_accum.items()}, step=step)
        with open(log_file, "a") as f:
            f.write(f"{step} train - {dictloss2str(losses_accum)}\n")

if ddp:
    destroy_process_group()
if master_process:
    wandb.finish()
