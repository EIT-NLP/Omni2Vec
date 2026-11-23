"""Train Omni2Vec-3B or Omni2Vec-7B for one scheduled epoch."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import get_linear_schedule_with_warmup
import yaml

from src.loss import TaskAwareContrastiveLoss
from src.model.encoder import Encoder, model_arguments
from src.model.processor import item_from_pair


def batches(epoch, rank, world):
    plan = json.loads((epoch / 'source_schedule.json').read_text())
    sources = yaml.safe_load((epoch / 'dataset_config.yaml').read_text())
    size = plan['global_batch_size']
    if size % world:
        raise ValueError('The global batch size must be divisible by the world size')
    files, offsets = {}, Counter()
    local = size // world
    for source in plan['source_schedule']:
        if source not in files:
            digest = sources[source]['selection_sha256']
            files[source] = pq.ParquetFile(epoch / 'batches' / (digest + '.parquet'))
        table = files[source].read_row_group(offsets[source])
        offsets[source] += 1
        if table.num_rows != size:
            raise ValueError(f'Incomplete global batch for {source}')
        yield table.slice(rank * local, local).to_pylist()


def sides(batch, rng, device):
    queries, docs, ids, tasks, valid = [], [], [], [], []
    for row in batch:
        queries.append(item_from_pair(row, 'query', rng))
        docs.append(item_from_pair(row, 'pos', rng))
        ids.append(row['pair_id'])
        tasks.append(row['task_id'])
        valid.append(row['valid_record'])
    query_meta = [torch.tensor(x, device=device) for x in (ids, tasks, valid)]
    for row in batch:
        if row['valid_record'] and (row.get('neg_text') or row.get('neg_image')):
            docs.append(item_from_pair(row, 'neg', rng))
            ids.append(row['negative_pair_id'])
            tasks.append(row['task_id'])
            valid.append(True)
    doc_meta = [torch.tensor(x, device=device) for x in (ids, tasks, valid)]
    return queries, docs, query_meta, doc_meta


def backward(model, criterion, batch, rng, device, chunk_size):
    queries, docs, query_meta, doc_meta = sides(batch, rng, device)
    # Replay the same dropout masks when backpropagating cached embedding gradients.
    items = queries + docs
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    states, cached = [], []
    with torch.no_grad():
        for chunk in chunks:
            states.append((torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
            cached.append(model(chunk))
    leaf = torch.cat(cached).requires_grad_()
    q, d = leaf.split([len(queries), len(docs)])
    loss = criterion(q, d, *query_meta, *doc_meta)
    gradient, = torch.autograd.grad(loss, leaf)
    grads = gradient.split([len(chunk) for chunk in chunks])
    for i, (chunk, grad, (cpu, gpu)) in enumerate(zip(chunks, grads, states)):
        sync = model.no_sync() if hasattr(model, 'no_sync') and i + 1 < len(chunks) else nullcontext()
        with torch.random.fork_rng(devices=[device] if gpu is not None else []), sync:
            torch.set_rng_state(cpu)
            if gpu is not None:
                torch.cuda.set_rng_state(gpu, device)
            model(chunk).backward(grad)
    return loss.detach()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--epoch', type=Path, default=Path('data/epoch'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--chunk-size', type=int, default=2)
    parser.add_argument('--save-every', type=int, default=2000)
    args = parser.parse_args()
    if min(args.chunk_size, args.save_every) < 1:
        parser.error('chunk size and save interval must be positive')
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', 0))) if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        if any(args.output.iterdir()):
            raise FileExistsError(f'Choose an empty output directory: {args.output}')
    if world > 1:
        dist.barrier()
    torch.manual_seed(42)
    encoder = Encoder(args, device, training=True)
    model = DistributedDataParallel(encoder, device_ids=[device.index] if device.type == 'cuda' else None,
                                    broadcast_buffers=False) if world > 1 else encoder
    torch.manual_seed(42 + rank)
    rng = random.Random(42 + rank)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=.01)
    steps = json.loads((args.epoch / 'source_schedule.json').read_text())['optimizer_steps']
    scheduler = get_linear_schedule_with_warmup(opt, min(1000, max(0, steps - 1)), steps)
    loss_fn = TaskAwareContrastiveLoss(temperature=.02)
    for step, batch in enumerate(batches(args.epoch, rank, world), 1):
        opt.zero_grad(set_to_none=True)
        loss = backward(model, loss_fn, batch, rng, device, args.chunk_size)
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        opt.step()
        scheduler.step()
        if rank == 0 and (step == 1 or step % 10 == 0):
            print(f'step={step}/{steps} loss={loss.item():.4f}', flush=True)
        if rank == 0 and (step == steps or step % args.save_every == 0):
            directory = args.output / ('final' if step == steps else f'step-{step}')
            tmp = directory.with_suffix('.partial')
            encoder.thinker.model.save_pretrained(tmp)
            tmp.rename(directory)
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
