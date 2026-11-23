"""Multi-positive InfoNCE with task masks and differentiable cross-rank candidates."""
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_gather


class TaskAwareContrastiveLoss(nn.Module):
    def __init__(self, temperature=.02):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, d, pair_ids, task_ids, valid, doc_ids, doc_tasks, doc_valid):
        distributed = dist.is_initialized() and dist.get_world_size() > 1
        world = dist.get_world_size() if distributed else 1
        if distributed:
            # Explicit negatives can give ranks different candidate counts.
            count = torch.tensor(len(d), device=d.device)
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            pad = int(count) - len(d)
            d = torch.cat([d, d.new_zeros(pad, d.shape[1])])
            metadata = torch.stack([doc_ids, doc_tasks, doc_valid.long()], dim=1)
            metadata = torch.cat([metadata, metadata.new_zeros(pad, 3)])
            blocks = [torch.empty_like(metadata) for _ in range(world)]
            dist.all_gather(blocks, metadata)
            doc_ids, doc_tasks, doc_valid = torch.cat(blocks).T
            d = torch.cat(all_gather(d.contiguous()))
        logits = q[valid].float() @ d.float().T / self.temperature
        visible = doc_valid.bool().unsqueeze(0) & task_ids[valid, None].eq(doc_tasks[None, :])
        positives = visible & pair_ids[valid, None].eq(doc_ids[None, :])
        if not positives.any(dim=1).all():
            raise ValueError('A valid query has no positive candidate in its task')
        total = (logits.masked_fill(~visible, -torch.inf).logsumexp(1) -
                 logits.masked_fill(~positives, -torch.inf).logsumexp(1)).sum()
        valid_count = valid.sum()
        if distributed:
            dist.all_reduce(valid_count)
        return total * world / valid_count.clamp_min(1)
