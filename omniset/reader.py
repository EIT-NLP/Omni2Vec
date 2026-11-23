from __future__ import annotations
import torch
import torch.nn as nn

class InstructionModalityReader(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.classifier = nn.Linear(dim, 4)

    def forward(self, instruction_embedding):
        with torch.autocast(device_type=instruction_embedding.device.type, enabled=False):
            return self.classifier(self.norm(instruction_embedding.float()))

    def payload(self):
        return {'dim': self.dim,
                'state_dict': {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}}
