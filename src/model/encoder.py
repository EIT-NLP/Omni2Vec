from collections import defaultdict
from pathlib import Path
import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration
from transformers import Qwen2_5OmniThinkerForConditionalGeneration
from peft import LoraConfig, PeftModel, get_peft_model

from src.model.processor import prepare_inputs

MODELS = {size: f'Qwen/Qwen2.5-Omni-{size.upper()}' for size in ('3b', '7b')}


def model_arguments(parser):
    parser.add_argument('--size', choices=MODELS, required=True)
    parser.add_argument('--base', help='Local Qwen model directory or Hugging Face model ID')
    parser.add_argument('--adapter', type=Path, help='Omni2Vec LoRA checkpoint')


class Encoder(nn.Module):
    def __init__(self, args, device, training=False):
        super().__init__()
        base = args.base or MODELS[args.size]
        dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32
        self.processor = AutoProcessor.from_pretrained(base)
        self.processor.tokenizer.padding_side = 'right'
        config_path = Path(base) / 'config.json'
        thinker_only = config_path.exists() and json.loads(config_path.read_text())['model_type'] == 'qwen2_5_omni_thinker'
        opts = {'torch_dtype': dtype, 'attn_implementation': 'sdpa'}
        if thinker_only:
            self.thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(base, **opts)
        else:
            config = Qwen2_5OmniConfig.from_pretrained(base)
            config.enable_audio_output = False
            self.thinker = Qwen2_5OmniForConditionalGeneration.from_pretrained(base, config=config, **opts).thinker
        self.thinker.requires_grad_(False)
        from src.model.optional_backends import disable_broken_peft_bitsandbytes
        disable_broken_peft_bitsandbytes()
        if args.adapter:
            self.thinker.model = PeftModel.from_pretrained(self.thinker.model, args.adapter, is_trainable=training)
        elif training:
            config = LoraConfig(r=16, lora_alpha=64, lora_dropout=.1, init_lora_weights='gaussian',
                                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
            self.thinker.model = get_peft_model(self.thinker.model, config)
        if training:
            self.thinker.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.to(device).train(training)

    def forward(self, items):
        groups = defaultdict(list)
        for index, item in enumerate(items):
            signature = tuple(bool(item.get(k)) for k in ('images', 'video', 'audio'))
            groups[signature].append(index)
        result = [None] * len(items)
        for indices in groups.values():
            inputs = prepare_inputs(self.processor, [items[i] for i in indices])
            inputs = inputs.to(self.thinker.device)
            if 'feature_attention_mask' in inputs:
                time = min(inputs.input_features.shape[-1], inputs.feature_attention_mask.shape[-1])
                inputs['input_features'] = inputs.input_features[..., :time]
                inputs['feature_attention_mask'] = inputs.feature_attention_mask[..., :time]
            outputs = self.thinker(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = outputs.hidden_states[-1].float()
            mask = inputs.attention_mask.unsqueeze(-1)
            vectors = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=1)
            for index, vector in zip(indices, vectors.unbind()):
                result[index] = vector
        return torch.stack(result)


@torch.inference_mode()
def encode(model, items, batch_size=4):
    from itertools import islice
    it = iter(items)
    chunks = []
    while batch := list(islice(it, batch_size)):
        chunks.append(model(batch).cpu())
    if not chunks:
        raise ValueError('Empty embedding input')
    return torch.cat(chunks)


def identity(args):
    adapter = None
    if args.adapter:
        with (args.adapter / 'adapter_model.safetensors').open('rb') as f:
            adapter = hashlib.file_digest(f, 'sha256').hexdigest()
    base = args.base or MODELS[args.size]
    config = Path(base) / 'config.json'
    base_id = hashlib.sha256(config.read_bytes()).hexdigest() if config.is_file() else str(base)
    return {'size': args.size, 'base': base_id, 'adapter_sha256': adapter}
