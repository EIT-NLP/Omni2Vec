"""Evaluate all twelve ordered text, image, video and audio retrieval directions."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from omniset.benchmark import load_benchmark, evaluate_rankings
from omniset.reader import InstructionModalityReader
from src.model.encoder import Encoder, encode, identity, model_arguments


def item(record, prompt=''):
    obj = {'text': prompt + record.get('text', '')}
    if record['modality'] == 'I':
        obj['images'] = [record['path']]
    elif record['modality'] == 'V':
        obj['video'] = record['path']
    elif record['modality'] == 'A':
        obj['audio'] = record['path']
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--data', type=Path, default=Path('data/omniset'))
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    bench = load_benchmark(args.data)
    encoder = Encoder(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    candidates, queries = bench['candidates'], bench['queries']
    docs = encode(encoder, (item(r) for r in candidates), args.batch_size)
    vectors = encode(encoder, (item(q['input'], q['instruction']) for q in queries), args.batch_size)
    if args.calibration:
        params = torch.load(args.calibration, weights_only=True, map_location='cpu')
        if params['encoder'] != identity(args):
            raise ValueError('The calibration and the retrieval encoder must match')
        reader = InstructionModalityReader(params['reader']['dim']).eval()
        reader.load_state_dict(params['reader']['state_dict'])
        prompts = list(dict.fromkeys(q['instruction'] for q in queries))
        inputs = encode(encoder, ({'text': p} for p in prompts), args.batch_size)
        with torch.no_grad():
            routes = dict(zip(prompts, reader(inputs).argmax(1).tolist()))
            shifts = params['directions'][[routes[q['instruction']] for q in queries]]
            vectors = F.normalize(vectors + params['alpha'] * shifts, dim=1)
    summary, _ = evaluate_rankings(bench, vectors.numpy(), docs.numpy(), [r['id'] for r in candidates])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
