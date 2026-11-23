"""Fit the instruction reader and shrinkage-LDA directions on training candidates."""
import argparse
import itertools
import json
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from omniset.benchmark import MODS, NAMES, instruction
from omniset.reader import InstructionModalityReader
from omniset.directions import fit_directions
from src.model.encoder import Encoder, encode, identity, model_arguments
from src.model.processor import item_from_pair
from train import batches

TRAIN = ('Search for {target} with the same subject as this {source}.',
         'The input is {source}. Retrieve a matching result in {target} format.',
         'I need {target} about the content presented in this {source}.',
         'Use this {source} to locate a corresponding {target} result.',
         'A matching {target} is requested for the supplied {source}.',
         'Among the available results, choose {target} related to this {source}.')
DEV = ('Given {source} content, the requested output format is {target}.',
       'Return a semantically related {target} result for this {source} input.')


def reader_data(encoder, templates, canonical=False):
    items, labels = [], []
    for source, target in itertools.permutations(MODS, 2):
        if {source, target} == {'I', 'V'}:
            continue
        texts = [t.format(source=NAMES[source], target=NAMES[target]) for t in templates]
        if canonical:
            texts.append(instruction(source, target))
        for text in texts:
            items.append({'text': text})
            labels.append(MODS.index(target))
    return encode(encoder, items).clone(), torch.tensor(labels)


def fit_reader(x, y, xd, yd):
    torch.manual_seed(42)
    reader = InstructionModalityReader(x.shape[1])
    optimizer = torch.optim.AdamW(reader.parameters(), lr=.001, weight_decay=.01)
    best_loss, best = float('inf'), None
    weight = 1. / torch.bincount(y, minlength=4)
    for step in range(1, 301):
        optimizer.zero_grad()
        F.cross_entropy(reader(x), y, weight=weight).backward()
        optimizer.step()
        if step % 25 == 0:
            with torch.no_grad():
                loss = F.cross_entropy(reader(xd), yd).item()
            if loss < best_loss:
                best_loss, best = loss, reader.payload()
    return best


def candidate_sample(encoder, epoch, per_modality):
    rng = random.Random(42)
    seen = [set() for _ in MODS]
    chunks, labels = [], []
    for batch in batches(epoch, 0, 1):
        items = []
        for row in batch:
            if not row['valid_record']:
                continue
            pair = row['pair_id']
            media = row.get('pos_image') or {}
            paths = media.get('paths') or []
            video = (len(paths) > 1 or (paths and str(paths[0]).startswith('local-video://')))
            mod = 3 if row.get('pos_audio') else 2 if video and not row['multi_image_as_images'] else 1 if media else 0
            if len(seen[mod]) < per_modality and pair not in seen[mod]:
                seen[mod].add(pair)
                items.append(item_from_pair(row, 'pos', rng))
                labels.append(mod)
        if items:
            chunks.append(encode(encoder, items))
        if all(len(s) == per_modality for s in seen):
            return torch.cat(chunks), labels
    raise ValueError(f'Insufficient distinct training candidates per modality: {[len(s) for s in seen]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--epoch', type=Path, default=Path('data/epoch'))
    parser.add_argument('--per-modality', type=int, default=512)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.per_modality < 2:
        parser.error('LDA requires at least two distinct candidates per modality')
    encoder = Encoder(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    x, labels = candidate_sample(encoder, args.epoch, args.per_modality)
    directions, *_ = fit_directions(x.numpy(), labels, shrinkage=.1)
    reader = fit_reader(*reader_data(encoder, TRAIN, True), *reader_data(encoder, DEV))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'encoder': identity(args), 'directions': torch.tensor(directions).float(),
                'reader': reader, 'alpha': .7}, args.output)


if __name__ == '__main__':
    main()
