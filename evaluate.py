"""Rank a retrieval task with Omni2Vec and evaluate binary and graded relevance."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

import torch

from src.model.encoder import Encoder, encode, model_arguments
from src.data.dataset.visrag_dataset import query_source2prompt, target_source2prompt


def rows(root, name):
    path = root / (name + '.jsonl')
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        import lance
        for batch in lance.dataset(root / (name + '.lance')).scanner(batch_size=32).to_batches():
            yield from batch.to_pylist()


def item(row, root, family, side):
    text = row.get('text') or ''
    frames = row.get('images') or []
    if family == 'visdoc':
        if root.name.startswith('VisRAG_'):
            source = root.name.removeprefix('VisRAG_')
            text = query_source2prompt.get(source, '') + text if side == 'queries' else target_source2prompt.get(source, '')
        else:
            text = ('Retrieve documents that best answer the following query: ' + text if side == 'queries'
                    else 'Generate an embedding for the following document, in the format of images: ')
    obj = {'text': text}
    def media(v):
        return root / v if isinstance(v, str) else v
    if frames:
        obj['video' if family == 'video' else 'images'] = [media(v) for v in frames]
    if row.get('video'):
        v = row['video']
        obj['video'] = [media(f) for f in v] if isinstance(v, list) else media(v)
    if row.get('audio'):
        audio = row['audio']
        if isinstance(audio, dict) and audio.get('path'):
            audio = dict(audio, path=str(root / audio['path']))
        obj['audio'] = media(audio)
    return obj


def judgments(root):
    rels, pools = defaultdict(dict), {}
    for row in rows(root, 'qrels'):
        q = str(row['query_id'])
        ids = list(map(str, row['candidate_ids']))
        scores = row['candidate_scores']
        if len(scores) != len(ids) or not all(math.isfinite(float(s)) for s in scores):
            raise ValueError(f'Invalid relevance scores: {q}')
        if row.get('mode') == 'exhaustive':
            pools.setdefault(q, {}).update(dict.fromkeys(ids))
        for cid, score in zip(ids, scores):
            if score > 0:
                rels[q][cid] = max(float(score), rels[q].get(cid, 0.))
    return rels, pools


def metrics(ranked, gold):
    grades = [gold.get(cid, 0.) for cid in ranked]
    def dcg(values):
        return sum(v / math.log2(i + 2) for i, v in enumerate(values[:5]))
    return [float(grades[0] > 0), dcg(grades) / dcg(sorted(gold.values(), reverse=True)),
            sum(g > 0 for g in grades[:10]) / len(gold)]


def evaluate(model, root, family, batch_size):
    rels, pools = judgments(root)
    candidate_ids, vectors = [], []
    candidates = iter(rows(root, 'candidates'))
    while batch := list(itertools.islice(candidates, batch_size)):
        candidate_ids.extend(str(r['id']) for r in batch)
        vectors.append(encode(model, (item(r, root, family, 'candidates') for r in batch), batch_size))
    index = {c: i for i, c in enumerate(candidate_ids)}
    if not vectors or len(index) != len(candidate_ids):
        raise ValueError('Expected unique, nonempty corpus')
    candidates = torch.cat(vectors)
    queries = (r for r in rows(root, 'queries') if str(r['id']) in rels)
    seen, sums = set(), [0., 0., 0.]
    while batch := list(itertools.islice(queries, batch_size)):
        qvec = encode(model, (item(r, root, family, 'queries') for r in batch), batch_size)
        for r, v in zip(batch, qvec):
            qid = str(r['id'])
            if qid in seen:
                raise ValueError(f'Duplicate query: {qid}')
            seen.add(qid)
            pool = list(pools[qid]) if qid in pools else candidate_ids
            if not rels[qid].keys() <= (pools[qid].keys() if qid in pools else index.keys()):
                raise ValueError(f'Positive missing from retrieval pool: {qid}')
            corpus = candidates[[index[c] for c in pool]] if qid in pools else candidates
            order = torch.argsort(corpus @ v, descending=True, stable=True)[:10].tolist()
            sums = [a + b for a, b in zip(sums, metrics([pool[i] for i in order], rels[qid]))]
    if not seen or seen != set(rels):
        raise ValueError('Missing judged queries')
    return dict(queries=len(seen), candidates=len(index),
                **dict(zip(('hit@1', 'ndcg_linear@5', 'recall@10'), [v / len(seen) for v in sums])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    model_arguments(parser)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--family', choices=['generic','image','visdoc','video'], default='generic')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    model = Encoder(args, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(json.dumps(evaluate(model, args.task, args.family, args.batch_size), indent=2))


if __name__ == '__main__':
    main()
