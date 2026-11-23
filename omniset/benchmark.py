"""Pure data/ranking contract matching OmniSET style0 without model dependencies."""
import hashlib
import json
import math
from pathlib import Path
from collections import Counter
import numpy as np

MODS = 'TIVA'
EXT = {'T': 'txt', 'I': 'jpg', 'V': 'mp4', 'A': 'wav'}
NAMES = {'T': 'text', 'I': 'image', 'V': 'video', 'A': 'audio'}


def name(identifier, mod): return f'COCO_val2014_{int(identifier):012d}.{EXT[mod]}'


def modality(filename): return {v: k for k, v in EXT.items()}[filename.rsplit('.', 1)[1]]


def semantic_id(filename): return int(Path(filename).stem.rsplit('_', 1)[1])


def instruction(source, target):
    assert source != target
    return f'Find the {NAMES[target]} that best matches the given {NAMES[source]}: '


def load_benchmark(root, mods='TIVA'):
    root = Path(root)
    if mods not in ['TIVA', 'TIV']: raise ValueError('Only fixed TIVA or TIV candidate pools are supported')
    rows = [json.loads(line) for line in (root / 'omniset.jsonl').open() if line.strip()]
    catalog = [json.loads(line) for line in (root / 'catalog.jsonl').open() if line.strip()]
    captions = {int(r['image_id']): r['captions'][0] for r in catalog if r.get('captions')}
    assert rows and len({r['image_id'] for r in rows}) == len(rows)
    candidates = {}; queries = []
    for row in rows:
        identifier = int(row['image_id'])
        ids = list(set([int(i) for i in row['hard_negatives']] + [identifier]))
        pool = [name(i, mod) for i in ids for mod in mods]
        for i in ids:
            if i not in captions: raise ValueError(f'Missing actual candidate caption: {i}')
            for mod in mods:
                filename = name(i, mod)
                if filename in candidates: continue
                item = {'id': filename, 'modality': mod}
                if mod == 'T': item['text'] = captions[i]
                else:
                    item['path'] = str(root / {'I': 'val2014', 'V': 'videos', 'A': 'audios'}[mod] / filename)
                    if not Path(item['path']).is_file(): raise FileNotFoundError(item['path'])
                candidates[filename] = item
        for source in mods:
            for target in mods:
                if source == target: continue
                source_item = dict(candidates[name(identifier, source)])
                if source == 'T': source_item['text'] = row['qry_text']
                queries.append({'direction': source.lower()+'2'+target.lower(),
                                'label': name(identifier, target), 'input': source_item,
                                'instruction': instruction(source, target), 'pool': pool})
    return {'modalities': mods, 'source_rows': len(rows), 'candidates': list(candidates.values()),
            'queries': queries, 'queries_sha256': hashlib.sha256((root/'omniset.jsonl').read_bytes()).hexdigest(),
            'catalog_sha256': hashlib.sha256((root/'catalog.jsonl').read_bytes()).hexdigest()}


def rank_metrics(label, ranking, scores):
    if label not in ranking or len(set(ranking)) != len(ranking): raise ValueError('Expected complete unique candidate ranking')
    if len(scores) != len(ranking) or not all(math.isfinite(float(x)) for x in scores): raise ValueError('Invalid scores')
    target = modality(label); sid = semantic_id(label); top = ranking[0]
    rank = ranking.index(label) + 1; top10 = ranking[:10]; counts = Counter(modality(x) for x in top10)
    return {'hit@1': float(rank == 1), 'hit@10': float(rank <= 10), 'mrr': 1.0/rank,
            'semantic_accuracy': float(semantic_id(top) == sid),
            'target_modality_accuracy': float(modality(top) == target),
            'joint_accuracy': float(top == label), 'top10_counts': dict(counts)}


def evaluate_rankings(benchmark, query_vectors, candidate_vectors, candidate_ids, scope=None):
    scope = scope or benchmark['modalities']; index = {n: i for i, n in enumerate(candidate_ids)}
    if len(index) != len(candidate_ids): raise ValueError('Duplicate encoded candidate IDs')
    if len(query_vectors) != len(benchmark['queries']): raise ValueError('Query embeddings are incomplete')
    if query_vectors.ndim != 2 or candidate_vectors.ndim != 2 or query_vectors.shape[1] != candidate_vectors.shape[1]:
        raise ValueError('Embedding dimensions disagree')
    if not np.isfinite(query_vectors).all() or not np.isfinite(candidate_vectors).all(): raise ValueError('Non-finite embeddings')
    directions = {}; predictions = []
    for row, vector in zip(benchmark['queries'], query_vectors):
        if row['direction'][0].upper() not in scope or row['direction'][-1].upper() not in scope: continue
        pool = [n for n in row['pool'] if modality(n) in scope]
        if not all(n in index for n in pool): raise ValueError('Unencoded candidate in query pool')
        scores = candidate_vectors[[index[n] for n in pool]] @ vector
        order = np.argsort(-scores, kind='stable'); ranking = [pool[i] for i in order]; ranked_scores = scores[order].tolist()
        metrics = rank_metrics(row['label'], ranking, ranked_scores)
        directions.setdefault(row['direction'], []).append(metrics)
        predictions.append({'direction': row['direction'], 'label': [row['label']],
                            'query': {'instruction': row['instruction'], 'input': row['input']},
                            'prediction': ranking, 'prediction_scores': ranked_scores})
    expected = {a.lower()+'2'+b.lower() for a in scope for b in scope if a != b}
    if set(directions) != expected or any(len(rows) != benchmark['source_rows'] for rows in directions.values()):
        raise ValueError('Directions or query counts incomplete')
    summary = {}
    keys = ['hit@1', 'hit@10', 'mrr', 'semantic_accuracy', 'target_modality_accuracy', 'joint_accuracy']
    for direction, rows in directions.items():
        counts = Counter()
        for r in rows: counts.update(r['top10_counts'])
        dominant, number = counts.most_common(1)[0]
        summary[direction] = {'num_pred': len(rows), 'num_data': len(rows),
                              **{m: sum(r[m] for r in rows)/len(rows) for m in keys},
                              'top10_dominant_modality': dominant, 'top10_dominant_share': number/sum(counts.values())}
    return {'complete': True, 'candidate_modalities': scope, 'direction_count': len(summary),
            'query_count': len(predictions), 'directions': summary,
            'macro': {m: sum(r[m] for r in summary.values())/len(summary) for m in keys}}, predictions
