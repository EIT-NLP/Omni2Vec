"""Convert an MMEB image-query test split to per-query candidate pools."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', required=True, type=Path, help='Directory of test Parquet shards')
    p.add_argument('--images', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    files = sorted(a.metadata.glob('test*.parquet'))
    if not files:
        raise FileNotFoundError(f'Expected test*.parquet in {a.metadata}')
    seen = set()
    a.output.mkdir(parents=True, exist_ok=False)
    def item(text, path, instruction):
        obj = {'text': ' '.join(filter(None, [instruction, text])).strip()}
        if path:
            obj['images'] = [os.path.relpath(a.images / path, a.output)]
        return obj
    with ExitStack() as stack:
        queries, candidates, judgments = [stack.enter_context((a.output / (n + '.jsonl')).open('w'))
                                          for n in ('queries','candidates','qrels')]
        index = 0
        for file in files:
            for batch in pq.ParquetFile(file).iter_batches(batch_size=64):
                for row in batch.to_pylist():
                    qid = str(index)
                    index += 1
                    queries.write(json.dumps(dict(id=qid, **item(row.get('qry_text'), row.get('qry_img_path'), row.get('qry_inst')))) + '\n')
                    texts, paths = row.get('tgt_text') or [], row.get('tgt_img_path') or []
                    if not any(texts):
                        texts = []
                    if not any(paths):
                        paths = []
                    if not (texts or paths) or (texts and paths and len(texts) != len(paths)):
                        raise ValueError(f'Invalid MMEB candidate list at query {qid}')
                    ids = []
                    for i in range(max(len(texts), len(paths))):
                        obj = item(texts[i] if texts else '', paths[i] if paths else '', row.get('tgt_inst'))
                        key = hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
                        if key not in seen:
                            seen.add(key)
                            candidates.write(json.dumps(dict(id=key, **obj)) + '\n')
                        ids.append(key)
                    judgments.write(json.dumps({'query_id': qid, 'mode': 'exhaustive', 'candidate_ids': ids,
                                                'candidate_scores': [1] + [0] * (len(ids)-1)}) + '\n')


if __name__ == '__main__':
    main()
