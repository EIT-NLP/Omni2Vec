"""Create one retrieval pair per session by concatenating its annotated evidence turns."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert(record):
    meta = record['meta']
    history = meta['turns'][:-1]
    query = next((msg['content'].strip() for msg in reversed(record['messages'])
                  if msg['role'] == 'user'), '')
    evidence = [str(history[i]).strip() for i in dict.fromkeys(map(int, meta.get('evidence_turns') or []))
                if 0 <= i < len(history) and history[i]]
    positive = '\n\n'.join(filter(None, evidence))
    if query and positive:
        return {'uid': str(record['id']), 'qry': query, 'pos_text': positive}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/memory/evo'))
    args = parser.parse_args()
    paths = sorted(args.input.rglob('data_chunk_*.json'), key=lambda p: int(p.stem.split('_')[-1]))
    if not paths:
        raise FileNotFoundError('Expected EvoTrain data_chunk_*.json in the input directory')
    args.output.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([('uid', pa.string()), ('qry', pa.string()), ('pos_text', pa.string())])
    tmp = args.output / 'train.partial'
    count = 0
    with pq.ParquetWriter(tmp, schema) as writer:
        for path in paths:
            rows = [pair for record in json.loads(path.read_text()) if (pair := convert(record))]
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=1024)
                count += len(rows)
    if not count:
        raise ValueError('No sessions with both a query and annotated history')
    tmp.replace(args.output / 'train.parquet')
    print(f'Wrote {count} merged positives')


if __name__ == '__main__':
    main()
