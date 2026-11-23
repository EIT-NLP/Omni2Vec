"""Select the proportional BGE-M3 training subset from the extracted release."""
import argparse
import gzip
import json
from pathlib import Path
import random


def partition(path, quotas):
    names = set(path.parts) | {path.name.split('.json')[0]}
    keys = set(quotas) & names
    if len(keys) > 1:
        raise ValueError(f'Ambiguous BGE subset for {path}')
    return next(iter(keys), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('data/text/bge-m3-train-200k/train.jsonl.gz'))
    args = parser.parse_args()
    spec = json.loads((Path(__file__).parents[1] / 'configs/bge_counts.json').read_text())
    totals = spec.pop('_total')
    paths = sorted(p for p in args.input.rglob('*') if p.is_file() and p.name.endswith(('.jsonl','.jsonl.gz','.json')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.partial')
    with gzip.open(tmp, 'wt', encoding='utf-8') as out:
        for group, counts in spec.items():
            indices = set(random.Random(totals['seed'] + ':' + group).sample(range(counts['available']), counts['selected']))
            n = 0
            for path in (p for p in paths if partition(p.relative_to(args.input), spec) == group):
                opener = gzip.open if path.suffix == '.gz' else open
                with opener(path, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        if not line.strip():
                            continue
                        if n in indices:
                            row = json.loads(line)
                            if not isinstance(row, dict) or not row.get('query'):
                                raise ValueError(f'Expected query-positive JSON lines in {path}')
                            out.write(line.rstrip() + '\n')
                        n += 1
            if n != counts['available']:
                raise ValueError(f'{group}: found {n} rows, expected {counts["available"]}')
    tmp.replace(args.output)


if __name__ == '__main__':
    main()
