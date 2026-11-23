from __future__ import annotations
from collections import Counter, defaultdict, deque
import hashlib
import json
import random
from pathlib import Path
import pyarrow.parquet as pq
import yaml

def select(groups, cap, seed):
    rng = random.Random(seed)
    queues = []
    for key in sorted(groups):
        rows = list(groups[key])
        rng.shuffle(rows)
        queues.append(deque(rows))
    rng.shuffle(queues)
    chosen = []
    while queues and len(chosen) < cap:
        remaining = []
        for queue in queues:
            if len(chosen) == cap:
                break
            chosen.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return sorted(chosen)

def metadata(files, columns, kind='parquet'):
    for f, path in enumerate(files):
        if kind == 'lance':
            import lance
            batches = lance.dataset(path).scanner(columns=columns, batch_size=4096).to_batches()
        else:
            batches = pq.ParquetFile(path).iter_batches(columns=columns, batch_size=4096)
        index = 0
        for batch in batches:
            for row in batch.to_pylist():
                yield f, index, row
                index += 1

def _build(root, config_path, seed, destination):
    selection_root = root / 'selections'
    selection_root.mkdir()
    original = yaml.safe_load(config_path.read_text())
    config, reports = {}, {}

    def add(name, family, parser, files, groups, cap, **extra):
        selected = select(groups, cap, seed)
        if not selected:
            raise ValueError(f'Empty source {name}')
        n = len(selected)
        padded = (n + 63) // 64 * 64
        identities = []
        for path in files:
            p = Path(path)
            identities.append({'path': str(p), 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns})
        spec = {'name': name, 'family': family, 'original_parser': parser, 'files': files,
                'kind': extra.pop('kind', 'parquet'), 'selection': selected,
                'available_rows': sum(map(len, groups.values())), 'selected_rows': n,
                'padded_rows': padded, 'global_batch_size': 64, 'cap': cap, 'seed': seed,
                'input_files': identities, **extra}
        path = selection_root / (name.replace('/', '__') + '.json')
        path.write_text(json.dumps(spec, ensure_ascii=False) + '\n')
        config[name] = {'family': family, 'dataset_parser': 'capped_epoch', 'selection_manifest': str(destination / 'selections' / path.name),
                        'selection_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'weight': n}
        reports[name] = {k: spec[k] for k in ('family', 'selected_rows', 'padded_rows', 'cap', 'available_rows')}
        print('CAPPED_SOURCE', name, n, 'padding', padded - n, flush=True)

    cls = {'ImageNet_1K', 'N24News', 'HatefulMemes', 'VOC2007', 'SUN397'}
    for name, spec in original.items():
        parser, family = spec['dataset_parser'], spec['family']
        if parser == 'mmeb_zip':
            task = spec['task_name']
            files = [str(p) for p in sorted((Path(spec['dataset_root']) / task).glob('train-*.parquet'))]
            groups = defaultdict(list)
            columns = ['qry', 'qry_image_path', 'pos_text', 'pos_image_path']
            if spec.get('require_explicit_text_negative'):
                columns.append('neg_text')
            for f, i, row in metadata(files, columns):
                if spec.get('require_explicit_text_negative') and not str(row.get('neg_text') or '').strip():
                    continue
                key = str(row['pos_text']) if task in cls else str(row.get('qry_image_path') or row.get('pos_image_path') or i)
                groups[key].append((f, i, -1))
            archive = str(Path(spec['dataset_root']) / 'images_zip' / f'{task}.zip')
            if not Path(archive).exists():
                alternatives = sorted((Path(spec['dataset_root']) / 'images_zip').glob(f'{task}*'))
                if len(alternatives) != 1:
                    raise ValueError(f'Missing image archive: {task}')
                archive = str(alternatives[0])
            add(name, family, parser, files, groups, 20000 if task in cls else 70000,
                task_name=task, archive=archive)
        elif parser == 'pixmo':
            files = [str(p) for p in sorted(Path(spec['parquet_dir']).glob('train-*.parquet'))]
            groups = defaultdict(list)
            for f, i, row in metadata(files, ['image_id', 'questions']):
                questions = (row.get('questions') or {}).get('question') or []
                if row.get('image_id') is not None and questions:
                    q = int(hashlib.sha256(f'{seed}:{row["image_id"]}'.encode()).hexdigest(), 16) % len(questions)
                    groups[str(row['image_id'])].append((f, i, q))
            # Repeated page IDs are represented by one record in this epoch.
            groups = {k: [v[0]] for k, v in groups.items()}
            add(name, family, parser, files, groups, 50000, global_dataset_name=spec['global_dataset_name'])
        elif parser in ('vidore', 'visrag'):
            base = Path(spec['dataset_path'])
            kind = 'lance' if base.suffix == '.lance' else 'parquet'
            files = [str(base)] if kind == 'lance' else [str(p) for p in sorted(base.glob('train*.parquet'))]
            columns = ['image_filename'] if parser == 'vidore' else ['source']
            groups = defaultdict(list)
            for f, i, row in metadata(files, columns, kind):
                key = str(row.get('image_filename') or row.get('source') or i)
                groups[key].append((f, i, -1))
            add(name, family, parser, files, groups, 10**9, kind=kind,
                global_dataset_name=spec['global_dataset_name'])
        elif parser == 'audio_pair':
            files = [str(p) for p in sorted(Path(spec['data_path']).glob('*.parquet'))]
            groups = defaultdict(list)
            for f, i, row in metadata(files, ['pos_text_candidates', 'uid', 'task_type']):
                key = json.dumps(row['pos_text_candidates']) if row['task_type'] == 'audio_text_classification' else str(row['uid'])
                groups[key].append((f, i, -1))
            ds = name.split('/')[-1]
            cap = {'nsynth': 20000, 'fsd50k': 30000, 'speechcommand': 30000}.get(ds, 50000)
            add(name, family, parser, files, groups, cap, data_path=spec['data_path'])
        elif parser == 'prebuilt_pair_mix':
            files = [str(p) for p in sorted(Path(spec['data_path']).glob(spec['parquet_pattern']))]
            groups = defaultdict(lambda: defaultdict(list))
            for f, i, row in metadata(files, ['global_dataset_name', 'sampling_group_id', 'pos_text']):
                source = row['global_dataset_name'] if family == 'video' else name
                key = str(row['pos_text']) if 'classification' in source else str(row.get('sampling_group_id') or i)
                groups[source][key].append((f, i, -1))
            if not groups:
                raise ValueError(f'Empty source {name}')
            for source, buckets in sorted(groups.items()):
                cap = 10**9
                if family == 'video':
                    cap = 50000 if 'classification' in source else 200000 if source.endswith('/qa') else 150000 if source.endswith('/caption') else 10**9
                add(source, family, parser, files, buckets, cap)
        elif parser == 'gui_agent':
            for gui in spec['source_names']:
                files = [str(Path(spec['dataset_root']) / gui / 'original.parquet')]
                groups = defaultdict(lambda: defaultdict(list))
                for f, i, row in metadata(files, ['retrieval_type', 'pos_id']):
                    groups[str(row['retrieval_type'])][str(row['pos_id'])].append((f, i, -1))
                if not groups:
                    raise ValueError(f'Empty GUI source {gui}')
                for task, buckets in sorted(groups.items()):
                    add(f'gui/{gui}/{task}', family, parser, files, buckets, 10**9,
                        gui_source=gui, image_root=str(Path(spec['dataset_root']) / gui / 'images'))
        elif parser == 'evo_memory':
            files = [str(p) for p in sorted(Path(spec['data_path']).glob('*.parquet'))]
            groups = defaultdict(list)
            for f, i, row in metadata(files, ['qry']):
                groups[str(row['qry'])].append((f, i, -1))
            add('memory/evo-full', family, parser, files, groups, 10**9)
        else:
            raise ValueError(parser)
    schedule = [name for name, report in reports.items() for _ in range(report['padded_rows'] // 64)]
    random.Random(seed).shuffle(schedule)
    manifest = {'version': 1, 'global_batch_size': 64, 'seed': seed, 'optimizer_steps': len(schedule),
                'source_schedule': schedule, 'family_blocks': dict(Counter(reports[n]['family'] for n in schedule)),
                'source_rows': reports, 'valid_rows': sum(r['selected_rows'] for r in reports.values()),
                'padding_rows': sum(r['padded_rows'] - r['selected_rows'] for r in reports.values()),
                'epoch_policy': 'each selected record once; masked padding; no cycling; Pixmo one question per page'}
    (root / 'dataset_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (root / 'source_schedule.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('CAPPED_EPOCH_PREPARED', manifest['valid_rows'], 'steps', len(schedule), 'padding', manifest['padding_rows'], flush=True)

def build(root, config_path, seed=42, reference=None):
    from tempfile import TemporaryDirectory
    if root.exists():
        raise FileExistsError(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root.parent) as staging:
        staged = Path(staging) / 'epoch'
        staged.mkdir()
        _build(staged, config_path, seed, root)
        if reference:
            import csv
            with reference.open() as stream:
                expected = {r['source']: int(r['selected_rows']) for r in csv.DictReader(stream)}
            report = json.loads((staged / 'source_schedule.json').read_text())
            actual = {k: v['selected_rows'] for k, v in report['source_rows'].items()}
            differences = {k: (expected.get(k), actual.get(k)) for k in sorted(expected.keys() | actual.keys())
                           if expected.get(k) != actual.get(k)}
            if differences:
                raise ValueError(f'Source counts differ (reference, prepared): {differences}')
        staged.rename(root)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/train.yaml'))
    parser.add_argument('--output', type=Path, default=Path('data/epoch'))
    parser.add_argument('--check-counts', type=Path, metavar='CSV', help='Require the published per-source row counts')
    args = parser.parse_args()
    build(args.output, args.config, reference=args.check_counts)
