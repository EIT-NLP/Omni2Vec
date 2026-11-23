"""Materialize one source at a time. Each Parquet row group is one global batch."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.data.dataset.capped_epoch_dataset import FEATURES, iter_selection
from src.data.dataset.gui_agent_dataset import _buffer_shuffle


def pack(epoch):
    index = yaml.safe_load((epoch / 'dataset_config.yaml').read_text())
    output = epoch / 'batches'
    output.mkdir(exist_ok=True)
    for entry in index.values():
        path = Path(entry['selection_manifest'])
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != entry['selection_sha256']:
            raise ValueError(f'Selection changed: {path}')
        dest = output / (digest + '.parquet')
        if dest.exists():
            continue
        spec = json.loads(raw)
        rows = iter_selection(spec, 'qwen2_5_omni')
        shuffled = _buffer_shuffle(itertools.islice(rows, spec['selected_rows']), 2048, spec['seed'])
        ordered = itertools.chain(shuffled, rows)
        tmp = dest.with_suffix('.partial')
        size = spec['global_batch_size']
        with pq.ParquetWriter(tmp, FEATURES.arrow_schema, compression='snappy') as writer:
            for _ in range(spec['padded_rows'] // size):
                batch = list(itertools.islice(ordered, size))
                if len(batch) != size:
                    raise ValueError(f'Incomplete batch in {path}')
                writer.write_table(pa.Table.from_pylist(batch, schema=FEATURES.arrow_schema), row_group_size=size)
            if next(ordered, None) is not None:
                raise ValueError(f'Extra records in {path}')
        tmp.replace(dest)
        print(spec['name'], spec['selected_rows'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', type=Path, default=Path('data/epoch'))
    pack(parser.parse_args().epoch)
