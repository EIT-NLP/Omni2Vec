"""Convert train-split annotation JSONL to audio-pair Parquet with embedded media."""
import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ASSET = pa.struct([('asset_id', pa.string()), ('bytes', pa.binary())])
REF = pa.struct([('asset_id', pa.string()), ('start', pa.float32()), ('end', pa.float32())])
SCHEMA = pa.schema([('uid', pa.string()), ('dataname', pa.string()), ('task_type', pa.string()),
                    ('query_text_candidates', pa.list_(pa.string())), ('pos_text_candidates', pa.list_(pa.string())),
                    ('query_audio', REF), ('pos_audio', REF), ('audio_assets', pa.list_(ASSET)),
                    ('pos_image_asset_ids', pa.list_(pa.string())), ('image_assets', pa.list_(ASSET))])
TASKS = {'audiocaps': 'text_audio_retrieval', 'clotho': 'text_audio_retrieval',
         'speechcoco': 'audio_image_retrieval', 'ave': 'audio_video_retrieval',
         **dict.fromkeys(['fsd50k', 'nsynth', 'speechcommand', 'urbansound8k'], 'audio_text_classification')}


def convert(row, source, root):
    task = TASKS[source]
    audio = row['audio']
    ref = {'asset_id': audio, 'start': row.get('start'), 'end': row.get('end')}
    positive_audio = task == 'text_audio_retrieval'
    images = row.get('images') or []
    if task in ('audio_image_retrieval', 'audio_video_retrieval') and not images:
        raise ValueError('Missing positive media')
    texts = row.get('texts') or []
    if (positive_audio or task == 'audio_text_classification') and not texts:
        raise ValueError('Missing captions or class labels')
    return {'uid': str(row['id']), 'dataname': source, 'task_type': task,
            'query_text_candidates': texts if positive_audio else [],
            'pos_text_candidates': texts if task == 'audio_text_classification' else [],
            'query_audio': None if positive_audio else ref, 'pos_audio': ref if positive_audio else None,
            'audio_assets': [{'asset_id': audio, 'bytes': (root / audio).read_bytes()}],
            'pos_image_asset_ids': images,
            'image_assets': [{'asset_id': key, 'bytes': (root / key).read_bytes()} for key in images]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, choices=TASKS)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix('.partial')
    count = 0
    with args.manifest.open() as stream, pq.ParquetWriter(tmp, SCHEMA) as writer:
        batch = []
        for line in stream:
            batch.append(convert(json.loads(line), args.source, args.manifest.parent))
            count += 1
            if len(batch) == 32:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    if not count:
        raise ValueError('Empty training manifest')
    tmp.replace(args.output)


if __name__ == '__main__':
    main()
