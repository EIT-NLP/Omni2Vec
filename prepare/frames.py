"""Convert video files or extracted frame folders to the eight-frame training format."""
import argparse
import io
import json
from pathlib import Path

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset.video_local_dataset import _decode_video, _natural_key, _sample_indices


def read_frames(path):
    if path.is_file():
        return _decode_video(str(path), 8)
    frames = sorted((p for p in path.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                    key=lambda p: _natural_key(p.name))
    payloads = []
    for index in _sample_indices(len(frames), 8):
        output = io.BytesIO()
        with Image.open(frames[index]) as img:
            img.convert('RGB').save(output, format='JPEG', quality=90)
        payloads.append(output.getvalue())
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path,
                        help='Training JSONL with video_id, path and optional pos_text (action label)')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    schema = pa.schema([('video_id', pa.string()), ('pos_text', pa.string()), ('frame_bytes', pa.list_(pa.binary()))])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    tmp = args.output.with_suffix('.partial')
    with args.manifest.open() as source, pq.ParquetWriter(tmp, schema) as writer:
        for line in source:
            r = json.loads(line)
            key = str(r['video_id'])
            if key in seen:
                raise ValueError(f'Duplicate video: {key}')
            seen.add(key)
            row = {'video_id': key, 'pos_text': r.get('pos_text'), 'frame_bytes': read_frames(args.manifest.parent / r['path'])}
            writer.write_table(pa.Table.from_pylist([row], schema=schema))
    if not seen:
        raise ValueError('Empty training manifest')
    tmp.replace(args.output)


if __name__ == '__main__':
    main()
