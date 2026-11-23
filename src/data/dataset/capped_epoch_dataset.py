from __future__ import annotations
from collections import defaultdict
import zipfile
import datasets
import pyarrow.parquet as pq
from src.data.dataset.visdoc_utils import VISDOC_FEATURES
from src.data.dataset.audio_pair import OUTPUT_FEATURES, _convert_row

FEATURES = datasets.Features({
    **dict(VISDOC_FEATURES),
    **dict(OUTPUT_FEATURES),
    'multi_image_as_images': datasets.Value('bool'),
    'valid_record': datasets.Value('bool'),
})

def selected_rows(files, selections, kind, batch_size=128):
    for file_index, path in enumerate(files):
        chosen = [(int(row), question) for f, row, question in selections if f == file_index]
        if not chosen:
            continue
        if kind == 'lance':
            import lance
            source = lance.dataset(path)
            for start in range(0, len(chosen), batch_size):
                part = chosen[start:start + batch_size]
                rows = source.take([row for row, _ in part]).to_pylist()
                yield from ((row, question) for row, (_, question) in zip(rows, part))
        else:
            source = pq.ParquetFile(path)
            by_group = defaultdict(list)
            offsets = []
            offset = 0
            for group in range(source.num_row_groups):
                offsets.append(offset)
                offset += source.metadata.row_group(group).num_rows
            import bisect
            for row, question in chosen:
                group = bisect.bisect_right(offsets, row) - 1
                by_group[group].append((row - offsets[group], question))
            for group in sorted(by_group):
                wanted = sorted(by_group[group])
                position, emitted = 0, 0
                for table in source.iter_batches(batch_size=batch_size, row_groups=[group]):
                    local = []
                    questions = []
                    while emitted < len(wanted) and wanted[emitted][0] < position + len(table):
                        local.append(wanted[emitted][0] - position)
                        questions.append(wanted[emitted][1])
                        emitted += 1
                    if local:
                        rows = table.take(local).to_pylist()
                        yield from zip(rows, questions)
                    position += len(table)
                    if emitted == len(wanted):
                        break
                if emitted != len(wanted):
                    raise ValueError(f'Selection exceeds parquet rows: {path}')

def normalized(example, valid=True):
    row = {key: example.get(key) for key in FEATURES}
    for key in ('query_text', 'pos_text'):
        value = row[key]
        row[key] = value if isinstance(value, list) else [str(value or ' ')]
    row['neg_text'] = row.get('neg_text') or ''
    row['valid_record'] = valid
    row['multi_image_as_images'] = bool(row.get('multi_image_as_images'))
    for key in ('pair_id', 'task_id', 'negative_pair_id'):
        row[key] = int(row[key] or 0)
    return row

def padding(example):
    result = dict(example)
    for key in ('query_text', 'pos_text'):
        result[key] = [' ']
    result['neg_text'] = ''
    for key in ('query_image', 'pos_image', 'neg_image', 'query_audio', 'pos_audio'):
        result[key] = None
    result['valid_record'] = False
    return result

def iter_selection(spec, model_backbone, resolution='low'):
    parser = spec['original_parser']
    archive = None
    if parser == 'mmeb_zip':
        archive = zipfile.ZipFile(spec['archive'])
    count, last = 0, None
    try:
        for row, question in selected_rows(spec['files'], spec['selection'], spec['kind']):
            if parser == 'mmeb_zip':
                from src.data.dataset.mmeb_zip_dataset import _convert_example
                from src.data.dataset.base_pair_dataset import RESOLUTION_MAPPING
                example = _convert_example(row, archive, spec['task_name'], model_backbone,
                                           list(RESOLUTION_MAPPING.get(resolution, RESOLUTION_MAPPING['low'])))
            elif parser == 'pixmo':
                from src.data.dataset.pixmo import _examples
                source_row = dict(row)
                source_row['questions'] = {'question': [row['questions']['question'][question]]}
                example = next(_examples([source_row], model_backbone, resolution,
                                        spec['global_dataset_name'], page_buffer_size=1))
            elif parser == 'vidore':
                from src.data.dataset.vidore_dataset import _examples
                example = next(_examples([row], model_backbone, resolution, spec['global_dataset_name']))
            elif parser == 'visrag':
                from src.data.dataset.visrag_dataset import process_query, query_source2prompt, target_source2prompt
                from src.model.processor import VLM_IMAGE_TOKENS
                from src.data.dataset.visdoc_utils import attach_identity
                image = row['image']
                identity = f"{row.get('source', '')}:{image.get('path') or ''}"
                example = attach_identity({
                    'query_text': process_query(str(row['query']), query_source2prompt.get(row['source'], ''), ''),
                    'pos_text': process_query('', target_source2prompt.get(row['source'], ''), VLM_IMAGE_TOKENS[model_backbone]),
                    'pos_image': {'bytes': [image.get('bytes')], 'paths': [image.get('path') or ''], 'resolutions': [[128, 128]]},
                }, spec['global_dataset_name'], identity)
            elif parser == 'audio_pair':
                example = _convert_row(row, spec['data_path'], model_backbone, resolution)
            elif parser == 'gui_agent':
                from src.data.dataset.gui_agent_dataset import _convert_example
                example = _convert_example(row, spec['gui_source'], spec['image_root'], model_backbone, 1)
            elif parser == 'evo_memory':
                from src.data.dataset.evo_memory_dataset import _convert_example
                example = _convert_example(row)
            elif parser == 'prebuilt_pair_mix':
                example = row
            else:
                raise ValueError(f'Unsupported capped parser: {parser}')
            last = normalized(example)
            count += 1
            yield last
        if count != spec['selected_rows'] or last is None:
            raise ValueError(f"Capped source count mismatch: {count} != {spec['selected_rows']}")
        for _ in range(spec['padded_rows'] - count):
            yield padding(last)
    finally:
        if archive is not None:
            archive.close()
