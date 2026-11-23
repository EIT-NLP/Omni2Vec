"""Qwen2.5-Omni chat inputs, with a shared pixel budget for each video."""
import io
from pathlib import Path

import librosa
from PIL import Image
import soundfile as sf

from src.data.dataset.video_local_dataset import _decode_video, _sample_indices, resolve_video_reference

VLM_IMAGE_TOKENS = {'qwen2_5_omni': '<|image_pad|>'}


def process_input_text(instruction, model_backbone, text='', add_image_token=False, add_video_token=False):
    token = '<|image_pad|>' if add_image_token else '<|video_pad|>' if add_video_token else ''
    return ' '.join(filter(None, [instruction, text, token]))


def open_image(value):
    if isinstance(value, dict):
        value = value.get('bytes') or value.get('path')
    with Image.open(io.BytesIO(value) if isinstance(value, bytes) else value) as img:
        img = img.convert('RGB')
    if min(img.size) < 28:
        canvas = Image.new('RGB', (max(28, img.width), max(28, img.height)))
        canvas.paste(img, (0, 0))
        return canvas
    return img


def open_audio(value):
    if isinstance(value, (Path, str, bytes)):
        value = {'bytes': value} if isinstance(value, bytes) else {'path': value}
    source = io.BytesIO(value['bytes']) if value.get('bytes') else value['path']
    with sf.SoundFile(source) as handle:
        rate = handle.samplerate
        start = int((value.get('start') or 0) * rate)
        end = value.get('end')
        handle.seek(start)
        samples = handle.read(int(end * rate) - start if end is not None and end > 0 else -1,
                              dtype='float32', always_2d=True).mean(axis=1)
    if not samples.size:
        raise ValueError('Empty audio interval')
    return librosa.resample(samples, orig_sr=rate, target_sr=16000) if rate != 16000 else samples


def item_from_pair(row, side, rng):
    text = row.get(side + '_text') or ' '
    if isinstance(text, list):
        text = rng.choice(text) if text else ' '
    item = {'text': text}
    visual = row.get(side + '_image')
    if visual:
        visual = resolve_video_reference(visual)
        paths, payloads = visual.get('paths') or [], visual.get('bytes') or []
        values = [(payloads[i] if i < len(payloads) else None) or (paths[i] if i < len(paths) else None)
                  for i in range(max(len(paths), len(payloads)))]
        if values:
            kind = 'video' if len(values) > 1 and not row['multi_image_as_images'] else 'images'
            item[kind] = values
    if row.get(side + '_audio'):
        item['audio'] = row[side + '_audio']
    return item


def prepare_inputs(processor, items):
    """A batch shares the same media signature. Flatten images in placeholder order."""
    texts, images, videos, audios = [], [], [], []
    for item in items:
        content = []
        if item.get('images'):
            for im in item['images']:
                images.append(open_image(im))
                content.append({'type': 'image'})
        if item.get('video'):
            frames = item['video']
            if isinstance(frames, (str, Path)):
                frames = _decode_video(str(frames), 8)
            videos.append([open_image(frames[i]) for i in _sample_indices(len(frames), 8)])
            content.append({'type': 'video'})
        if item.get('audio'):
            audios.append(open_audio(item['audio']))
            content.append({'type': 'audio'})
        text = item.get('text') or ' '
        for token in ('<|image_1|>', '<|image_pad|>', '<|video_pad|>', '<video>', '<image>'):
            text = text.replace(token, ' ')
        content.append({'type': 'text', 'text': ' '.join(text.split()) or ' '})
        texts.append(processor.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], chat_template=processor.chat_template,
            add_generation_prompt=False, tokenize=False))
    options = dict(text=texts, return_tensors='pt',
                   text_kwargs={'padding': True, 'truncation': False})
    if images:
        options.update(images=images, images_kwargs={'min_pixels': 3136, 'max_pixels': 1003520})
    if videos:
        options.update(videos=videos, videos_kwargs={'min_pixels': 3136, 'max_pixels': 1003520 // 8,
                                                     'use_audio_in_video': False})
    if audios:
        options.update(audio=audios, audio_kwargs={'sampling_rate': 16000, 'max_length': 2048000})
    if not (images or videos or audios):
        options['text_kwargs'].update(truncation=True, max_length=2048)
    return processor(**options)
