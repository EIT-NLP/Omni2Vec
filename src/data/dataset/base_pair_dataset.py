from datasets import Features, Value, Sequence

MULTIMODAL_FEATURES = Features(**{
    "query_text": Value(dtype='string'),
    "query_image": {
        "paths": Sequence(Value(dtype='string')),  # List of image paths (frames)
        "bytes": Sequence(Value(dtype='binary')),  # List of pre-saved image bytes
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))  # List of [width, height] pairs
    },
    "pos_text": Value(dtype='string'),
    "pos_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "neg_text": Value(dtype='string'),
    "neg_image": {
        "paths": Sequence(Value(dtype='string')),
        "bytes": Sequence(Value(dtype='binary')),
        "resolutions": Sequence(Sequence(Value(dtype='int32'), length=2))
    },
    "global_dataset_name": Value(dtype='string'),
})

RESOLUTION_MAPPING = {
    "high": (1344, 1344),
    "mid": (672, 672),
    "low": (128, 128),
}
