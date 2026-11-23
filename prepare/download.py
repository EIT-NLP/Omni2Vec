"""Download the public files of named training sources."""
import argparse
from fnmatch import fnmatch
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
import yaml


def main():
    with (Path(__file__).parents[1] / 'configs/sources.yaml').open() as f:
        sources = yaml.safe_load(f)['hub']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sources', nargs='+', choices=sources)
    args = parser.parse_args()
    for key in args.sources:
        spec = sources[key]
        repo = spec['repo']
        info = HfApi().dataset_info(repo, revision=spec.get('revision', 'main'))
        paths = [s.rfilename for s in info.siblings]
        for pattern in spec['include']:
            if not any(fnmatch(p, pattern) for p in paths):
                raise FileNotFoundError(f'{repo}: no files match {pattern}')
        print(f'{key}: {repo}@{info.sha}', flush=True)
        snapshot_download(repo, repo_type='dataset', revision=info.sha,
                          local_dir=spec['directory'], allow_patterns=spec['include'])


if __name__ == '__main__':
    main()
