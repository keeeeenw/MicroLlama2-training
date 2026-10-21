"""Fetch the immutable 50B restart directly to an explicitly selected data drive."""
import argparse
import hashlib
import json
import os
from pathlib import Path

REVISION = '0153a0917a2f4dfa570bc23af01344a02369e082'
PREFIX = 'checkpoint/fineweb-edu-338m-8k-local-engram-4090/step-00381824'
MANIFEST_SHA = '9968e0d8db74a6618750ca38c71e68332c3947921f8dd75a591415afd965840c'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--max-download-bytes', type=int, default=4_700_000_000,
                        help='Maximum missing payload bytes; 0 allows verified local files only')
    args = parser.parse_args()
    root = args.root.resolve()
    if not Path('/Volumes/TrainData').is_mount() or not str(root).startswith('/Volumes/TrainData/'):
        raise ValueError('This setup stages large files only on /Volumes/TrainData')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_CACHE'] = str(root / 'hf-cache')
    os.environ['HF_XET_CACHE'] = str(root / 'xet-cache')
    from huggingface_hub import hf_hub_download
    def fetch(name):
        return Path(hf_hub_download('keeeeenw/MicroLlama2-388M-checkpoints', f'{PREFIX}/{name}',
                                   revision=REVISION, local_dir=root / 'hub'))
    manifest = fetch('checkpoint_manifest.json')
    if digest(manifest) != MANIFEST_SHA:
        raise ValueError('Pinned final manifest mismatch')
    files = json.loads(manifest.read_text())['files']
    missing = sum(item['bytes'] for name, item in files.items()
                  if not (manifest.parent / name).is_file()
                  or (manifest.parent / name).stat().st_size != item['bytes'])
    if missing > args.max_download_bytes:
        raise ValueError(f'Missing checkpoint payload {missing:,} exceeds download budget')
    for name, item in files.items():
        if Path(name).name != name:
            raise ValueError('Unsafe manifest name')
        path = fetch(name)
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {name}')
        print(json.dumps({'verified': name, 'bytes': item['bytes']}), flush=True)
    receipt = {'schema': 'mac-final50b-download-v1', 'revision': REVISION,
               'manifest_sha256': MANIFEST_SHA, 'checkpoint': str(manifest.parent),
               'verified_files': len(files) + 1,
               'bytes': sum(x['bytes'] for x in files.values()) + manifest.stat().st_size,
               'missing_payload_bytes_before_fetch': missing,
               'storage': str(root), 'training_started': False}
    (root / 'checkpoint-local-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt), flush=True)

if __name__ == '__main__':
    main()
