"""Remove copied OS sidecars only from this task's isolated Python environment.

ExFAT wheel-cache copies can carry ._*.dist-info files into APFS. Python then
mistakes these AppleDouble headers for package metadata. No data/cache payloads
or files outside the explicitly marked virtual environment are removed.
"""
from pathlib import Path
import argparse


def clean(root):
    root = Path(root).resolve()
    if root.name != '.venv-mac' or not (root / 'pyvenv.cfg').is_file():
        raise ValueError('Only the isolated .venv-mac environment may be cleaned')
    files = []
    for path in root.rglob('._*'):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024**2:
            raise ValueError('Unexpected sidecar type; no cleanup performed')
        with path.open('rb') as stream:
            if stream.read(8) not in (bytes.fromhex('0005160700020000'), bytes.fromhex('0005160700010000')):
                raise ValueError('Unrecognized sidecar; no cleanup performed')
        files.append(path)
    for path in files:
        path.unlink()
    return len(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('environment', type=Path)
    args = parser.parse_args()
    print(f'Removed {clean(args.environment)} copied AppleDouble metadata files from the isolated environment.')
