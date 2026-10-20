import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from microllama.common.paths import data_root
from microllama.data_io.corpus import split_for


def test_parallel_bytes_match_sequential_and_resume(tmp_path, monkeypatch):
    pa = pytest.importorskip('pyarrow')
    pq = pytest.importorskip('pyarrow.parquet')
    tokenizer = data_root() / 'tokenizer'
    if not (tokenizer / 'tokenizer.json').exists():
        pytest.skip('Pinned tokenizer fixture not cached')
    train = next(f'train document {i} ordinary <|im_start|> words' for i in range(10000)
                 if split_for(f'train document {i} ordinary <|im_start|> words') == 'train')
    val = next(f'validation document {i} more ordinary words' for i in range(10000)
               if split_for(f'validation document {i} more ordinary words') == 'validation')
    raw = tmp_path / 'raw'
    items = []
    for index in range(3):
        path = raw / f'sample/10BT/{index}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist([{'id': f'{index}-{j}', 'text': train if j % 2 else val}
                                            for j in range(520)]), path)
        items.append({'path': str(path.relative_to(raw)), 'bytes': path.stat().st_size,
                      'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    source = {'files': items, 'revision': 'fixture', 'data_dir': str(raw),
              'download_complete': True, 'verified': True}
    roots = [tmp_path / 'sequential', tmp_path / 'parallel']
    for root in roots:
        shutil.copytree(tokenizer, root / 'tokenizer', ignore=shutil.ignore_patterns('.cache'))
        manifest = root / 'raw/fineweb-edu-10bt/manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(source))
    from microllama.data_io.prepare import main as sequential
    monkeypatch.setenv('MICROLLAMA_DATA_ROOT', str(roots[0]))
    monkeypatch.setattr(sys, 'argv', ['prepare', '--chunk-tokens', '1000'])
    sequential()
    # Recover one complete source and the first immutable chunk of an interrupted source.
    source_tokens = roots[0] / 'tokens'
    target_tokens = roots[1] / 'tokens'
    target_tokens.mkdir()
    for path in source_tokens.glob('0-*'):
        shutil.copytree(path, target_tokens / path.name)
    first = sorted(source_tokens.glob('1-*'))[0]
    shutil.copytree(first, target_tokens / first.name)
    (target_tokens / '.sources').mkdir()
    shutil.copy2(source_tokens / '.sources/0.json', target_tokens / '.sources/0.json')
    (target_tokens / '1-000001.partial').mkdir()
    # The next source may arrive later; consume the ready source before the download is complete.
    import threading
    import time
    from microllama.data_io.transfer import atomic_json
    cache = tmp_path / 'arriving-cache'
    ready = cache / items[1]['path']
    ready.parent.mkdir(parents=True)
    shutil.copy2(raw / items[1]['path'], ready)
    streaming_source = {**source, 'data_dir': str(cache), 'download_complete': False, 'verified': False}
    raw_manifest = roots[1] / 'raw/fineweb-edu-10bt/manifest.json'
    atomic_json(raw_manifest, streaming_source)
    observed = []
    def finish_download():
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (target_tokens / '1-000001/manifest.json').exists():
            time.sleep(.05)
        observed.append((target_tokens / '1-000001/manifest.json').exists())
        shutil.copy2(raw / items[2]['path'], cache / items[2]['path'])
        atomic_json(raw_manifest, {**streaming_source, 'download_complete': True, 'verified': True})
    downloader = threading.Thread(target=finish_download)
    downloader.start()
    from microllama.data_io.parallel_prepare import main as parallel
    from microllama.common.resources import available_cpus
    monkeypatch.setenv('MICROLLAMA_DATA_ROOT', str(roots[1]))
    monkeypatch.setattr(sys, 'argv', ['parallel', '--root', str(roots[1]), '--workers', str(min(2, available_cpus())),
                                    '--threads', '1', '--chunk-tokens', '1000'])
    try:
        parallel()
    finally:
        downloader.join(timeout=25)
    assert observed == [True], 'No token chunk was published while the remaining shard was downloading'
    def inventory(root):
        return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*')
                if p.is_file() and p.name != '.producer.lock'}
    assert inventory(target_tokens) == inventory(source_tokens)
    before = inventory(target_tokens)
    parallel()
    assert inventory(target_tokens) == before
