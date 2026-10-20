"""Parallelize source shards with isolated writers and one corpus publisher."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import time

from microllama.common.paths import data_root
from microllama.common.resources import tokenization_budget
from microllama.data_io.download import sha256
from microllama.data_io.encoding import ENCODING_POLICY_VERSION, load_ordinary_text_tokenizer, require_encoding_policy
from microllama.data_io.prepare import prepare_source, update_summary
from microllama.data_io.transfer import atomic_json


def link_chunk(source, destination):
    destination.mkdir()
    for file in source.iterdir():
        if not file.is_file() or file.is_symlink():
            raise RuntimeError(f'Unexpected chunk entry: {file}')
        os.link(file, destination / file.name)


def worker(item, source, tokenizer_path, digest, eos, stage, chunk_tokens, threads):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    os.environ['RAYON_NUM_THREADS'] = str(threads)
    os.environ['OMP_NUM_THREADS'] = '1'
    import pyarrow
    pyarrow.set_cpu_count(threads)
    pyarrow.set_io_thread_count(1)
    path = Path(source['data_dir']) / item['path']
    if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
        raise RuntimeError(f'Source checksum mismatch: {path}')
    tok = load_ordinary_text_tokenizer(tokenizer_path)
    print(f'Worker tokenizing {item["path"]} with {threads} tokenizer threads', flush=True)
    return prepare_source(path, item, source, tok, digest, eos, stage, chunk_tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=data_root())
    parser.add_argument('--workers', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--chunk-tokens', type=int, default=8_388_608)
    args = parser.parse_args()
    if args.chunk_tokens < 1 or any(v is not None and v < 1 for v in (args.workers, args.threads)):
        parser.error('Worker, thread and chunk counts must be positive')
    root = args.root
    output = root / 'tokens'
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.producer.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((root / 'raw/fineweb-edu-10bt/manifest.json').read_text())
    budget = tokenization_budget(len(source['files']), args.workers, args.threads)
    args.workers, args.threads = budget['workers'], budget['threads']
    tokenizer_path = root / 'tokenizer/tokenizer.json'
    info = json.loads((root / 'tokenizer/source_manifest.json').read_text())
    digest = sha256(tokenizer_path)
    if digest != info['tokenizer_sha256']:
        raise RuntimeError('Tokenizer provenance mismatch')
    work = root / '.parallel-tokenization' / str(time.time_ns())
    work.mkdir(parents=True)
    interrupted = work / 'interrupted'
    interrupted.mkdir()
    for partial in output.glob('*.partial'):
        partial.rename(interrupted / partial.name)
    completed = output / '.sources'
    completed.mkdir(exist_ok=True)
    pending = []
    for item in source['files']:
        prefix = Path(item['path']).stem
        marker = completed / (prefix + '.json')
        existing = sorted(output.glob(prefix + '-*/manifest.json'))
        for path in existing:
            meta = json.loads(path.read_text())
            require_encoding_policy(meta, path)
            if (meta['source_revision'], meta['source_sha256'], meta['tokenizer_sha256']) != (
                    source['revision'], item['sha256'], digest):
                raise RuntimeError(f'Existing chunk provenance differs: {path}')
        if marker.exists():
            state = json.loads(marker.read_text())
            require_encoding_policy(state, marker)
            if state['revision'] != source['revision'] or state['sha256'] != item['sha256']:
                raise RuntimeError(f'Completed source provenance differs: {marker}')
            if not existing or json.loads(existing[-1].read_text())['source_rows_end'] != state['documents']:
                raise RuntimeError(f'Completed source chunks missing: {marker}')
            continue
        stage = work / prefix
        stage.mkdir()
        for path in existing:
            link_chunk(path.parent, stage / path.parent.name)
        pending.append((item, stage))
    update_summary(output, source, digest, complete=False)
    print(json.dumps({'phase': 'parallel-preparing', 'workers': args.workers,
                      'threads_per_worker': args.threads, 'remaining_sources': len(pending),
                      'staging': str(work)}), flush=True)
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'))
    jobs = {}
    try:
        while pending or jobs or not source.get('verified') or not source.get('download_complete'):
            source = json.loads((root / 'raw/fineweb-edu-10bt/manifest.json').read_text())
            for item, stage in list(pending):
                path = Path(source.get('data_dir', root / 'raw/fineweb-edu-10bt')) / item['path']
                if len(jobs) >= args.workers:
                    break
                if path.is_file() and path.stat().st_size == item['bytes']:
                    future = pool.submit(worker, item, source, tokenizer_path, digest, info['eos_token_id'],
                                         stage, args.chunk_tokens, args.threads)
                    jobs[future] = (item, stage)
                    pending.remove((item, stage))
            changed = False
            for future, (item, stage) in list(jobs.items()):
                finished = future.done()
                if finished:
                    rows = future.result()  # Raise a worker failure before claiming completion.
                for path in sorted(stage.glob('*/manifest.json')):
                    if path.parent.name.endswith('.partial'):
                        continue
                    final = output / path.parent.name
                    if final.exists():
                        if sha256(final / 'manifest.json') != sha256(path):
                            raise RuntimeError(f'Published chunk changed: {final}')
                        continue
                    temporary = final.with_name(final.name + '.partial')
                    link_chunk(path.parent, temporary)
                    temporary.rename(final)
                    changed = True
                if finished:
                    atomic_json(completed / (Path(item['path']).stem + '.json'),
                        {'revision': source['revision'], 'sha256': item['sha256'], 'documents': rows,
                         'encoding_policy_version': ENCODING_POLICY_VERSION})
                    print(f'Completed source {item["path"]}: {rows:,} documents', flush=True)
                    del jobs[future]
            if changed:
                update_summary(output, source, digest, complete=False)
                summary = json.loads((output / 'manifest.json').read_text())
                print(f'Published corpus total: {summary["train_tokens"]:,} train tokens', flush=True)
            if pending or jobs or not source.get('verified') or not source.get('download_complete'):
                time.sleep(2)
        pool.shutdown(wait=True)
        update_summary(output, source, digest, complete=True)
        print('All source documents tokenized. Parallel corpus is complete.', flush=True)
    except BaseException as error:
        # Stop other writers and preserve all immutable chunks for recovery.
        for process in list((pool._processes or {}).values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        update_summary(output, source, digest, complete=False, status='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
