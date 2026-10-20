"""Offline Qwen3.8 feature builder; run only in the isolated teacher environment.

Loads embeddings and the first 24 text blocks in original BF16, omitting the
vision tower, LM head and later blocks. Identity final norm exposes the exact
intermediate representation, not a newly normalized truncated-model output.
Required official HF shards are downloaded one at a time into an owned scratch
directory, SHA-verified, copied into CUDA parameters and removed. Shared HF
caches and user checkpoints are never modified. Published banks retain receipts.

See docs/experiments.md for setup, bounded construction and recipient training.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from importlib.metadata import version
from pathlib import Path
import signal
import shutil
import subprocess
import sys

import torch
from torch import nn

from ..data_io.download import sha256
from ..data_io.transfer import atomic_json


TEACHER_REPO = 'Qwen/Qwen3.8-27B'
TEACHER_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SOURCE_LAYER = 24
PREFIX = 'model.language_model.'
DISK_RESERVE = 20 * 2**30


def prepare_snapshot(directory):
    """Pin metadata/tokenizer only; the weight loader controls temporary storage."""
    from huggingface_hub import HfApi, hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(TEACHER_REPO, revision=TEACHER_REVISION, files_metadata=True)
    if info.sha != TEACHER_REVISION:
        raise RuntimeError('Teacher revision did not resolve to its pinned commit')
    files = {f.rfilename: {'bytes': f.size, 'sha256': f.lfs.sha256 if f.lfs else None}
             for f in info.siblings if f.rfilename.endswith('.safetensors')}
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json'):
        hf_hub_download(TEACHER_REPO, name, revision=TEACHER_REVISION, local_dir=directory)
        files[name] = {'bytes': (directory/name).stat().st_size, 'sha256': sha256(directory/name)}
    atomic_json(directory/'source.json', {'repo': TEACHER_REPO, 'revision': TEACHER_REVISION, 'files': files})


def prefix_config(text_config, source_layer=SOURCE_LAYER):
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    values = copy.deepcopy(text_config)
    if not 1 <= source_layer < values['num_hidden_layers']:
        raise ValueError('Extract an intermediate block, before the final model norm')
    values['num_hidden_layers'] = source_layer
    values['layer_types'] = values['layer_types'][:source_layer]
    config = Qwen3_5TextConfig(**values)
    config._attn_implementation = 'sdpa'
    return config


def empty_prefix(text_config, source_layer=SOURCE_LAYER):
    """Allocate metadata on meta; do not randomly initialize 10B real parameters."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    with torch.device('meta'):
        model = Qwen3_5TextModel(prefix_config(text_config, source_layer)).to(dtype=torch.bfloat16)
    model.norm = nn.Identity()
    return model


@torch.no_grad()
def load_teacher_prefix(directory, scratch, report):
    """Materialize every prefix parameter exactly once from verified official shards."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    receipt = json.loads((directory/'source.json').read_text())
    config = json.loads((directory/'config.json').read_text())
    if (receipt['repo'] != TEACHER_REPO or receipt['revision'] != TEACHER_REVISION
            or config['model_type'] != 'qwen3_5' or config['text_config']['num_hidden_layers'] != 64):
        raise ValueError('Unexpected teacher identity or architecture')
    for name in ('config.json', 'model.safetensors.index.json', 'tokenizer.json', 'tokenizer_config.json'):
        if sha256(directory/name) != receipt['files'][name]['sha256']:
            raise RuntimeError(f'Teacher metadata checksum mismatch: {name}')
    weight_map = json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    model = empty_prefix(config['text_config'])
    expected = {PREFIX+name: name for name in model.state_dict()}
    if not expected or set(expected) - weight_map.keys():
        raise RuntimeError('Official weight index is missing required text-prefix tensors')
    size = sum(p.numel()*p.element_size() for p in model.parameters())
    free, _ = torch.cuda.mem_get_info()
    if size + 3*2**30 > free:
        raise RuntimeError('Teacher prefix cannot fit with its 3GiB activation reserve')
    # to_empty would leave rotary buffers uninitialized. Recompute them from the
    # official config in FP32, as in the full model's BF16 from_pretrained path.
    model.to_empty(device='cuda')
    model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=model.config).cuda()
    shards = sorted({weight_map[name] for name in expected})
    scratch.mkdir(parents=True, exist_ok=False)
    loaded = set()
    verified = {}
    for number, shard in enumerate(shards, start=1):
        record = receipt['files'][shard]
        if not record['sha256'] or '/' in shard:
            raise RuntimeError('Official shard has no checksum or unsafe path')
        if shutil.disk_usage(scratch).free < DISK_RESERVE + record['bytes'] + 1024*2**20:
            raise RuntimeError('Teacher shard would consume the 20GiB disk reserve')
        report('downloading_teacher', shard=shard, shard_number=number, shards=len(shards), prefix_bytes=size)
        path = Path(hf_hub_download(TEACHER_REPO, shard, revision=TEACHER_REVISION, local_dir=scratch))
        if path.stat().st_size != record['bytes'] or sha256(path) != record['sha256']:
            raise RuntimeError(f'Teacher weight checksum mismatch: {shard}')
        with safe_open(path, framework='pt', device='cpu') as handle:
            for official, local in expected.items():
                if weight_map[official] != shard:
                    continue
                value = handle.get_tensor(official)
                parameter = model.get_parameter(local)
                if value.shape != parameter.shape or not value.is_floating_point() or not torch.isfinite(value).all():
                    raise RuntimeError(f'Invalid teacher tensor: {official}')
                # Preserve the checkpoint's FP32 state-space scalars where needed.
                if local.endswith(('A_log', 'dt_bias')):
                    parameter.data = value.to(device='cuda', dtype=torch.float32)
                else:
                    parameter.copy_(value)
                loaded.add(official)
                del value
        # Only this builder's new scratch file is removed, after the handle closes.
        path.unlink()
        verified[shard] = record
        report('loaded_teacher_shard', shard=shard, loaded_tensors=len(loaded), expected_tensors=len(expected))
    # Finiteness was checked on CPU before each copy; avoid a multi-GiB CUDA
    # temporary for another isfinite over the very large embedding matrix.
    if loaded != set(expected) or any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Teacher prefix materialization is incomplete or nonfinite')
    model.eval().requires_grad_(False)
    identity = {'repo': TEACHER_REPO, 'revision': TEACHER_REVISION,
                'source_layer': SOURCE_LAYER, 'original_text_layers': 64,
                'loaded_parameters': sum(p.numel() for p in model.parameters()),
                'loaded_bytes': sum(p.numel()*p.element_size() for p in model.parameters()),
                'loaded_tensors': len(loaded), 'verified_shards': verified,
                'final_norm': 'identity', 'dtype': 'bfloat16-with-FP32-state-scalars',
                'transformers_version': version('transformers'), 'torch_version': torch.__version__,
                'loader_sha256': sha256(Path(__file__))}
    report('teacher_ready', **{k: v for k, v in identity.items() if k != 'verified_shards'})
    return model, identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--token-dir', type=Path, required=True)
    parser.add_argument('--teacher-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detach', action='store_true', help='Launch a durable background builder and return its PID')
    parser.add_argument('--resume', action='store_true', help='Reuse checksum-verified encoded batches from the same partial bank')
    args = parser.parse_args()
    if args.detach:
        output = args.output.resolve()
        if output.exists() or (output.with_name(output.name+'.partial').exists() and not args.resume):
            parser.error('Choose a fresh bank output, or explicitly resume a saved encoding plan')
        output.parent.mkdir(parents=True,exist_ok=True)
        command = [sys.executable,'-m','microllama.training.graft_teacher',
            '--checkpoint',str(args.checkpoint.resolve()),'--token-dir',str(args.token_dir.resolve()),
            '--teacher-dir',str(args.teacher_dir.resolve()),'--output',str(output)]
        if args.resume:
            command.append('--resume')
        log_path = output.with_name(output.name+'.builder.log')
        with log_path.open('a') as log:
            child = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True)
        record = {'pid':child.pid,'command':command,'log':str(log_path),
            'started_utc':datetime.now(timezone.utc).isoformat(),'resume':args.resume,
            'loader_sha256':sha256(Path(__file__))}
        atomic_json(output.with_name(output.name+'.launch.json'),record)
        atomic_json(output.with_name(output.name+'.worker-status.json'),
            {'phase':'launching','pid':child.pid,'started_utc':record['started_utc']})
        print(json.dumps(record),flush=True)
        return
    from .gpu import gpu_lock, idle_gpu
    from .grafting import build_bank
    import os
    status_path = args.output.resolve().with_name(args.output.name+'.worker-status.json')
    def status(phase, **extra):
        atomic_json(status_path,{'phase':phase,'pid':os.getpid(),
            'updated_utc':datetime.now(timezone.utc).isoformat(),**extra})
    def interrupted(number, frame):
        raise InterruptedError(f'Bank builder stopped by signal {number}')
    def timed_out(number, frame):
        raise TimeoutError('Offline bank build exceeded its one-hour attempt budget')
    try:
        status('running')
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        signal.signal(signal.SIGALRM,timed_out)
        with gpu_lock():
            idle_gpu()
            torch.set_num_threads(2)
            signal.alarm(3600)
            prepare_snapshot(args.teacher_dir.resolve())
            build_bank(args.checkpoint.resolve(), args.token_dir.resolve(),
                       args.teacher_dir.resolve(), args.output.resolve(),resume=args.resume)
            signal.alarm(0)
        status('completed')
    except BaseException as error:
        status('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    main()
