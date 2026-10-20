"""Verified, uncached float32 likelihood inference for a saved grafted reader.

The native model retains its class and tokenizer. Before each full decoder
forward, prepare the same exact-suffix addresses and document-reset mask used
in adaptation. This module never recalibrates, trains, loads a teacher, or
supports cached generation. The public harness owns batching and tokenization.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from ..data_io.download import sha256
from ..training.grafting import GraftMemory, GraftReader
from ..training.xmem_pilot import ReaderInjection


def verify_graft_artifacts(checkpoint, reader_directory, bank_directory):
    """Bind adapter to the exact parent and bank, independent of file location."""
    checkpoint, reader_directory, bank_directory = map(Path, (checkpoint, reader_directory, bank_directory))
    manifest_path = reader_directory / 'reader-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('artifact_type') != 'graft-reader-only':
        raise ValueError('Expected a dependency-bound graft-reader-only artifact')
    reader_path = reader_directory / 'reader-state.pt'
    if reader_path.stat().st_size != manifest['bytes'] or sha256(reader_path) != manifest['sha256']:
        raise ValueError('Graft reader artifact checksum mismatch')
    identity = manifest['identity']
    parent = identity['parent']
    if sha256(checkpoint / 'checkpoint_manifest.json') != parent['manifest_sha256']:
        raise ValueError('Graft reader requires its exact parent checkpoint manifest')
    for name, digest in parent['files'].items():
        if sha256(checkpoint / name) != digest:
            raise ValueError(f'Graft parent file changed: {name}')
    for name, record in identity['donor']['files'].items():
        path = bank_directory / name
        if ('bytes' in record and path.stat().st_size != record['bytes']) or sha256(path) != record['sha256']:
            raise ValueError(f'Graft bank file changed: {name}')
    training_root = Path(__file__).resolve().parents[1] / 'training'
    for source, digest in identity['implementation'].items():
        if sha256(training_root / Path(source).name) != digest:
            raise ValueError(f'Graft training implementation changed: {source}; use the recorded run code')
    if identity['condition'] not in ('pretrained', 'shuffled'):
        raise ValueError('Unknown saved graft condition')
    return {'reader_manifest_sha256':sha256(manifest_path),
        'reader_sha256':manifest['sha256'], 'reader_bytes':manifest['bytes'],
        'reader_step':manifest['step'], 'reader_valid_targets':manifest['new_valid_targets'],
        'condition':identity['condition'], 'parent':parent,
        'bank':identity['donor'], 'trained_implementation':identity['implementation'],
        'reader_directory':str(reader_directory.resolve()), 'bank_directory':str(bank_directory.resolve()),
        'scorer_sha256':sha256(Path(__file__)), 'cached_generation':False,
        'inference_dtype':'float32', 'bank_storage_dtype':'bfloat16',
        'bank_value_conversion':'exact bfloat16 to float32; no recalibration'}


@dataclass
class GraftInference:
    """Own hook lifetimes without changing the native HF model architecture."""
    injection: ReaderInjection
    prepare_handle: object
    metadata: dict

    def close(self):
        self.prepare_handle.remove()
        self.injection.close()


def attach_graft_reader(model, reader, bank, metadata=None):
    """Attach a reader on the native model's CPU or CUDA scoring device."""
    device = next(model.parameters()).device
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Graft benchmark scoring requires CPU or CUDA')
    model.config.use_cache = False
    reader.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    bank.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    model.eval().requires_grad_(False)
    injection = ReaderInjection(model, reader, bank, bank.addressing, layer=4)

    def prepare(module, args, kwargs):
        if kwargs.get('past_key_values') is not None or kwargs.get('use_cache') is True:
            raise ValueError('Graft likelihood evaluation requires full uncached input sequences')
        ids = kwargs.get('input_ids', args[0] if args else None)
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long:
            raise ValueError('Graft scoring requires long input_ids, not inputs_embeds')
        injection.batch(ids)

    handle = model.model.register_forward_pre_hook(prepare, with_kwargs=True)
    metadata = dict(metadata or {}, inference_device=str(device))
    return GraftInference(injection, handle, metadata)


def load_graft_reader(model, checkpoint, reader_directory, bank_directory):
    """Load trained weights strictly, including the saved residual and convolution."""
    metadata = verify_graft_artifacts(checkpoint, reader_directory, bank_directory)
    state = torch.load(Path(reader_directory) / 'reader-state.pt', map_location='cpu', weights_only=True, mmap=True)
    if state['identity'] != json.loads((Path(reader_directory)/'reader-manifest.json').read_text())['identity']:
        raise ValueError('Reader payload identity differs from its manifest')
    settings = state['settings']
    if (settings.get('memory_kind') != 'graft' or not settings['parent_frozen']
            or not settings['bank_frozen'] or settings['injection'] != 'pre-layer4-after-native-engram-layer3'
            or state['step'] != metadata['reader_step'] or state['tokens'] != metadata['reader_valid_targets']):
        raise ValueError('Unsupported graft reader settings or restart step')
    if not all(torch.isfinite(v).all() for v in state['reader'].values() if v.is_floating_point()):
        raise ValueError('Nonfinite graft reader weights')
    bank = GraftMemory.from_directory(Path(bank_directory), Path(checkpoint)/'tokenizer.json')
    if metadata['condition'] == 'shuffled':
        bank.shuffle_rows(settings['shuffled_seed'])
    reader = GraftReader(model.config.hidden_size, bank.d_mem)
    reader.load_state_dict(state['reader'], strict=True)
    if sum(p.numel() for p in reader.parameters()) != settings['reader_parameters']:
        raise ValueError('Graft reader parameter count mismatch')
    return attach_graft_reader(model, reader, bank, metadata)
