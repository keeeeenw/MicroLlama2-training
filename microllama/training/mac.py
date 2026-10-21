"""Explicit CUDA-to-MPS handoff; diagnostic outputs cannot resume production.

The historical four-rank CUDA runner and its admission policy stay unchanged.
FP32 parameters/moments, CPU int64 lookup arithmetic and an unfused AdamW are
intentional platform changes. Data order and the masked objective are preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import time
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from .engram import EngramMemory, EngramState, load_engram_model
from .objective import chunked_loss, normalize_accumulated_gradients, target_mask
from .optimizers import build_optimizer

FINAL_MANIFEST = '9968e0d8db74a6618750ca38c71e68332c3947921f8dd75a591415afd965840c'
FINAL_STEP = 381824
FINAL_TOKENS = 50_000_054_177
SCHEMA = 'microllama-mac-continuation-v1'


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def install_mps_engram_hash_adapter(model):
    """Keep original signed-int64 hash and suffix operations exactly on CPU.

    Only non-differentiable token IDs/masks cross devices; all memory table and
    gate computation and their gradients stay on MPS. No model source changes.
    """
    memory = model.model.engram
    if getattr(memory, '_mac_integer_adapter', False):
        return
    # A shallow mirror shares only table-size metadata, never copies weights.
    mirror = copy.copy(memory)
    mirror._buffers = dict(memory._buffers)
    if memory.revised:
        mirror._buffers['canonical_ids'] = memory.canonical_ids.detach().cpu().clone()

    def indices(self, input_ids, state):
        values = EngramMemory._indices(mirror, input_ids.cpu(), EngramState(state.history_ids.cpu()))
        return tuple(value.to(input_ids.device) for value in values)

    def next_state(self, input_ids, state):
        value = EngramMemory._next_state(mirror, input_ids.cpu(), EngramState(state.history_ids.cpu()))
        return EngramState(value.history_ids.to(input_ids.device))

    memory._indices = types.MethodType(indices, memory)
    memory._next_state = types.MethodType(next_state, memory)
    memory._mac_integer_adapter = True


def install_mps_padding_adapter(model):
    """Preserve Embedding padding semantics on the qualified MPS backend.

    torch2.9.1/M1 testing found native embedding backward accumulates PAD input
    gradients. Mask only this embedding lookup's gradient, never the tied LM
    head parameter gradient. Preserve the original PAD forward value exactly.
    """
    embedding = model.get_input_embeddings()
    if getattr(embedding, '_mac_padding_adapter', False):
        return
    if embedding.padding_idx is None or embedding.max_norm is not None or embedding.sparse:
        raise ValueError('Unexpected embedding configuration for this adapter')

    def forward(self, ids):
        values = F.embedding(ids, self.weight, padding_idx=None,
                             scale_grad_by_freq=self.scale_grad_by_freq, sparse=False)
        return torch.where(ids.ne(self.padding_idx).unsqueeze(-1), values, values.detach())

    embedding.forward = types.MethodType(forward, embedding)
    embedding._mac_padding_adapter = True


class _OrderedEmbedding(torch.autograd.Function):
    """MPS lookup, explicit CPU row reduction for reproducible embedding grads.

    Repeated IDs in torch2.9.1/M1 embedding backward use nondeterministic
    floating-point atomics even when deterministic algorithms are requested.
    Only the gather gradient reduction is on CPU; forward lookup, parameters,
    Adam moments and all other differentiable model work remain on MPS.
    """
    @staticmethod
    def forward(ctx, weight, ids, padding_idx):
        ctx.save_for_backward(ids.detach().cpu())
        ctx.shape, ctx.device, ctx.dtype, ctx.padding_idx = weight.shape, weight.device, weight.dtype, padding_idx
        return F.embedding(ids, weight, padding_idx=None)

    @staticmethod
    def backward(ctx, upstream):
        (ids,) = ctx.saved_tensors
        ids = ids.reshape(-1)
        values = upstream.detach().cpu().reshape(-1, ctx.shape[1])
        if ctx.padding_idx is not None:
            keep = ids.ne(ctx.padding_idx)
            ids, values = ids[keep], values[keep]
        gradient = torch.zeros(ctx.shape, dtype=ctx.dtype, device='cpu')
        gradient.index_add_(0, ids, values)
        return gradient.to(ctx.device), None, None


def install_mps_embedding_reduction_adapter(model):
    """Retain every embedding's forward and PAD semantics; order its reduction."""
    for embedding in model.modules():
        if not isinstance(embedding, torch.nn.Embedding) or getattr(embedding, '_mac_ordered_embedding', False):
            continue
        if embedding.max_norm is not None or embedding.sparse or embedding.scale_grad_by_freq:
            raise ValueError('Ordered embedding adapter requires the unchanged dense lookup recipe')
        def forward(self, ids):
            return _OrderedEmbedding.apply(self.weight, ids, self.padding_idx)
        embedding.forward = types.MethodType(forward, embedding)
        embedding._mac_ordered_embedding = True


def portable_adam_state(saved):
    """Copy metadata before load: saved fused=True would override construction."""
    groups = []
    for original in saved['param_groups']:
        group = {**original, 'params': list(original['params'])}
        group.update(fused=False, foreach=False, capturable=False, differentiable=False)
        groups.append(group)
    return {'state': saved['state'], 'param_groups': groups}


def restore_adam(model, saved, config):
    optimizer = build_optimizer(model, adam_lr=config['learning_rate'],
                                weight_decay=config['weight_decay'], fused=False,
                                memory_lr_multiplier=5)
    expected = optimizer.state_dict()['param_groups']
    ids = [p for group in saved['param_groups'] for p in group['params']]
    if len(set(ids)) != len(ids) or set(saved['state']) != set(ids):
        raise ValueError('Every unique trainable parameter requires its saved Adam state')
    if len(expected) != len(saved['param_groups']) or any(
        len(a['params']) != len(b['params']) for a, b in zip(expected, saved['param_groups'])
    ):
        raise ValueError('Adam parameter group shape/order differs')
    if [g.get('group_name') for g in expected] != [g.get('group_name') for g in saved['param_groups']]:
        raise ValueError('Adam memory group identity differs')
    for index, group in enumerate(saved['param_groups']):
        memory = group.get('group_name') == 'memory_tables'
        if (group['lr'] != config['learning_rate'] * (5 if memory else 1)
                or tuple(group['betas']) != tuple(config['betas'])
                or group['eps'] != 1e-8
                or group.get('amsgrad', False) or group.get('maximize', False)
                or group['weight_decay'] != (0 if memory else expected[index]['weight_decay'])):
            raise ValueError('Saved Adam recipe differs from the held-rate checkpoint')
    optimizer.load_state_dict(portable_adam_state(saved))
    for parameter, slot in optimizer.state.items():
        if set(slot) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Incomplete or unexpected Adam slots')
        slot['step'] = slot['step'].detach().cpu()
        for key in ('exp_avg', 'exp_avg_sq'):
            if slot[key].shape != parameter.shape or slot[key].dtype != torch.float32:
                raise ValueError('Adam moment shape/dtype mismatch')
        if any(not bool(torch.isfinite(value).all()) for value in slot.values()):
            raise ValueError('Nonfinite restored Adam state')
    return optimizer


def install_activation_checkpointing(model):
    """Recompute decoder layers, leaving the frozen Engram implementation intact.

    This is an explicit memory policy in the Mac plan. No cache or dropout is
    allowed; the same masked objective and parameter gradients are retained.
    """
    from torch.utils.checkpoint import checkpoint
    if model.config.attention_dropout != 0:
        raise ValueError('Mac checkpointing qualification assumes zero attention dropout')
    for layer in model.model.layers:
        if getattr(layer, '_mac_checkpointed', False):
            continue
        original = layer.forward

        def forward(self, *args, _original=original, **kwargs):
            if self.training and torch.is_grad_enabled():
                if kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
                    raise ValueError('Activation checkpointing requires cache-free training')
                return checkpoint(_original, *args, use_reentrant=False,
                                  preserve_rng_state=True, **kwargs)
            return _original(*args, **kwargs)

        layer.forward = types.MethodType(forward, layer)
        layer._mac_checkpointed = True


def load_final(checkpoint, device):
    from .distributed import validate_mixture_checkpoint, verify_checkpoint
    checkpoint = Path(checkpoint)
    if sha256(checkpoint / 'checkpoint_manifest.json') != FINAL_MANIFEST:
        raise ValueError('Only the pinned immutable final50B parent is admitted')
    verify_checkpoint(checkpoint)
    state = torch.load(checkpoint / 'training_state.pt', map_location='cpu', weights_only=True)
    if state['step'] != FINAL_STEP or state['tokens'] != FINAL_TOKENS:
        raise ValueError('Final step/token mismatch')
    validate_mixture_checkpoint(state, state['config'], engram=True, world_size=4)
    if state['data']['parent_tokens'] + sum(state['data']['source_targets'].values()) != FINAL_TOKENS:
        raise ValueError('Final data accounting does not reconcile')
    model = load_engram_model(checkpoint).float()
    model.set_attn_implementation('sdpa')
    install_mps_engram_hash_adapter(model)
    install_mps_padding_adapter(model)
    install_mps_embedding_reduction_adapter(model)
    model.to(device)
    optimizer = restore_adam(model, state['optimizer'], state['config'])
    # Original four CUDA RNG streams remain in parent state/provenance. They
    # cannot be translated into one MPS stream; seed and record a new stream.
    torch.manual_seed(state['config']['seed'])
    if device.type == 'mps':
        torch.mps.manual_seed(state['config']['seed'])
    return model, optimizer, state


def update(model, optimizer, batches, *, grad_clip=1., chunk_size=256, progress=None):
    """One update; normalize summed gradients by all valid masked targets."""
    optimizer.zero_grad(set_to_none=True)
    model.train()
    nll, count = 0., 0
    for index, (x, y) in enumerate(batches):
        mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
        valid = int(mask.sum().cpu())
        if valid < 1:
            raise ValueError('Empty masked microbatch')
        loss = chunked_loss(model, x, y, chunk_size=chunk_size, loss_mask=mask, reduction='sum')
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        nll += float(loss.detach().cpu())  # Python float/CPU64, never MPS FP64.
        count += valid
        if progress:
            progress({'microbatches_done': index + 1, 'accumulated_valid_targets': count,
                      'accumulated_loss': nll / count})
    normalize_accumulated_gradients(model, count)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {'loss': nll / count, 'valid_targets': count, 'grad_norm': float(norm.cpu())}


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(item) for item in value)
    if isinstance(value, str):
        # torch.__version__ is a TorchVersion str subclass; checkpoints must
        # remain loadable with weights_only=True and no extra pickle globals.
        return str(value)
    return value


def publish_diagnostic(path, model, optimizer, parent_state, report):
    path = Path(path)
    if path.exists() or path.with_name(path.name + '.partial').exists():
        raise ValueError('Never overwrite an existing diagnostic checkpoint')
    if shutil.disk_usage(path.parent).free < 10 * 2**30:
        raise RuntimeError('Need checkpoint allowance plus free-space reserve on data drive')
    temporary = path.with_name(path.name + '.partial')
    temporary.mkdir()
    model.save_pretrained(temporary, safe_serialization=True)
    state = {'schema': SCHEMA, 'kind': 'diagnostic-only',
             'parent_manifest_sha256': FINAL_MANIFEST, 'parent_step': FINAL_STEP,
             'parent_tokens': FINAL_TOKENS, 'config': parent_state['config'],
             'data': parent_state['data'], 'optimizer': cpu_tree(optimizer.state_dict()),
             'torch_rng': torch.get_rng_state(), 'mps_rng': torch.mps.get_rng_state(),
             'report': report}
    torch.save(cpu_tree(state), temporary / 'mac_state.pt')
    from ..data_io.mac_staging import payload_files
    payloads = payload_files(temporary, {'config.json', 'generation_config.json', 'model.safetensors',
                                       'engram_config.json', 'mac_state.pt'})
    files = {item.name: {'bytes': item.stat().st_size, 'sha256': sha256(item)}
             for item in payloads.values()}
    write_json(temporary / 'mac_manifest.json', {'schema': SCHEMA, 'kind': 'diagnostic-only',
                                               'files': files})
    temporary.rename(path)
    return path


def require_production_parent(state):
    if state.get('kind') == 'diagnostic-only':
        raise ValueError('A shortened/synthetic diagnostic is never a production parent')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sequence-length', type=int, default=128)
    parser.add_argument('--updates', type=int, choices=(1, 2), default=1)
    parser.add_argument('--microbatches', type=int, choices=(1, 2, 4, 16), default=1)
    parser.add_argument('--probe-batch', type=Path, help='Optional CPU tensor of real pretokenized rows')
    parser.add_argument('--activation-checkpointing', action='store_true',
                        help='Explicit decoder recomputation memory policy')
    parser.add_argument('--attention-tile', type=int, default=0)
    args = parser.parse_args()
    if not 16 <= args.sequence_length <= 8192:
        raise ValueError('Diagnostic sequence length outside supported bounds')
    if not Path('/Volumes/TrainData').is_mount() or not str(args.output.resolve()).startswith('/Volumes/TrainData/'):
        raise ValueError('Large diagnostic output must be on TrainData')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable (sandbox access can hide the GPU)')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Blanket CPU fallback would hide unsupported training operations')
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device('mps')
    torch.mps.set_per_process_memory_fraction(.8)
    started = time.monotonic()
    model, optimizer, state = load_final(args.checkpoint, device)
    if args.activation_checkpointing:
        install_activation_checkpointing(model)
    if args.attention_tile:
        from .mac_attention import install_tiled_attention
        install_tiled_attention(model, args.attention_tile)
    original_data = copy.deepcopy(state['data'])
    # Keep provenance/RNG but release the old CPU moment storage after loading.
    del state['optimizer']
    generator = torch.Generator().manual_seed(1234)
    if args.probe_batch:
        rows = torch.load(args.probe_batch, map_location='cpu', weights_only=True)
        if rows.ndim != 2 or rows.shape[1] < args.sequence_length + 1 or len(rows) < args.microbatches:
            raise ValueError('Insufficient real probe rows')
        rows = rows[:args.microbatches, :args.sequence_length + 1].long()
        source = 'saved real rows; isolated cursor copy'
    else:
        rows = torch.randint(2, model.config.vocab_size, (args.microbatches, args.sequence_length + 1), generator=generator)
        rows[:, 8] = model.config.eos_token_id
        rows[:, 15] = model.config.pad_token_id
        source = 'synthetic functional fixture, never a quality measurement'
    batches = [(row[None, :-1].to(device), row[None, 1:].to(device)) for row in rows]
    measurements = []
    for index in range(args.updates):
        before = time.monotonic()
        measured = update(model, optimizer, batches, grad_clip=state['config']['grad_clip'])
        torch.mps.synchronize()
        measured.update(update=index + 1, seconds=time.monotonic() - before,
                        allocated_bytes=torch.mps.current_allocated_memory(),
                        driver_bytes=torch.mps.driver_allocated_memory())
        measurements.append(measured)
        print(json.dumps(measured), flush=True)
    if state['data'] != original_data:
        raise RuntimeError('Diagnostic mutated the official parent cursor')
    report = {'schema': SCHEMA, 'kind': 'diagnostic-only', 'device': platform.processor(),
              'torch': str(torch.__version__), 'macos': platform.mac_ver()[0],
              'parent_manifest_sha256': FINAL_MANIFEST, 'parent_step': FINAL_STEP,
              'parent_tokens': FINAL_TOKENS, 'source': source, 'precision': 'float32',
              'hash_arithmetic': 'original CPU int64; MPS memory embeddings/gates',
              'padding_backward': 'explicit input-PAD detach; tied LM head remains trainable',
              'sequence_length': args.sequence_length, 'microbatches': args.microbatches,
              'activation_checkpointing': args.activation_checkpointing,
              'attention_query_tile': args.attention_tile,
              'mps_memory_fraction': .8,
              'production_batch_shape_exercised': args.sequence_length == 8192 and args.microbatches == 16,
              'production_training_qualified': False,
              'measurements': measurements, 'seconds_before_publication': time.monotonic() - started,
              'official_tokens_advanced': 0, 'official_data_cursor_changed': False}
    saved = publish_diagnostic(args.output / 'checkpoint', model, optimizer, state, report)
    del model, optimizer, batches
    torch.mps.empty_cache()
    # Reopen the actual exported model/Adam; state comparison happens on CPU to
    # avoid another live copy of all MPS weights and moments.
    exported = torch.load(saved / 'mac_state.pt', map_location='cpu', weights_only=True)
    reloaded = load_engram_model(saved)
    loaded_optimizer = restore_adam(reloaded, exported['optimizer'], exported['config'])
    if exported['data'] != original_data:
        raise RuntimeError('Exported diagnostic changed parent data')
    report.update(publication_reload_pass=True, adam_parameters=len(loaded_optimizer.state),
                  adam_tensor_slots=sum(len(v) for v in loaded_optimizer.state.values()),
                  checkpoint=str(saved), checkpoint_manifest_sha256=sha256(saved / 'mac_manifest.json'))
    write_json(args.output / 'receipt.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
