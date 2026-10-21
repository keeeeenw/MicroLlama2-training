"""Read-only publication verification for CUDA and native MLX evaluation inputs."""
import json
from pathlib import Path

from microllama.data_io.download import sha256


def manifest_path(path):
    path = Path(path)
    return path / ('mlx_manifest.json' if (path / 'mlx_manifest.json').is_file()
                   else 'checkpoint_manifest.json')


def verify_checkpoint(path):
    path = Path(path)
    manifest = manifest_path(path)
    if manifest.name == 'checkpoint_manifest.json':
        from microllama.training.distributed import verify_checkpoint as verify_cuda
        return verify_cuda(path)
    data = json.loads(manifest.read_text())
    required = {'model.safetensors', 'optimizer.safetensors', 'rng.safetensors',
                'state.json', 'config.json', 'engram_config.json', 'run_config.json', 'tokenizer.json'}
    if (path.is_symlink() or data.get('schema') != 'microllama-mac-mlx-v1'
            or data.get('kind') != 'mac-production' or not required <= set(data['files'])):
        raise ValueError('Incomplete or non-production MLX checkpoint')
    for name, expected in data['files'].items():
        file = path / name
        if (Path(name).name != name or file.is_symlink() or not file.is_file()
                or file.stat().st_size != expected['bytes'] or sha256(file) != expected['sha256']):
            raise ValueError(f'MLX checkpoint payload differs: {name}')
    state = json.loads((path / 'state.json').read_text())
    if (any(state.get(k) != data[k] for k in ('schema', 'kind', 'plan_sha256', 'step', 'tokens'))
            or state.get('initial_parent_kind') != 'original-vast-cuda'):
        raise ValueError('MLX checkpoint state or provenance differs')
    return sha256(manifest)


def load_model(path):
    """Use native Torch tensors; MLX assets retain the parent's derived scalar."""
    from microllama.training.engram import load_engram_model
    path = Path(path)
    if manifest_path(path).name != 'mlx_manifest.json':
        return load_engram_model(path)
    from microllama.training.engram import EngramQwen3Config, EngramQwen3ForCausalLM, LOOKUP_V2
    config = EngramQwen3Config.from_pretrained(path, local_files_only=True)
    model = EngramQwen3ForCausalLM.from_pretrained(path, config=config,
        local_files_only=True, torch_dtype=None).cpu()
    if (config.engram_lookup_version == LOOKUP_V2
            and model.model.engram.canonical_ids.tolist() != config.engram_token_id_map):
        raise ValueError('MLX lookup buffer differs from configured token map')
    inherited = json.loads((path / 'engram_config.json').read_text())
    actual = model.engram_metadata()
    differences = {k for k in set(inherited) | set(actual) if inherited.get(k) != actual.get(k)}
    if differences - {'learned_residual_scale'}:
        raise ValueError('MLX structural Engram metadata differs')
    model._evaluation_metadata_receipt = {
        'inherited_learned_residual_scale': inherited['learned_residual_scale'],
        'actual_learned_residual_scale_from_weights': actual['learned_residual_scale'],
        'structural_metadata_exact': True, 'all_weights_strictly_restored': True}
    return model
