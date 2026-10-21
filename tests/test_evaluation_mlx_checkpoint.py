"""MLX evaluation admission and strict loading of inherited descriptive metadata."""
import json

import pytest

from scripts.evaluation.checkpoint_identity import verify_checkpoint, load_model
from microllama.data_io.download import sha256


def publication(tmp_path):
    state={'schema':'microllama-mac-mlx-v1','kind':'mac-production','plan_sha256':'f'*64,
           'step':123,'tokens':456,'initial_parent_kind':'original-vast-cuda'}
    for name in ['model.safetensors','optimizer.safetensors','rng.safetensors',
                 'config.json','engram_config.json','run_config.json','tokenizer.json']:
        (tmp_path/name).write_text('{}')
    (tmp_path/'state.json').write_text(json.dumps(state))
    files={p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in tmp_path.iterdir()}
    manifest={k:v for k,v in state.items() if k!='initial_parent_kind'}
    manifest['files']=files
    (tmp_path/'mlx_manifest.json').write_text(json.dumps(manifest))


def test_complete_mlx_publication_verifies(tmp_path):
    publication(tmp_path)
    assert verify_checkpoint(tmp_path)==sha256(tmp_path/'mlx_manifest.json')


@pytest.mark.parametrize('failure',['bytes','state','kind','missing'])
def test_invalid_mlx_publication_refused(tmp_path,failure):
    publication(tmp_path)
    if failure=='bytes':(tmp_path/'model.safetensors').write_text('changed')
    manifest=json.loads((tmp_path/'mlx_manifest.json').read_text())
    if failure=='state':manifest['step']+=1
    if failure=='kind':manifest['kind']='qualification-only'
    if failure=='missing':del manifest['files']['optimizer.safetensors']
    (tmp_path/'mlx_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):verify_checkpoint(tmp_path)


def test_native_model_loads_actual_scalar_and_checks_structural_metadata(tmp_path):
    import torch
    from test_engram import tiny_engram
    model=tiny_engram(revised=True)
    model.save_pretrained(tmp_path)
    metadata=json.loads((tmp_path/'engram_config.json').read_text())
    with torch.no_grad():model.model.engram.residual_scale.fill_(.25)
    model.save_pretrained(tmp_path)
    (tmp_path/'engram_config.json').write_text(json.dumps(metadata))
    (tmp_path/'mlx_manifest.json').write_text('{}')
    restored=load_model(tmp_path)
    assert restored.model.engram.residual_scale.item()==.25
    assert restored._evaluation_metadata_receipt['structural_metadata_exact']
    metadata['layers_zero_based']=[123]
    (tmp_path/'engram_config.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='structural'):load_model(tmp_path)
