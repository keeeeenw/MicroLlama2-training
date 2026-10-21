"""Real packed-reader/Adam/restart integration on tiny fixtures, no network."""
import copy
import json
import time

import numpy as np
import pytest
import torch

from test_mac_staging import corpus, derived, stager
from microllama.training.engram import build_engram_model
from microllama.training.mac import (install_activation_checkpointing, install_mps_engram_hash_adapter,
    install_mps_padding_adapter, install_mps_embedding_reduction_adapter)
from microllama.training.mac_run import (FrozenValidation, TrainingLoop, check_losses,
    publish, read_checkpoint, require_qualification, restore_model, source_bindings, state_hashes)
from microllama.training.mac_run import metadata_coverage
from microllama.training.optimizers import build_optimizer
from microllama.training.mac_attention import install_tiled_attention, tiled_sdpa


def small_model(device='cpu', checkpointing=False):
    model = build_engram_model(dict(vocab_size=8192, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=7, bos_token_id=1,
        attention_dropout=0., tie_word_embeddings=True), table_size=31, embedding_dim=4,
        layers=(0, 1), lookup_version='normalized-multihead-v2', hash_heads=4,
        token_id_map=list(range(8192)))
    model.set_attn_implementation('sdpa')
    install_mps_engram_hash_adapter(model)
    install_mps_padding_adapter(model)
    install_mps_embedding_reduction_adapter(model)
    install_tiled_attention(model)
    if checkpointing:
        install_activation_checkpointing(model)
    return model.to(device)


def fixtures(corpus, device='cpu'):
    catalog, _ = derived(corpus)
    cache = stager(corpus, catalog)
    reader = cache.reader(corpus['state'])
    model = small_model(device, checkpointing=True)
    optimizer = build_optimizer(model, fused=False, memory_lr_multiplier=5)
    for group in optimizer.param_groups:
        group.update(foreach=False, capturable=False)
    total = reader.parent_tokens + sum(reader.targets.values())
    config = dict(sequence_length=4, gradient_accumulation_steps=2, micro_batch_size=1,
        learning_rate=.0003, weight_decay=.1, betas=[.9, .95], grad_clip=1.,
        loss_chunk_size=4, target_tokens=total + 12, eval_every=2, save_every=1)
    spec = {'sequence_length': 4, 'seed': 42, 'blocks': 2, 'minimum_valid_targets': 1,
            'chunk_size': 4, 'tokenizer_sha256': 'tok',
            'sources': {source: {'directory': entry['directory'],
                'chunks': {name: value['manifest_sha256'] for name, value in entry['chunks'].items()}}
                for source, entry in cache.catalog['sources'].items()}}
    for source in cache.catalog['sources']:
        cache.ensure_chunk(source, 'a'); cache.ensure_chunk(source, 'b')
    validator = FrozenValidation(spec)
    state = {'step': 5, 'tokens': total, 'input_positions': total, 'data': reader.state_dict(),
             'config': config, 'validation': spec}
    baseline = {name: value['loss'] for name, value in validator(model).items()}
    return model, optimizer, reader, validator, state, config, baseline


def runner(parts, tmp_path, **changes):
    model, optimizer, reader, validator, state, config, baseline = parts
    options = dict(validation_reference=baseline, max_validation_increase=1.,
        save=lambda m, o, s: publish(tmp_path / 'checkpoints', m, o, s, plan_sha='a' * 64, reserve_bytes=0))
    options.update(changes)
    return TrainingLoop(model, optimizer, reader, validator, state, config, **options)


def test_real_reader_finite_endpoint_and_safe_publication(corpus, tmp_path):
    torch.manual_seed(9)
    parts = fixtures(corpus)
    loop = runner(parts, tmp_path)
    result = loop.run(max_updates=9, max_seconds=30)
    assert result['phase'] == 'completed' and result['updates'] == 2
    assert 0 <= result['tokens'] - parts[5]['target_tokens'] < 8
    saved = read_checkpoint(result['checkpoint'], 'a' * 64)
    assert saved['tokens'] == saved['data']['parent_tokens'] + sum(saved['data']['source_targets'].values())
    assert saved['phase'] == 'completed'
    assert saved['last_validation']['step'] == saved['step']
    assert len(saved['optimizer']['state']) == len(list(parts[0].parameters()))
    assert json.loads((tmp_path / 'checkpoints/latest.json').read_text())['checkpoint'].endswith('00000007')


def test_until_stopped_has_no_wall_update_cap_but_keeps_dataset_endpoint(corpus, tmp_path):
    parts = fixtures(corpus)
    result = runner(parts, tmp_path).run(max_updates=None, max_seconds=None)
    assert result['phase'] == 'completed' and result['updates'] == 2


def test_until_stopped_honors_user_stop_without_advancing(corpus, tmp_path):
    parts = fixtures(corpus)
    result = runner(parts, tmp_path).run(max_updates=None, max_seconds=None, stopped=lambda: True)
    assert result['phase'] == 'paused' and result['updates'] == 0
    assert result['tokens'] == parts[4]['tokens']


def test_offline_coverage_matches_real_reader_without_opening_payloads(corpus):
    reader = corpus['reader']
    assert metadata_coverage(reader.catalog, reader.state_dict(), 190) == reader.coverage(190)


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS GPU unavailable in this process'))])
def test_real_reader_interrupted_restart_exact(corpus, tmp_path, device):
    torch.manual_seed(31)
    if device == 'mps':
        torch.mps.manual_seed(31)
    parts = fixtures(corpus, device)
    first = runner(parts, tmp_path).run(max_updates=1, max_seconds=30)
    saved = read_checkpoint(first['checkpoint'], 'a' * 64)
    # Continue the live model for one step, then independently reload saved step1.
    live_state = {key: value for key, value in saved.items() if key != 'optimizer'}
    live = list(parts)
    live[4] = live_state
    uninterrupted = runner(live, tmp_path / 'uninterrupted').run(max_updates=1, max_seconds=30)
    expected = state_hashes(parts[0], parts[1])
    cursor = parts[2].state_dict()
    model, optimizer = restore_model(first['checkpoint'], saved, torch.device(device), checkpointing=True)
    parts[2].load_state_dict(saved['data'])
    restarted = [model, optimizer, parts[2], parts[3], live_state, parts[5], parts[6]]
    resumed = runner(restarted, tmp_path / 'resumed', initial_checkpoint=first['checkpoint']).run(max_updates=1, max_seconds=30)
    assert state_hashes(model, optimizer) == expected
    assert parts[2].state_dict() == cursor
    assert resumed['tokens'] == uninterrupted['tokens']


def test_interrupted_update_rolls_back_cursor_and_never_publishes(corpus, tmp_path, monkeypatch):
    parts = fixtures(corpus)
    before = copy.deepcopy(parts[2].state_dict())
    published = []
    def fail(*args, **kwargs):
        raise RuntimeError('simulated partial backward failure')
    monkeypatch.setattr('microllama.training.mac_run.update', fail)
    with pytest.raises(RuntimeError, match='partial backward'):
        runner(parts, tmp_path, save=lambda *args: published.append(args)).run(max_updates=1, max_seconds=30)
    assert parts[2].state_dict() == before and published == []


def test_bad_final_validation_never_publishes_completed_state(corpus, tmp_path):
    parts = list(fixtures(corpus))
    parts[5]['target_tokens'] = parts[4]['tokens'] + 1
    count = 0
    def validator(model):
        nonlocal count
        count += 1
        return {name: {'loss': (value if count == 1 else value + 2)} for name, value in parts[6].items()}
    parts[3] = validator
    published = []
    with pytest.raises(RuntimeError, match='validation differs'):
        runner(parts, tmp_path, save=lambda *args: published.append(args)).run(max_updates=1, max_seconds=30)
    assert published == []


def test_resumed_wall_limit_without_update_keeps_existing_checkpoint(corpus, tmp_path):
    parts = list(fixtures(corpus))
    original = parts[3]
    def delayed(model):
        time.sleep(.01)
        return original(model)
    parts[3] = delayed
    result = runner(parts, tmp_path, initial_checkpoint='already-published',
                    save=lambda *args: pytest.fail('Unchanged checkpoint must not be overwritten')).run(
                        max_updates=1, max_seconds=.001)
    assert result['phase'] == 'paused' and result['updates'] == 0
    assert result['checkpoint'] == 'already-published'


def test_validation_rng_is_unchanged_and_changed_files_refused(corpus):
    model, _, _, validator, _, _, _ = fixtures(corpus)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    first, second = validator(model), validator(model)
    assert first == second and torch.equal(torch.get_rng_state(), before)
    path = next(iter(validator.identities.values()))[1][0]
    path.write_bytes(b'bad')
    with pytest.raises(ValueError, match='changed during'):
        validator(model)


def test_corrupt_or_diagnostic_restart_and_mismatched_hardware_refused(corpus, tmp_path):
    parts = fixtures(corpus)
    result = runner(parts, tmp_path).run(max_updates=1, max_seconds=30)
    path = __import__('pathlib').Path(result['checkpoint'])
    with pytest.raises(ValueError, match='schema, kind or plan'):
        read_checkpoint(path, 'b' * 64)
    (path / 'model.safetensors').write_bytes(b'bad')
    with pytest.raises(ValueError, match='payload hash'):
        read_checkpoint(path, 'a' * 64)
    receipt = {'schema': 'microllama-mac-run-v1', 'kind': 'mac8k-qualification', 'passed': True,
        'plan_sha256': 'a', 'hardware': {'chip': 'M1'}, 'implementation': source_bindings(),
        'sequence_length': 128, 'microbatches': 1, 'real_reader_restart_exact': True,
        'full_frozen_validation_pass': True}
    with pytest.raises(ValueError, match='real-data 8K'):
        require_qualification(receipt, 'a', {'chip': 'M1'})
    receipt.update(sequence_length=8192, microbatches=16)
    with pytest.raises(ValueError, match='real-data 8K'):
        require_qualification(receipt, 'a', {'chip': 'M5'})
    require_qualification(receipt, 'a', {'chip': 'M1'})


def test_activation_checkpointing_matches_gradients_and_keeps_ties():
    torch.manual_seed(5)
    first = small_model()
    second = small_model(checkpointing=True)
    second.load_state_dict(first.state_dict())
    rows = torch.tensor([[2, 3, 7, 4, 5, 0, 8, 9]])
    from microllama.training.objective import chunked_loss, target_mask
    for model in (first, second):
        x, y = rows[:, :-1], rows[:, 1:]
        chunked_loss(model, x, y, 3, loss_mask=target_mask(x, y, 7, 0)).backward()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)
    assert second.lm_head.weight is second.model.embed_tokens.weight


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS GPU unavailable in this process'))])
@pytest.mark.parametrize('masked', [False, True])
def test_tiled_full_causal_attention_matches_forward_and_all_gradients(device, masked):
    torch.manual_seed(7)
    values = [torch.randn(2, 4, 31, 8).to(device) for _ in range(3)]
    left = [value.detach().clone().requires_grad_(True) for value in values]
    right = [value.detach().clone().requires_grad_(True) for value in values]
    mask = torch.ones(2, 1, 31, 31, dtype=torch.bool, device=device) if masked else None
    if mask is not None:
        mask[..., 11] = False
    actual = tiled_sdpa(*left, mask=mask, query_tile=7, causal=True, scaling=.25)
    full_mask = torch.ones(31, 31, dtype=torch.bool, device=device).tril()
    if mask is not None:
        full_mask = full_mask & mask
    expected = torch.nn.functional.scaled_dot_product_attention(*right, attn_mask=full_mask,
                                                              dropout_p=0., scale=.25)
    actual.square().sum().backward(); expected.square().sum().backward()
    for first, second in [(actual, expected), *[(a.grad, b.grad) for a, b in zip(left, right)]]:
        torch.testing.assert_close(first, second, atol=3e-5, rtol=3e-4)
