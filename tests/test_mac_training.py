"""Functional CPU tests for platform migration, not a new benchmark."""
import copy

import pytest
import torch

from microllama.training.engram import build_engram_model
from microllama.training.mac import (install_mps_engram_hash_adapter,
    install_mps_padding_adapter, install_mps_embedding_reduction_adapter,
    portable_adam_state, require_production_parent,
    restore_adam, update, cpu_tree)
from microllama.training.optimizers import build_optimizer


def model():
    return build_engram_model(dict(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=64, pad_token_id=0, eos_token_id=7, bos_token_id=1,
        attention_dropout=0., tie_word_embeddings=True), table_size=31, embedding_dim=4,
        layers=(0, 1), lookup_version='normalized-multihead-v2', hash_heads=4,
        token_id_map=list(range(32)))


CONFIG = dict(learning_rate=.0003, weight_decay=.1, betas=[.9, .95])


@pytest.mark.parametrize('device', ['cpu', pytest.param('mps', marks=pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason='MPS unavailable'))])
def test_ordered_embedding_repeated_ids_and_tied_head(device):
    generator = torch.Generator().manual_seed(874)
    weights = torch.randn(128, 64, generator=generator) * .01
    ids = torch.randint(0, 16, (1, 1024), generator=generator)
    upstream = torch.randn(1, 1024, 64, generator=generator) * .01
    reference = torch.nn.Embedding(128, 64, padding_idx=0)
    reference.weight.data.copy_(weights)
    # Include the separate tied-head contribution, including its PAD row.
    (reference(ids) * upstream).sum().add(reference.weight.square().sum()).backward()
    repeats = []
    for _ in range(3):
        embedding = torch.nn.Embedding(128, 64, padding_idx=0).to(device)
        embedding.weight.data.copy_(weights.to(device))
        install_mps_embedding_reduction_adapter(embedding)
        values = embedding(ids.to(device))
        torch.testing.assert_close(values.cpu(), reference(ids), atol=0, rtol=0)
        (values * upstream.to(device)).sum().add(embedding.weight.square().sum()).backward()
        repeats.append(embedding.weight.grad.cpu())
    for actual in repeats:
        torch.testing.assert_close(actual, reference.weight.grad, atol=1e-7, rtol=1e-6)
        assert torch.equal(actual, repeats[0])
    assert repeats[0][0].abs().sum() > 0  # Tied head PAD gradient retained.


def test_padding_adapter_preserves_tied_head_gradient_and_forward():
    embedding = torch.nn.Embedding(8, 4, padding_idx=0)
    with torch.no_grad():
        embedding.weight[0].fill_(.25)  # Test nonzero PAD forward, too.
    original = copy.deepcopy(embedding)
    holder = torch.nn.Module()
    holder.get_input_embeddings = lambda: embedding
    install_mps_padding_adapter(holder)
    ids = torch.tensor([[0, 1, 0, 3]])
    torch.testing.assert_close(embedding(ids), original(ids), atol=0, rtol=0)
    # A tied output head is a separate contribution; do not zero its PAD row.
    for item in (embedding, original):
        (item(ids).sum() + item.weight.sum() * 2).backward()
    torch.testing.assert_close(embedding.weight.grad, original.weight.grad, atol=0, rtol=0)
    assert torch.equal(embedding.weight.grad[0], torch.full((4,), 2.))


def test_adam_conversion_preserves_moments_groups_and_next_update():
    torch.manual_seed(4)
    first = model()
    optimizer = build_optimizer(first, fused=False, memory_lr_multiplier=5)
    rows = torch.tensor([[2, 3, 7, 8, 9, 0, 10, 11, 12]])
    batch = [(rows[:, :-1], rows[:, 1:])]
    update(first, optimizer, batch, chunk_size=4)
    saved = copy.deepcopy(optimizer.state_dict())
    for group in saved['param_groups']:
        group['fused'] = True
    second = copy.deepcopy(first)
    restored = restore_adam(second, saved, CONFIG)
    assert all(g['fused'] is False and g['foreach'] is False and g['capturable'] is False
               for g in restored.param_groups)
    assert saved['param_groups'][0]['fused'] is True
    assert restored.param_groups[-1]['lr'] == .0003 * 5
    assert restored.param_groups[-1]['weight_decay'] == 0
    assert all(slot['step'].device.type == 'cpu' for slot in restored.state.values())
    update(first, optimizer, batch, chunk_size=4)
    update(second, restored, batch, chunk_size=4)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)


def test_accumulation_uses_actual_masked_targets_and_exact_hash_adapter():
    torch.manual_seed(8)
    first, rows = model(), torch.tensor([[2, 3, 7, 8, 9], [5, 0, 10, 11, 12]])
    second = copy.deepcopy(first)
    install_mps_engram_hash_adapter(second)
    left = build_optimizer(first, fused=False, memory_lr_multiplier=5)
    right = build_optimizer(second, fused=False, memory_lr_multiplier=5)
    a = update(first, left, [(rows[:, :-1], rows[:, 1:])], chunk_size=4)
    b = update(second, right, [(row[None, :-1], row[None, 1:]) for row in rows], chunk_size=4)
    assert a['valid_targets'] == b['valid_targets'] == 5
    assert a['loss'] == pytest.approx(b['loss'], abs=1e-6)
    for x, y in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(x, y, atol=1e-6, rtol=1e-5)


def test_diagnostic_cannot_be_a_production_parent():
    with pytest.raises(ValueError, match='never a production parent'):
        require_production_parent({'kind': 'diagnostic-only'})
    saved = {'state': {}, 'param_groups': [{'fused': True, 'params': [1]}]}
    portable_adam_state(saved)['param_groups'][0]['params'] = [2]
    assert saved['param_groups'][0]['params'] == [1]


def test_version_metadata_round_trips_with_weights_only(tmp_path):
    path = tmp_path / 'state.pt'
    torch.save(cpu_tree({'report': {'torch': torch.__version__}, 'tensor': torch.ones(2)}), path)
    restored = torch.load(path, weights_only=True)
    assert type(restored['report']['torch']) is str
    assert restored['report']['torch'] == str(torch.__version__)
