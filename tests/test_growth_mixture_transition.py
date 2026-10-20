"""Explicit three-source continuation without replay or historical reweighting."""
import copy
import json

import numpy as np
import pytest

from microllama.data_io.download import sha256
from microllama.data_io.growth_reader import (
    MODE, GrowthExhausted, MixtureTokenBlocks, digest, load_catalog,
    publish_catalog, usable_targets,
)
from microllama.data_io.transfer import atomic_json


def chunk(directory, name, source, rows, start):
    folder = directory / name
    folder.mkdir(parents=True)
    metadata = {'source_revision': source, 'tokenizer_sha256': 'tok', 'splits': {}}
    for split in ('train', 'validation'):
        path = folder / f'{split}.bin'
        path.write_bytes(np.arange(start, start + rows * 4 + 1, dtype='<u2').tobytes())
        metadata['splits'][split] = {'tokens': rows * 4 + 1, 'sha256': sha256(path)}
    atomic_json(folder / 'manifest.json', metadata)
    return {'manifest_sha256': sha256(folder / 'manifest.json'),
            'valid_targets': usable_targets(folder / 'train.bin', 4)}


def parent_fixture(tmp_path, *, weights=(.6, .4), rows=250, consumed=100):
    legacy = tmp_path / 'legacy.json'
    atomic_json(legacy, {'chunks': {'old': {'manifest_sha256': 'old-sha'}}})
    recipe = {'sources': {s: {'weight': w, 'revision': s, 'filter': 'frozen'}
                          for s, w in zip(('edu', 'dclm'), weights)},
              'sequence_length': 4, 'seed': 42, 'eos': 7, 'pad': 0,
              'tokenizer_sha256': 'tok',
              'legacy': {'inventory': str(legacy), 'inventory_sha256': sha256(legacy),
                         'root': str(tmp_path / 'old')}}
    value = {'schema_version': 1, 'mode': MODE, 'recipe': recipe, 'recipe_id': digest(recipe),
             'initial_tokens': 100, 'target_tokens': 1000, 'status': 'preparing', 'complete': False,
             'sources': {s: {'directory': str(tmp_path / s),
                             'chunks': {'a': chunk(tmp_path / s, 'a', s, 10, start),
                                        'b': chunk(tmp_path / s, 'b', s, rows - 10, start + 1000)}}
                         for s, start in [('edu', 1000), ('dclm', 4000)]}}
    directory = tmp_path / 'parent-catalog'
    publish_catalog(directory, value)
    reader = MixtureTokenBlocks(directory, 4)
    reader.migrate({'mode': 'growing', 'sequence_length': 4, 'seed': 42,
                    'known': {'old': 'old-sha'}, 'cursor': 88, 'epoch': 1,
                    'current': 'old', 'consumed': []}, 100)
    reader.next_batch(consumed)
    return directory, reader.state_dict()


def child_fixture(tmp_path, parent, *, weights=(.4, .35, .25), rows=250,
                  complete=False, mutate=None):
    value = copy.deepcopy(load_catalog(parent))
    for s, w in zip(('edu', 'dclm', 'cosmo'), weights):
        if s == 'cosmo':
            value['recipe']['sources'][s] = {'weight': w, 'revision': s, 'filter': 'science-v1'}
            value['sources'][s] = {'directory': str(tmp_path / s), 'chunks': {}}
            if rows:
                value['sources'][s]['chunks']['a'] = chunk(tmp_path / s, 'a', s, rows, 7000)
        else:
            value['recipe']['sources'][s]['weight'] = w
    value.update(complete=complete, status='complete' if complete else 'preparing')
    if mutate:
        mutate(value)
    value['recipe_id'] = digest(value['recipe'])
    directory = tmp_path / 'child-catalog'
    publish_catalog(directory, value)
    return directory


def migrate(reader, state, parent, checksum=None):
    checksum = checksum or json.loads((parent / 'manifest.json').read_text())['sha256']
    reader.migrate_mixture(state, parent, parent_catalog_sha256=checksum)


def test_prospective_weights_preserved_cursors_and_fresh_restart(tmp_path):
    parent, state = parent_fixture(tmp_path)
    before = copy.deepcopy(state)
    child = child_fixture(tmp_path, parent)
    reader = MixtureTokenBlocks(child, 4)
    migrate(reader, state, parent)
    migrated = reader.state_dict()
    assert state == before
    for source in ('edu', 'dclm'):
        assert migrated['sources'][source] == state['sources'][source]
        assert migrated['sources'][source]['consumed'] == ['a']
        assert migrated['source_targets'][source] == state['source_targets'][source]
    assert migrated['parent_tokens'] == state['parent_tokens'] == 100
    assert migrated['retired_reader'] == state['retired_reader']
    assert migrated['source_targets']['cosmo'] == 0
    assert migrated['policy_origin'] == {**state['source_targets'], 'cosmo': 0}
    assert migrated['mixture_transition']['starting_tokens'] == 500
    # The new source starts at its prospective share, not a historical catch-up.
    batch = reader.next_batch(100)
    assert batch[0, 0] < 4000  # First choice is Edu (.4), not historically absent Cosmo.
    deltas = {s: reader.targets[s] - migrated['policy_origin'][s] for s in reader.weights}
    assert sum(deltas.values()) == 400
    assert all(abs(deltas[s] - 400 * w) <= 4 for s, w in reader.weights.items())
    saved = json.loads(json.dumps(reader.state_dict()))
    expected = reader.next_batch(17)
    restarted = MixtureTokenBlocks(child, 4)
    restarted.load_state_dict(saved)
    np.testing.assert_array_equal(restarted.next_batch(17), expected)
    assert restarted.state_dict() == reader.state_dict()
    assert reader.coverage(1000)['remaining_required_targets'] == 1000 - 100 - sum(reader.targets.values())


def test_legacy_state_without_origin_has_identical_next_batches(tmp_path):
    parent, state = parent_fixture(tmp_path, consumed=1)
    assert 'policy_origin' not in state and 'mixture_transition' not in state
    a, b = MixtureTokenBlocks(parent, 4), MixtureTokenBlocks(parent, 4)
    a.load_state_dict(state)
    b.load_state_dict(copy.deepcopy(state))
    # Original cumulative-deficit policy picks DCLM after the first Edu row.
    # Resetting the origin on legacy load would incorrectly pick Edu again.
    batch = a.next_batch(31)
    assert batch[0, 0] >= 4000
    np.testing.assert_array_equal(batch, b.next_batch(31))
    assert a.state_dict() == b.state_dict()
    assert 'policy_origin' not in a.state_dict()
    child = MixtureTokenBlocks(child_fixture(tmp_path, parent), 4)
    with pytest.raises(ValueError, match='recipe_id'):
        child.load_state_dict(state)


@pytest.mark.parametrize('change', [
    lambda v: v['recipe']['sources']['edu'].update(revision='changed'),
    lambda v: v['recipe']['sources']['dclm'].update(filter='changed'),
    lambda v: v['sources']['edu'].update(directory='/different/source'),
    lambda v: v['sources']['edu']['chunks']['a'].update(valid_targets=999),
    lambda v: v['recipe'].update(pad=1),
    lambda v: v['recipe']['legacy'].update(inventory_sha256='changed'),
    lambda v: v.update(initial_tokens=101),
    lambda v: v.update(target_tokens=1001),
])
def test_changed_pins_inventory_controls_and_budget_rejected_atomically(tmp_path, change):
    parent, state = parent_fixture(tmp_path)
    reader = MixtureTokenBlocks(child_fixture(tmp_path, parent, mutate=change), 4)
    before = reader.state_dict()
    with pytest.raises(ValueError, match='Mixture migration changed'):
        migrate(reader, state, parent)
    assert reader.state_dict() == before


@pytest.mark.parametrize('change', [
    lambda s: s.update(recipe_id='unbound'),
    lambda s: s['sources']['edu']['known'].update(a='changed'),
    lambda s: s['sources']['edu'].update(cursor=-1),
    lambda s: s['sources']['edu'].update(cursor=True),
    lambda s: s['sources']['edu'].update(cursor=10000),
    lambda s: s['sources']['edu'].update(consumed=['a', 'a']),
    lambda s: s['source_targets'].update(edu=-1),
    lambda s: s.update(policy_origin={'edu': 999999, 'dclm': 0}),
])
def test_invalid_parent_state_rejected_without_mutating_reader(tmp_path, change):
    parent, state = parent_fixture(tmp_path)
    reader = MixtureTokenBlocks(child_fixture(tmp_path, parent), 4)
    before = reader.state_dict()
    change(state)
    with pytest.raises(ValueError):
        migrate(reader, state, parent)
    assert reader.state_dict() == before


def test_bound_checksum_and_corrupt_active_payload_rejected(tmp_path):
    parent, state = parent_fixture(tmp_path)
    reader = MixtureTokenBlocks(child_fixture(tmp_path, parent), 4)
    before = reader.state_dict()
    with pytest.raises(ValueError, match='snapshot checksum'):
        migrate(reader, state, parent, '0' * 64)
    for invalid in (None, '', 'unbound', 'x' * 64):
        with pytest.raises(ValueError, match='bound parent catalog SHA'):
            reader.migrate_mixture(state, parent, parent_catalog_sha256=invalid)
    path = tmp_path / 'edu' / state['sources']['edu']['current'] / 'train.bin'
    path.write_bytes(b'x' * path.stat().st_size)
    with pytest.raises(ValueError, match='Corrupt'):
        migrate(reader, state, parent)
    assert reader.state_dict() == before


def test_repeated_migration_cannot_reset_used_destination(tmp_path):
    parent, state = parent_fixture(tmp_path)
    reader = MixtureTokenBlocks(child_fixture(tmp_path, parent), 4)
    migrate(reader, state, parent)
    reader.next_batch(10)
    before = reader.state_dict()
    with pytest.raises(ValueError, match='unused destination'):
        migrate(reader, state, parent)
    assert reader.state_dict() == before


def test_direct_invalid_restore_keeps_live_cursor_and_maps(tmp_path):
    parent, state = parent_fixture(tmp_path)
    reader = MixtureTokenBlocks(parent, 4)
    reader.load_state_dict(state)
    before = reader.state_dict()
    bad = copy.deepcopy(state)
    bad['sources']['edu']['cursor'] = 10000
    with pytest.raises(ValueError, match='cursor'):
        reader.load_state_dict(bad)
    assert reader.state_dict() == before
    control = MixtureTokenBlocks(parent, 4)
    control.load_state_dict(state)
    np.testing.assert_array_equal(reader.next_batch(13), control.next_batch(13))


@pytest.mark.parametrize('change', [
    lambda s: s.pop('mixture_transition'),
    lambda s: s.update(mixture_transition=None),
    lambda s: s['mixture_transition'].update(starting_tokens=501),
    lambda s: s['mixture_transition'].update(starting_tokens=True),
    lambda s: s['mixture_transition'].update(parent_catalog_sha256=None),
    lambda s: s['mixture_transition'].update(parent_recipe_id='unbound'),
    lambda s: s.pop('policy_origin'),
    lambda s: s.update(policy_origin=None),
])
def test_malformed_transition_binding_rejected_before_restore(tmp_path, change):
    parent, state = parent_fixture(tmp_path)
    reader = MixtureTokenBlocks(child_fixture(tmp_path, parent), 4)
    migrate(reader, state, parent)
    before = reader.state_dict()
    bad = copy.deepcopy(before)
    change(bad)
    with pytest.raises(ValueError, match='transition|policy origin'):
        reader.load_state_dict(bad)
    assert reader.state_dict() == before


def test_wait_and_exhaustion_restore_full_transition_state(tmp_path):
    parent, state = parent_fixture(tmp_path)
    child = child_fixture(tmp_path, parent, rows=0)
    reader = MixtureTokenBlocks(child, 4)
    migrate(reader, state, parent)
    before = reader.state_dict()
    assert reader.try_batch(20) is None
    assert reader.state_dict() == before
    value = load_catalog(child)
    value['sources']['cosmo']['chunks']['a'] = chunk(tmp_path / 'cosmo', 'a', 'cosmo', 1, 7000)
    publish_catalog(child, value)
    reader._refresh()
    before = reader.state_dict()
    value.update(complete=True, status='complete')
    publish_catalog(child, value)
    with pytest.raises(GrowthExhausted, match='No unconsumed cosmo'):
        reader.next_batch(20)
    assert reader.state_dict() == before
    with pytest.raises(ValueError, match='cannot cover'):
        reader.coverage(1000)


def test_corrupt_future_source_rolls_back_then_repaired_batch_matches_restart(tmp_path):
    parent, state = parent_fixture(tmp_path)
    child = child_fixture(tmp_path, parent)
    reader = MixtureTokenBlocks(child, 4)
    migrate(reader, state, parent)
    before = reader.state_dict()
    path = tmp_path / 'cosmo' / 'a' / 'train.bin'
    original = path.read_bytes()
    path.write_bytes(b'x' * len(original))
    with pytest.raises(ValueError, match='Corrupt'):
        reader.next_batch(20)
    assert reader.state_dict() == before
    path.write_bytes(original)
    control = MixtureTokenBlocks(child, 4)
    control.load_state_dict(before)
    np.testing.assert_array_equal(reader.next_batch(20), control.next_batch(20))
    assert reader.state_dict() == control.state_dict()


def test_coverage_only_requires_future_weighted_budget_and_counter_reconciliation(tmp_path):
    parent, state = parent_fixture(tmp_path, weights=(.5, .5), rows=110, consumed=200)
    child = child_fixture(tmp_path, parent, weights=(.4, .4, .2), rows=5, complete=True)
    reader = MixtureTokenBlocks(child, 4)
    migrate(reader, state, parent)
    coverage = reader.coverage(1000)
    assert state['source_targets'] == {'edu': 400, 'dclm': 400}
    assert coverage['remaining_required_targets'] == 100
    assert coverage['balanced_unconsumed_targets'] == 100
    # Every partial checkpoint still has adequate future coverage, including
    # after a source has already supplied its entire quota.
    for prefix in range(1, 26):
        reader.next_batch(1)
        partial = reader.coverage(1000)
        assert partial['remaining_required_targets'] == 100 - prefix * 4
        assert partial['balanced_unconsumed_targets'] == 100 - prefix * 4
    assert reader.parent_tokens + sum(reader.targets.values()) == 1000
    assert reader.coverage(1000)['remaining_required_targets'] == 0
    with pytest.raises(ValueError, match='outside'):
        reader.coverage(1001)
    reader.targets['edu'] += 100
    with pytest.raises(ValueError, match='outside|Consumed'):
        reader.coverage(1000)


@pytest.mark.parametrize('rows, message', [(3, 'cannot cover'), (0, 'exhausted')])
def test_complete_acquisition_rejects_short_or_exhausted_new_source(tmp_path, rows, message):
    parent, state = parent_fixture(tmp_path, weights=(.5, .5), rows=110, consumed=200)
    child = child_fixture(tmp_path, parent, weights=(.4, .4, .2), rows=rows, complete=True)
    reader = MixtureTokenBlocks(child, 4)
    before = reader.state_dict()
    with pytest.raises(ValueError, match=message):
        migrate(reader, state, parent)
    assert reader.state_dict() == before


def test_three_source_next_optimizer_update_and_rng_survive_fresh_restart(tmp_path):
    torch = pytest.importorskip('torch')
    torch.set_num_threads(1)
    parent, initial = parent_fixture(tmp_path)
    old = MixtureTokenBlocks(parent, 4)
    old.load_state_dict(initial)
    torch.manual_seed(13)

    def model_factory():
        # Dropout makes next-update parity depend on restoring RNG as well as Adam.
        return torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout(.2), torch.nn.Linear(8, 1))

    def update(reader, model, optimizer):
        batch = torch.tensor(reader.next_batch(5)[:, :-1], dtype=torch.float32) / 10000
        optimizer.zero_grad(set_to_none=True)
        model(batch).square().mean().backward()
        optimizer.step()

    model = model_factory()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    update(old, model, optimizer)
    update(old, model, optimizer)
    paused_optimizer = copy.deepcopy(optimizer.state_dict())
    paused_rng = torch.get_rng_state().clone()
    child = child_fixture(tmp_path, parent)
    reader = MixtureTokenBlocks(child, 4)
    migrate(reader, old.state_dict(), parent)
    assert torch.equal(torch.get_rng_state(), paused_rng)
    assert all(slot['step'].item() == 2 for slot in optimizer.state.values())
    payload = {'model': copy.deepcopy(model.state_dict()), 'optimizer': paused_optimizer,
               'data': reader.state_dict(), 'rng': paused_rng, 'step': 2,
               'tokens': reader.parent_tokens + sum(reader.targets.values())}
    path = tmp_path / 'training_state.pt'
    torch.save(payload, path)
    update(reader, model, optimizer)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_data = reader.state_dict()
    expected_rng = torch.get_rng_state().clone()

    loaded = torch.load(path, map_location='cpu', weights_only=True)
    restarted = MixtureTokenBlocks(child, 4)
    restarted.load_state_dict(loaded['data'])
    restored_model = model_factory()
    restored_model.load_state_dict(loaded['model'])
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=.003)
    restored_optimizer.load_state_dict(loaded['optimizer'])
    assert loaded['tokens'] == restarted.parent_tokens + sum(restarted.targets.values())
    torch.set_rng_state(loaded['rng'])
    update(restarted, restored_model, restored_optimizer)
    for name, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name], rtol=0, atol=0)
    actual_optimizer = restored_optimizer.state_dict()
    assert actual_optimizer['param_groups'] == expected_optimizer['param_groups']
    for parameter, slot in actual_optimizer['state'].items():
        assert slot['step'].item() == loaded['step'] + 1 == 3
        for name, value in slot.items():
            torch.testing.assert_close(value, expected_optimizer['state'][parameter][name], rtol=0, atol=0)
    assert restarted.state_dict() == expected_data
    assert torch.equal(torch.get_rng_state(), expected_rng)
