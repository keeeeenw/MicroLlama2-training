"""Budget handoff refusal checks; no GPU, network or live process mutations."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('mac_continue_worker',
    Path(__file__).resolve().parents[1] / 'scripts/mac/continue_worker.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def plans():
    old = {'parent_tokens': 100, 'coverage': {'balanced_unconsumed_targets': 500000},
           'config': {'target_tokens': 200, 'learning_rate': .0003},
           'validation': {'blocks': 128}, 'implementation': {'source': 'sha'},
           'target_authorization': 'pilot'}
    return old, worker.extend_plan(old, {'target_tokens': 300000})


def test_only_target_and_authorization_change():
    old, new = plans()
    assert new['config']['target_tokens'] == 300000
    state = {'config': copy.deepcopy(old['config']), 'validation': old['validation'], 'tokens': 205}
    worker.check_transition(old, new, state)
    assert old['config']['target_tokens'] == 200
    for section, key, value in [('config', 'learning_rate', .001), ('validation', 'blocks', 2),
                                ('implementation', 'source', 'different')]:
        bad = copy.deepcopy(new)
        bad[section][key] = value
        with pytest.raises(ValueError):
            worker.check_transition(old, bad, state)


def test_target_stays_inside_existing_capacity_with_update_margin():
    old, _ = plans()
    result = worker.extend_plan(old, {'target_tokens': 900000})
    assert result['config']['target_tokens'] == 100 + 500000 - 131072
    with pytest.raises(ValueError):
        worker.extend_plan(old, {'target_tokens': 200})


@pytest.mark.parametrize('phase', ['failed', 'qualifying', 'starting-training', 'paused-before-training'])
def test_failed_unknown_or_no_production_cannot_resume(phase):
    old, _ = plans()
    with pytest.raises(RuntimeError):
        worker.healthy_budget_end({'phase': phase}, {}, old, now=900)


def test_user_stop_cancels_handoff_and_exhausted_budget_allows_it():
    old, _ = plans()
    launch = {'unix': 0, 'max_seconds': 86400, 'max_updates': 10000}
    status = {'phase': 'paused', 'updates': 30, 'checkpoint': '/external/step', 'tokens': 180}
    with pytest.raises(RuntimeError, match='user stop'):
        worker.healthy_budget_end(status, launch, old, now=1000)
    worker.healthy_budget_end(status, launch, old, now=86410)
    worker.healthy_budget_end({**status, 'phase': 'completed', 'tokens': 205}, launch, old, now=1000)
    with pytest.raises(RuntimeError):
        worker.healthy_budget_end({**status, 'phase': 'completed', 'tokens': 199}, launch, old, now=1000)


def test_transfer_budget_is_not_reset():
    launch = {'max_download_bytes': 7000}
    qualification = {'download_bytes_requested': 6400}
    events = [{'phase': 'training'}, {'download_bytes_requested': 100}, {'download_bytes_requested': 100}]
    assert worker.remaining_download_bytes(launch, qualification, events) == 500
    with pytest.raises(ValueError):
        worker.remaining_download_bytes(launch, qualification, [{'download_bytes_requested': 800}])
    with pytest.raises(ValueError):
        worker.remaining_download_bytes(launch, qualification, [
            {'download_bytes_requested': 100}, {'download_bytes_requested': 0}])
