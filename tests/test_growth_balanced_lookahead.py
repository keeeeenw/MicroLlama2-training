"""Variable boundary masks must not push a four-source mix outside one row."""
import copy
import random

import numpy as np

from microllama.data_io.growth_reader import MixtureTokenBlocks

WEIGHTS = {'edu': .48, 'dclm': .32, 'cosmopedia': .15, 'finemath': .05}


def test_variable_target_lookahead_reproduces_and_prevents_old_discrepancy():
    rng = random.Random(7)
    costs = {s: [rng.randint(7900, 8192) for _ in range(30_000)] for s in WEIGHTS}
    reader = object.__new__(MixtureTokenBlocks)
    reader.weights = WEIGHTS
    maxima = []
    for lookahead in (False, True):
        counts = dict.fromkeys(WEIGHTS, 0)
        cursors = dict.fromkeys(WEIGHTS, 0)
        maximum = 0
        for _ in range(30_000):
            valid = {s: costs[s][cursors[s]] for s in WEIGHTS}
            total = sum(counts.values())
            source = (reader._balanced_source(counts, valid) if lookahead else
                      max(WEIGHTS, key=lambda s: (total*WEIGHTS[s]-counts[s], WEIGHTS[s], s)))
            counts[source] += valid[source]
            cursors[source] += 1
            maximum = max(maximum, *(abs(counts[s]-sum(counts.values())*w) for s, w in WEIGHTS.items()))
        maxima.append(maximum)
    assert maxima[0] > 8192  # Real failure mode: varying document-boundary masks.
    assert maxima[1] <= 8192


def test_lookahead_never_consumes_a_valid_row(tmp_path):
    from test_growth_mixture_transition import parent_fixture
    parent, state = parent_fixture(tmp_path, consumed=1)
    reader = MixtureTokenBlocks(parent, 4)
    reader.load_state_dict(state)
    before = copy.deepcopy(reader.state_dict())
    expected = reader._row('edu', consume=False).copy()
    peeked = copy.deepcopy(reader.state_dict())
    assert peeked['source_targets'] == before['source_targets']
    assert peeked['sources']['edu']['cursor'] == before['sources']['edu']['cursor']
    np.testing.assert_array_equal(reader._row('edu', consume=False), expected)
    np.testing.assert_array_equal(reader._row('edu'), expected)
    assert reader.state_dict()['sources']['edu']['cursor'] == before['sources']['edu']['cursor']+1
    assert reader.targets['edu'] == before['source_targets']['edu']+4
