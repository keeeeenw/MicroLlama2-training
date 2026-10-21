"""Architecture ablation must remove only the Engram residual/optimizer state."""
import numpy as np
import pytest

mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from microllama.training.mac_mlx_fast import FastEngine
from scripts.mac.ablate_ultra_engram import without_engram
from test_mac_ultra_run import fixture


def test_disabled_engram_matches_zero_residual_for_remaining_parameters():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        original, rows, _ = fixture(1)
        params = dict(original.params)
        params['model.engram.residual_scale'] = mx.zeros_like(params['model.engram.residual_scale'])
        enabled = FastEngine(params, original.slots, original.policy, original.config, chunk=4)
        disabled = without_engram(original)
        ev, eg = enabled.gradient(enabled.params, mx.array(rows[:1]), enabled.indices(rows[:1]))
        dv, dg = disabled.gradient(disabled.params, mx.array(rows[:1]), disabled.indices(rows[:1]))
        mx.eval(ev, eg, dv, dg)
        assert float(ev) == pytest.approx(float(dv), abs=1e-6)
        assert disabled.config['engram_layers'] == []
        assert original.config['engram_layers']
        assert set(disabled.params) == set(disabled.slots) == set(disabled.policy)
        assert not any(n.startswith('model.engram.') for n in disabled.params)
        for name in disabled.params:
            np.testing.assert_allclose(np.asarray(eg[name]), np.asarray(dg[name]), atol=1e-6, rtol=1e-5, err_msg=name)
    finally:
        mx.set_default_device(previous)
