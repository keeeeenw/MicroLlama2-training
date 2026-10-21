"""Regression for large-embedding gradient sampling; no GPU/model allocation."""
import numpy as np
import pytest
from scripts.mac.bench_speed import gradient_sample_indices
from microllama.training.mac_gradient_check import full_gradient_check


@pytest.mark.parametrize('size', [1, 127, 2**26, 65536 * 1024])
def test_gradient_sample_indices_stay_in_bounds_and_include_endpoints(size):
    indices = gradient_sample_indices(size)
    assert indices.dtype == np.int64
    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == size - 1
    assert np.all(indices >= 0) and np.all(indices < size)
    assert np.all(indices[1:] >= indices[:-1])


def test_gradient_sample_indices_refuse_empty_parameters():
    with pytest.raises(ValueError, match='empty'):
        gradient_sample_indices(0)


def test_full_gradient_check_reconciles_all_elements_and_refuses_tampering(tmp_path):
    values = {'large': np.arange(40, dtype=np.float32).reshape(5, 8), 'small': np.array([.125], np.float32)}
    root = tmp_path / 'reference'
    exported = full_gradient_check(values, export=root)
    assert exported['elements'] == 41
    identical = full_gradient_check(values, reference=root)
    assert identical['cosine'] == pytest.approx(1.)
    assert identical['relative_l2'] == 0
    changed = {**values, 'small': np.array([1.125], np.float32)}
    difference = full_gradient_check(changed, reference=root)
    assert difference['maximum_absolute_error'] == 1
    assert difference['per_parameter']['small']['difference_l2'] == 1
    with (root / '0000.npy').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        full_gradient_check(values, reference=root)
