"""Stream a full diagnostic gradient comparison without backend allocations."""
import json
from pathlib import Path

import numpy as np

from .mac import sha256, write_json


def full_gradient_check(values, *, export=None, reference=None):
    if (export is None) == (reference is None):
        raise ValueError('Choose one gradient export or reference')
    if export is not None:
        export = Path(export)
        export.mkdir(exist_ok=False)
        manifest = {}
    else:
        reference = Path(reference)
        manifest = json.loads((reference / 'manifest.json').read_text())
        if set(manifest) != set(values):
            raise ValueError('Gradient parameter names differ')
    actual_sq = expected_sq = dot = error_sq = 0.
    maximum_error = 0.
    elements = 0
    per_parameter = {}
    for index, name in enumerate(sorted(values)):
        value = values[name]
        if hasattr(value, 'detach'):
            value = value.detach().float().cpu().numpy()
        else:
            value = np.asarray(value, dtype=np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite gradient: {name}')
        if export is not None:
            path = export / f'{index:04d}.npy'
            np.save(path, value, allow_pickle=False)
            manifest[name] = {'file': path.name, 'shape': list(value.shape), 'sha256': sha256(path)}
        else:
            item = manifest[name]
            if Path(item['file']).name != item['file']:
                raise ValueError('Unsafe reference filename')
            path = reference / item['file']
            if sha256(path) != item['sha256']:
                raise ValueError(f'Reference gradient hash differs: {name}')
            expected = np.load(path, mmap_mode='r', allow_pickle=False)
            if value.shape != expected.shape or list(value.shape) != item['shape']:
                raise ValueError(f'Gradient shape differs: {name}')
        flat = value.reshape(-1)
        local_actual = local_expected = local_dot = local_error = 0.
        for start in range(0, flat.size, 262144):
            a = flat[start:start + 262144].astype(np.float64)
            local_actual += float(np.dot(a, a))
            if reference is not None:
                b = expected.reshape(-1)[start:start + 262144].astype(np.float64)
                delta = a - b
                local_expected += float(np.dot(b, b))
                local_dot += float(np.dot(a, b))
                local_error += float(np.dot(delta, delta))
                maximum_error = max(maximum_error, float(np.max(np.abs(delta), initial=0)))
        actual_sq += local_actual
        expected_sq += local_expected
        dot += local_dot
        error_sq += local_error
        elements += flat.size
        if reference is not None:
            per_parameter[name] = {'actual_l2': local_actual ** .5, 'reference_l2': local_expected ** .5,
                                   'difference_l2': local_error ** .5}
        del value, flat
        if reference is not None:
            del expected
    result = {'parameter_count': len(values), 'elements': elements, 'actual_l2': actual_sq ** .5}
    if export is not None:
        write_json(export / 'manifest.json', manifest)
        result.update(export=str(export), manifest_sha256=sha256(export / 'manifest.json'))
    else:
        if min(actual_sq, expected_sq) <= 0:
            raise ValueError('Cannot compare zero gradients')
        result.update(reference=str(reference), reference_manifest_sha256=sha256(reference / 'manifest.json'),
                      reference_l2=expected_sq ** .5, cosine=dot / (actual_sq * expected_sq) ** .5,
                      relative_l2=(error_sq / expected_sq) ** .5, maximum_absolute_error=maximum_error,
                      per_parameter=per_parameter)
    return result
