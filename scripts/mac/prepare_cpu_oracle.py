"""Export the independent full8K FP32 CPU oracle without starting training."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    import mlx.core as mx
    import torch
    from microllama.training import mac_ultra_run as ultra, mac_run as base
    torch.set_num_threads(args.threads)
    mx.set_default_device(mx.cpu)
    plan, old = ultra.load_plan(args.plan)
    output = base.external(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def report(value):
        print(json.dumps({'unix': time.time(), **value}), flush=True)
    with base.device_lock(Path(old['archive_cache']).parent):
        ultra.cpu_reference(output, plan, old, report)
    report({'phase': 'completed', 'production_started': False,
            'official_targets_advanced': 0})


if __name__ == '__main__':
    main()
