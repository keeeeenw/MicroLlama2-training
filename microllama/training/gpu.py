"""Shared local GPU admission for offline construction and bounded experiments.

The advisory lock survives a coordinator through an inherited descriptor. An
independent NVIDIA process check prevents competition with non-project jobs.
Call admission before allocating CUDA tensors, while holding the lock.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import subprocess

import torch


@contextmanager
def gpu_lock(inherited_fd=None):
    path = Path('.runtime/experiments/gpu.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    if inherited_fd is not None:
        stat, expected = os.fstat(inherited_fd), path.stat()
        if (stat.st_dev, stat.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError('Inherited descriptor is not the project GPU lock')
        fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield inherited_fd
    else:
        with path.open('a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield handle.fileno()


def idle_gpu():
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True)
    if any(line.strip() != str(os.getpid()) for line in apps.splitlines() if line.strip()):
        raise RuntimeError('A CUDA process is already active; do not compete with it')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 CUDA is unavailable')
