"""A distinct local Mac W&B identity; never inherit the closed CUDA run."""
from __future__ import annotations
import json
import os
from pathlib import Path
import uuid

from .mac import write_json

OLD_CUDA_RUN = 'original-cuda-run'


def identity(directory, plan_sha, *, mode):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'identity.json'
    if path.exists():
        result = json.loads(path.read_text())
        if (result.get('plan_sha256') != plan_sha or result.get('project') != 'MicroLlama2'
                or result.get('id') == OLD_CUDA_RUN or not result.get('id')):
            raise ValueError('W&B Mac identity or plan differs; never reuse the CUDA run')
        return result
    result = {'id': uuid.uuid4().hex[:16], 'project': 'MicroLlama2', 'plan_sha256': plan_sha,
              'name': 'm1-mac-continuation-' + uuid.uuid4().hex[:8], 'initial_mode': mode}
    write_json(path, result)
    return result


class Tracker:
    def __init__(self, directory, plan_sha, config):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        os.environ['WANDB_CACHE_DIR'] = str(directory / 'cache')
        os.environ['WANDB_DATA_DIR'] = str(directory / 'data')
        import wandb
        from wandb.sdk.lib import apikey
        mode = 'online' if apikey.api_key() else 'offline'
        self.identity = identity(directory, plan_sha, mode=mode)
        self.directory, self.mode, self.error = directory, mode, None
        self.run = wandb.init(project='MicroLlama2', id=self.identity['id'],
            name=self.identity['name'], dir=str(directory), mode=mode,
            resume='allow' if mode == 'online' else None, config=config,
            settings=wandb.Settings(disable_git=True, disable_code=True, init_timeout=30))
        self.run.define_metric('counters/optimizer_step')
        self.run.define_metric('training/*', step_metric='counters/optimizer_step')
        self.run.define_metric('validation/*', step_metric='counters/optimizer_step')
        write_json(directory / 'status.json', {**self.identity, 'mode': mode, 'active': True,
                                               'old_cuda_run_reused': False})

    def log(self, event):
        payload = {f"{event.get('phase', 'run')}/{key}": value for key, value in event.items()
                   if isinstance(value, (int, float)) and not isinstance(value, bool)}
        if 'sources' in event:
            payload.update({f'validation/{source}/loss': item['loss'] for source, item in event['sources'].items()})
        if 'step' in event:
            payload['counters/optimizer_step'] = event['step']
        try:
            self.run.log(payload)
        except Exception as error:
            self.error = f'{type(error).__name__}: {error}'
            write_json(self.directory / 'error.json', {'error': self.error, 'mode': self.mode})

    def finish(self, exit_code=0):
        self.run.finish(exit_code=exit_code)
        write_json(self.directory / 'status.json', {**self.identity, 'mode': self.mode, 'active': False,
                                                   'old_cuda_run_reused': False, 'error': self.error})
