import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microllama.data_io.encoding import ENCODING_POLICY_VERSION, encoding_policy
from microllama.evaluation.likelihood import (
    generation_suppression_ids,
    rolling_windows,
    target_nll_sum,
)
from microllama.evaluation.lightweight import make_versioned_fixture, repetition_rate


def test_fresh_checkout_creates_a_fixture_without_a_legacy_evaluator(tmp_path, monkeypatch):
    from microllama.evaluation import lightweight
    settings = {"documents": 1, "max_characters_per_document": 12}
    expected = ({"samples": [{"id": "one"}]}, tmp_path / "fixture-new.json")
    calls = []

    def create(root, config, directory):
        calls.append((config, directory))
        return expected

    monkeypatch.setattr(lightweight, "fixed_fixture", create)
    assert lightweight.find_frozen_fixture(tmp_path, settings, "tokenizer") == expected
    assert calls == [(settings, tmp_path)]
    with pytest.raises(RuntimeError):
        lightweight.find_frozen_fixture(tmp_path, settings, "tokenizer", tmp_path / "missing.json")


def test_rolling_windows_cover_each_target_once_with_overlap():
    windows = list(rolling_windows(7, context_length=4, stride=2))
    assert [(w.target_start, w.target_end) for w in windows] == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert sum(w.target_end - w.target_start for w in windows) == 7
    assert all(w.target_end - w.target_start <= 2 for w in windows)
    with pytest.raises(ValueError, match="smaller"):
        list(rolling_windows(7, context_length=4, stride=4))


def test_target_nll_matches_one_by_one_causal_scoring_across_rolling_windows():
    torch = pytest.importorskip("torch")

    class Body:
        def __init__(self):
            self.inputs = []

        def __call__(self, input_ids, use_cache):
            assert use_cache is False
            self.inputs.append(input_ids[0].tolist())
            return SimpleNamespace(last_hidden_state=input_ids.unsqueeze(-1).float())

    class Model:
        def __init__(self):
            self.model = Body()
            self.lm_head = SimpleNamespace(weight=torch.tensor([[0.0], [0.1], [0.2], [0.3], [0.4]]))

    model = Model()
    conditioning = [1, 2]
    targets = [3, 4, 1, 2, 3]
    context_length = 4
    observed = target_nll_sum(model, conditioning, targets, context_length, stride=2,
                              projection_chunk_size=2)

    full = conditioning + targets
    expected = 0.0
    for i, target in enumerate(targets):
        target_position = len(conditioning) + i
        history = full[max(0, target_position - context_length):target_position]
        logits = model.lm_head.weight[:, 0] * history[-1]
        expected += torch.nn.functional.cross_entropy(logits[None, :], torch.tensor([target]),
                                                       reduction="sum").item()
    assert observed == pytest.approx(expected)
    assert len(model.model.inputs) == 3
    assert all(len(window) <= context_length for window in model.model.inputs)
    assert model.model.inputs[1] == [2, 3, 4, 1]  # Includes rolling overlap before target 2.


def test_generation_suppresses_pad_control_and_unused_ids_but_keeps_eos():
    tokenizer_json = {"added_tokens": [
        {"id": 1, "content": "<|bos|>", "special": True},
        {"id": 7, "content": "<|eos|>", "special": True},
        {"id": 20, "content": "<|reserved|>", "special": True},
        {"id": 30, "content": "<|tool_call_start|>", "special": False},
        {"id": 31, "content": "ordinary_added_word", "special": False},
    ]}
    suppressed = generation_suppression_ids(tokenizer_json, 40, 48, eos_token_id=7, pad_token_id=0)
    assert 7 not in suppressed
    assert {0, 1, 20, 30}.issubset(suppressed)
    assert set(range(40, 48)).issubset(suppressed)
    assert 31 not in suppressed


def test_repetition_rate_counts_only_duplicate_fourgrams():
    assert repetition_rate([1, 2, 3, 4, 1, 2, 3, 4]) == pytest.approx(0.2)
    assert repetition_rate([1, 2, 3]) == 0.0


def test_v2_fixture_versions_protocol_and_preserves_frozen_samples(tmp_path, monkeypatch):
    base_path = tmp_path / "fixture-v1.json"
    samples = [{"id": "d1", "text": "a literal <|im_end|> string", "truncated": True}]
    base = {"identity": {"documents": 1}, "identity_sha256": "frozen-id", "samples": samples}
    base_path.write_text(json.dumps(base))
    monkeypatch.setattr("microllama.evaluation.lightweight.sha256", lambda path: "base-file-sha")
    settings = {"encoding_policy": encoding_policy(), "context_length": 512, "rolling_stride": 256}
    fixture, path = make_versioned_fixture(base_path, base, settings, tmp_path / "v2")
    assert fixture["identity"]["protocol_version"] == "text-only-rolling-bpb-v2"
    assert fixture["identity"]["encoding_policy_version"] == ENCODING_POLICY_VERSION
    assert fixture["samples"] == samples
    assert path.name.startswith("fixture-v2-")


def test_engram_greedy_matches_full_forward_and_resets_prompt_state():
    torch = pytest.importorskip("torch")
    from test_engram import tiny_engram
    from microllama.evaluation.lightweight import greedy_engram
    torch.manual_seed(456)
    model = tiny_engram(revised=True).eval()
    suppressed = [0, 1, 7, 63]
    prompt = [1, 5, 12, 30]
    expected = []
    with torch.inference_mode():
        for _ in range(6):
            output = model(input_ids=torch.tensor([prompt + expected]), use_cache=False)
            logits = output.logits[0, -1].clone()
            logits[suppressed] = -torch.inf
            expected.append(int(logits.argmax()))
    assert greedy_engram(model, prompt, 6, 7, suppressed) == expected
    greedy_engram(model, [1, 48, 23], 4, 7, suppressed)
    assert greedy_engram(model, prompt, 6, 7, suppressed) == expected
    assert not set(expected).intersection(suppressed)
    assert greedy_engram(model, prompt, 6, expected[0], suppressed) == expected[:1]
    with pytest.raises(ValueError, match="context"):
        greedy_engram(model, prompt, 64, 7, suppressed)


def test_same_run_upload_is_file_only_retryable_and_escapes_text(tmp_path, monkeypatch):
    import wandb
    from microllama.evaluation import lightweight
    from scripts.evaluation.watch_checkpoints import evaluation_complete
    output = tmp_path / "results/step-00005000.json"
    output.parent.mkdir()
    uploads = []

    class Run:
        id = "test-run"
        url = "https://wandb.ai/entity/project/runs/test-run"

        def upload_file(self, path, root):
            uploads.append((str(Path(path).relative_to(root)), Path(path).read_text()))

    monkeypatch.setattr(wandb, "Api", lambda **kwargs: SimpleNamespace(run=lambda path: Run()))
    monkeypatch.setattr(wandb, "init", lambda **kwargs: pytest.fail("A second run logger is unsafe"))
    monkeypatch.setattr(lightweight, "RUNTIME", tmp_path / "runtime")
    report = {"step": 5000, "complete": True, "fixture_sha256": "fixture",
              "settings_sha256": "settings", "protocol_version": lightweight.PROTOCOL_VERSION,
              "continuations": [{"prompt": "<script>alert(1)</script>", "continuation": "A & B",
                                 "generated_tokens": 3, "ended_with_eos": False,
                                 "repeated_4gram_fraction": 0}]}
    lightweight.publish_to_training_run(report, output, "entity/project/test-run")
    assert evaluation_complete(output, "entity/project/test-run")
    assert not evaluation_complete(output, "entity/project/other-run")
    assert len(uploads) == 4
    assert '&lt;script&gt;' in uploads[1][1] and 'A &amp; B' in uploads[1][1]
    assert '<script>' not in uploads[1][1]
    lightweight.publish_to_training_run(report, output, "entity/project/test-run")
    assert len(uploads) == 8
    assert [name for name, _ in uploads[:4]] == [name for name, _ in uploads[4:]]

    monkeypatch.setattr(wandb, "Api", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")))
    lightweight.publish_to_training_run(report, output, "entity/project/test-run")
    assert evaluation_complete(output)
    assert not evaluation_complete(output, "entity/project/test-run")


def test_native_tables_use_shared_logger_and_keep_checkpoint_rows(tmp_path, monkeypatch):
    import wandb
    from microllama.evaluation import lightweight
    from scripts.evaluation.watch_checkpoints import evaluation_complete
    output = tmp_path / 'results/step-00006000.json'
    output.parent.mkdir()
    report = {'step': 6000, 'complete': True, 'fixture_sha256': 'fixture',
              'settings_sha256': 'settings', 'protocol_version': lightweight.PROTOCOL_VERSION,
              'text_bits_per_utf8_byte': 1.03, 'cloze_accuracy': 11 / 12,
              'continuations': [{'prompt': 'A triangle has', 'continuation': ' three sides',
                                'generated_tokens': 2, 'ended_with_eos': False,
                                'repeated_4gram_fraction': 0}]}
    run_path = 'entity/project/test-run'
    old = {**report, 'step': 5000, 'wandb_training_run': run_path}
    (output.parent / 'step-00005000.json').write_text(json.dumps(old))
    (output.parent / 'step-00005500.json').write_text(json.dumps({**old, 'step': 5500, 'fixture_sha256': 'other'}))
    uploads = []
    summary = {}
    monkeypatch.setattr(wandb, 'Api', lambda **kwargs: SimpleNamespace(run=lambda path: SimpleNamespace(
        id='test-run', url='https://wandb.ai/entity/project/runs/test-run',
        upload_file=lambda path, root: uploads.append(path))))
    monkeypatch.setattr(wandb, 'init', lambda **kwargs: pytest.fail('Do not resume the trainer'))
    monkeypatch.setattr(lightweight, 'RUNTIME', tmp_path / 'runtime')
    attachments = []

    def attach(pid, path):
        attachments.append((pid, path))
        return SimpleNamespace(summary=summary)  # No log() or finish() available.

    monkeypatch.setattr(lightweight, 'attach_training_logger', attach)
    lightweight.publish_to_training_run(report, output, run_path, train_pid=123)
    assert attachments == [(123, run_path)]
    assert summary['quality/checkpoint_step'] == 6000
    assert [row[0] for row in summary['quality/fixed_prompts'].data] == [5000, 6000]
    assert len(summary['quality/checkpoint_metrics'].data) == 2
    assert len(uploads) == 4
    assert evaluation_complete(output, run_path, require_native_table=True)
    report.pop('wandb_native_table_status')
    output.write_text(json.dumps(report))
    assert not evaluation_complete(output, run_path, require_native_table=True)


def test_shared_logger_validates_trainer_run_and_child_service(tmp_path, monkeypatch):
    import wandb
    from wandb.sdk.lib import service_token
    from microllama.evaluation.lightweight import attach_training_logger
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'wandb-id.json').write_text(json.dumps({'id': 'test-run'}))
    parent = tmp_path / 'proc/123'
    (parent / 'task/123').mkdir(parents=True)
    (parent / 'cmdline').write_bytes(('python\0-m\0microllama.training.distributed\0--config\0'
                                     + str(logs / 'launch-config.json') + '\0').encode())
    (parent / 'task/123/children').write_text('456')
    child = tmp_path / 'proc/456'
    child.mkdir()
    port = tmp_path / 'port-123.txt'
    port.write_text('sock=12345\nEOF')
    (child / 'cmdline').write_bytes('\0'.join(['/bin/wandb-core', '--port-filename', str(port), '--pid', '123', '']).encode())
    tokens = []
    setups = []
    monkeypatch.setattr(wandb, 'run', None)
    monkeypatch.setattr(service_token, 'set_service_token', lambda **kwargs: tokens.append(kwargs))
    monkeypatch.setattr(wandb, 'setup', lambda **kwargs: setups.append(kwargs))
    logger = SimpleNamespace(path='entity/project/test-run')
    monkeypatch.setattr(wandb, 'attach', lambda run_id: logger)
    assert attach_training_logger(123, logger.path, proc_root=tmp_path / 'proc') is logger
    assert tokens == [dict(parent_pid=123, transport='tcp', host='localhost', port=12345)]
    assert setups[0]['settings'].root_dir == str(logs)
    port.unlink()
    (child / 'fd').mkdir()
    (child / 'fd/4').symlink_to('socket:[1234]')
    (child / 'net').mkdir()
    (child / 'net/tcp').write_text(
        'header\n0: 0100007F:3039 00000000:0000 0A 0 0 0 1000 0 1234\n'
        '1: 00000000:303A 00000000:0000 0A 0 0 0 1000 0 1234\n'
        '2: 0100007F:303B 00000000:0000 0A 0 0 0 1000 0 9999\n')
    assert attach_training_logger(123, logger.path, proc_root=tmp_path / 'proc') is logger
    assert tokens[-1]['port'] == 12345  # Ignore non-loopback and another process's sockets.
    with pytest.raises(RuntimeError, match='do not match'):
        attach_training_logger(123, 'entity/project/other-run', proc_root=tmp_path / 'proc')
    (parent / 'task/123/children').write_text('')
    with pytest.raises(RuntimeError, match='one live'):
        attach_training_logger(123, logger.path, proc_root=tmp_path / 'proc')
