"""Versioned text-only rolling BPB and small diagnostics on CPU."""
import argparse
import fcntl
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time

from ..data_io.encoding import (
    ENCODING_POLICY_VERSION,
    encode_ordinary_text,
    encoding_policy,
    load_ordinary_text_tokenizer,
)
from ..data_io.transfer import atomic_json
from ..common.paths import data_root
from .fixtures import fixed_fixture
from .likelihood import generation_suppression_ids, target_nll_sum


PROTOCOL_VERSION = "text-only-rolling-bpb-v2"
REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO / "configs/evaluation/lightweight.json"
RUNTIME = REPO / ".runtime/evaluation"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def repetition_rate(ids, n=4):
    grams = [tuple(ids[i:i + n]) for i in range(max(0, len(ids) - n + 1))]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    if settings.get("cpu_threads") != 2:
        raise ValueError("The lightweight evaluation is fixed to two CPU threads")
    if settings.get("documents", 0) < 1 or settings.get("max_characters_per_document", 0) < 1:
        raise ValueError("Fixture document count and character limit must be positive")
    context_length = settings.get("sequence_length", 0)
    if context_length < 2:
        raise ValueError("sequence_length must be at least two")
    settings.update({
        "protocol_version": PROTOCOL_VERSION,
        "encoding_policy": encoding_policy(),
        "context_length": context_length,
        "rolling_stride": min(256, context_length - 1),
        "logit_chunk_size": 128,
        "text_metric": "text-token NLL in nats divided by ln(2) and raw UTF-8 text bytes; EOS excluded",
        "eos_metric": "EOS target NLL reported separately for complete, untruncated fixture documents",
    })
    return settings


def find_frozen_fixture(directory, settings, tokenizer_sha256, requested=None):
    """Find the existing v1 fixture without opening source data or changing it."""
    if requested is not None:
        candidates = [Path(requested)]
    else:
        candidates = sorted(Path(directory).glob("fixture-*.json"))
    matches = []
    for path in candidates:
        if "fixture-v2-" in path.name or not path.is_file():
            continue
        try:
            fixture = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        identity = fixture.get("identity", {})
        if (identity.get("documents") == settings["documents"]
                and identity.get("max_characters_per_document") == settings["max_characters_per_document"]
                and identity.get("tokenizer_sha256") == tokenizer_sha256
                and len(fixture.get("samples", [])) == settings["documents"]):
            matches.append((path, fixture))
    if not matches and requested is None:
        return fixed_fixture(data_root(), settings, Path(directory))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one matching frozen v1 fixture, found {len(matches)}")
    return matches[0]


def make_versioned_fixture(base_path, base_fixture, settings, output_directory):
    identity = {
        "base_fixture_sha256": sha256(base_path),
        "base_fixture_identity_sha256": base_fixture["identity_sha256"],
        "protocol_version": PROTOCOL_VERSION,
        "encoding_policy_version": ENCODING_POLICY_VERSION,
        "encoding_policy": settings["encoding_policy"],
        "settings_sha256": hashlib.sha256(
            json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "context_length": settings["context_length"],
        "rolling_stride": settings["rolling_stride"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = Path(output_directory) / f"fixture-v2-{key[:16]}.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved.get("identity_sha256") != key or saved.get("identity") != identity:
            raise RuntimeError(f"Cached v2 fixture identity mismatch: {destination}")
        if saved.get("samples") != base_fixture["samples"]:
            raise RuntimeError(f"Cached v2 fixture differs from the frozen v1 samples: {destination}")
        return saved, destination
    versioned = {"identity": identity, "identity_sha256": key,
                 "base_fixture_samples_sha256": hashlib.sha256(json.dumps(
                     base_fixture["samples"], sort_keys=True, ensure_ascii=False,
                     separators=(",", ":")).encode()).hexdigest(),
                 "source": base_fixture.get("source"),
                 "source_sha256": base_fixture.get("source_sha256"),
                 "samples": base_fixture["samples"]}
    atomic_json(destination, versioned)
    return versioned, destination


def score_choice(model, tokenizer, bos_id, prompt, answer, settings):
    prompt_ids = encode_ordinary_text(tokenizer, prompt).ids
    answer_ids = encode_ordinary_text(tokenizer, answer).ids
    if not answer_ids:
        raise ValueError("Cloze answer must encode to at least one ordinary-text token")
    nll = target_nll_sum(model, [bos_id] + prompt_ids, answer_ids,
                         settings["context_length"], settings["rolling_stride"],
                         projection_chunk_size=settings["logit_chunk_size"])
    return -nll / len(answer_ids)


def greedy_engram(model, prompt_ids, max_new_tokens, eos_id, suppressed_ids):
    """One prompt at a time; carry both the KV cache and explicit n-gram suffix.

    All state belongs to this call, so independent fixed prompts cannot share memory.
    Engram's HF generate() deliberately rejects implicit state management.
    """
    import torch
    from ..training.engram import engram_prefill, engram_decode_step

    if len(prompt_ids) + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation exceeds checkpoint context")
    generated = []
    with torch.inference_mode():
        output = engram_prefill(model, torch.tensor([prompt_ids], dtype=torch.long))
        for index in range(max_new_tokens):
            logits = output.logits[0, -1].clone()
            logits[suppressed_ids] = -torch.inf
            token = int(logits.argmax())
            generated.append(token)
            if token == eos_id or index + 1 == max_new_tokens:
                break
            output = engram_decode_step(
                model, torch.tensor([[token]], dtype=torch.long),
                past_key_values=output.past_key_values, engram_state=output.engram_state,
            )
    return generated


def attach_training_logger(train_pid, run_path, *, proc_root=Path("/proc")):
    """Connect to the live trainer's W&B service; never create a second service.

    Linux PID/command, run identity and child ownership checks prevent attaching to
    another job. The pinned SDK's localhost port token contains no API credential.
    An attached process does not own the service and must never call finish().
    """
    import wandb
    from wandb.sdk.lib.service_token import set_service_token
    from wandb.sdk.service.port_file import PortFile

    if wandb.run is not None:
        raise RuntimeError("Table publishing must run in a separate evaluation process")
    parent = Path(proc_root) / str(train_pid)
    command = parent.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
    if (train_pid <= 0 or "microllama.training.distributed" not in command
            or "--config" not in command or parent.stat().st_uid != os.getuid()):
        raise RuntimeError("PID is not this user's distributed training process")
    logs = Path(command[command.index("--config") + 1]).parent
    saved_id = json.loads((logs / "wandb-id.json").read_text())["id"]
    if saved_id != run_path.split("/")[-1]:
        raise RuntimeError("Training PID and requested W&B run do not match")
    ports = []
    for pid in parent.joinpath("task", str(train_pid), "children").read_text().split():
        child = Path(proc_root) / pid
        try:
            argv = child.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0")
        except FileNotFoundError:
            continue
        if (not argv or Path(argv[0]).name != "wandb-core" or "--port-filename" not in argv
                or "--pid" not in argv or argv[argv.index("--pid") + 1] != str(train_pid)):
            continue
        port = PortFile()
        try:
            port.read(argv[argv.index("--port-filename") + 1])
        except FileNotFoundError:
            # W&B removes its startup port file once the parent is connected.
            # Inspect only this verified child's own IPv4 loopback listeners.
            inodes = set()
            for descriptor in child.joinpath("fd").iterdir():
                try:
                    link = os.readlink(descriptor)
                except FileNotFoundError:
                    continue
                if link.startswith("socket:["):
                    inodes.add(link[8:-1])
            for line in child.joinpath("net/tcp").read_text().splitlines()[1:]:
                fields = line.split()
                if (fields[9] in inodes and fields[3] == "0A"
                        and fields[1].split(":")[0] == "0100007F"):
                    ports.append(int(fields[1].split(":")[1], 16))
        else:
            if port.is_valid and port.sock_port:
                ports.append(port.sock_port)
    if len(ports) != 1:
        raise RuntimeError(f"Expected one live trainer W&B service, found {len(ports)}")
    set_service_token(parent_pid=train_pid, transport="tcp", host="localhost", port=ports[0])
    wandb.setup(settings=wandb.Settings(root_dir=str(logs), disable_git=True, disable_code=True, console="off"))
    attached = wandb.attach(run_id=saved_id)
    if attached is None or attached.path != run_path:
        raise RuntimeError("Attached W&B logger has a different run identity")
    return attached


def publish_to_training_run(report, output, run_path, *, train_pid=None):
    """Upload readable files and optionally publish native Tables via shared service.

    Native summary tables use dedicated quality keys, never write history steps,
    and never initialize/resume/finish the trainer's run. Upload retries reuse the
    completed report and deterministic file paths without repeating inference.
    """
    parts = run_path.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError("--wandb-run must be ENTITY/PROJECT/RUN_ID")
    output = Path(output)
    upload_lock = None
    try:
        import wandb
        run = wandb.Api(timeout=60).run(run_path)
        if run.id != parts[2]:
            raise RuntimeError("W&B returned a different run")
        root = RUNTIME / "wandb-files" / parts[2]
        directory = root / "quality/fixed-prompts"
        directory.mkdir(parents=True, exist_ok=True)
        upload_lock = (root / ".upload.lock").open("a")
        fcntl.flock(upload_lock, fcntl.LOCK_EX)
        candidates = [report]
        for path in output.parent.glob("step-*.json"):
            if path == output:
                continue
            item = json.loads(path.read_text())
            if (item.get("complete") is True and item.get("wandb_training_run") == run_path
                    and item.get("fixture_sha256") == report["fixture_sha256"]
                    and item.get("settings_sha256") == report["settings_sha256"]):
                candidates.append(item)
        candidates.sort(key=lambda item: item["step"])
        columns = ["checkpoint_step", "prompt", "continuation", "generated_tokens",
                   "ended_with_eos", "repeated_4gram_fraction"]
        rows = [[item["step"], sample["prompt"], sample["continuation"],
                 sample["generated_tokens"], sample["ended_with_eos"], sample["repeated_4gram_fraction"]]
                for item in candidates for sample in item["continuations"]]
        table = wandb.Table(columns=columns, data=rows)
        metrics_columns = ["checkpoint_step", "text_bits_per_utf8_byte", "cloze_accuracy",
                           "repeated_4gram_fraction"]
        metrics_table = wandb.Table(columns=metrics_columns, data=[
            [item["step"], *[item.get(key) for key in metrics_columns[1:]]] for item in candidates])

        def render(reports):
            rows = []
            for item in reports:
                for sample in item["continuations"]:
                    values = [item["step"], sample["prompt"], sample["continuation"],
                              sample["generated_tokens"], sample["ended_with_eos"],
                              f'{sample["repeated_4gram_fraction"]:.3f}']
                    rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
            return ('<!doctype html><meta charset="utf-8"><title>Fixed prompt checks</title>'
                    '<style>body{font:16px system-ui;margin:2rem}table{border-collapse:collapse;width:100%}'
                    'td,th{border:1px solid #bbb;padding:12px;text-align:left}td{white-space:pre-wrap}</style>'
                    '<h1>Fixed prompt checks</h1><p>Greedy, at most '
                    f'{int(report.get("generation_max_new_tokens", 32))} new tokens; two CPU threads. '
                    'Pretraining continuations, not instruction-following tests.</p>'
                    f'<p>Protocol: {html.escape(report["protocol_version"])}; '
                    f'fixture: {html.escape(report["fixture_sha256"])}; '
                    f'settings: {html.escape(report["settings_sha256"])}</p>'
                    '<table><thead><tr><th>Checkpoint step</th><th>Prompt</th><th>Continuation</th>'
                    '<th>Tokens</th><th>EOS</th><th>Repeated 4-grams</th></tr></thead><tbody>'
                    + "".join(rows) + '</tbody></table>')

        stem = f'step-{report["step"]:08d}'
        json_path = directory / f"{stem}.json"
        # Upload receipts describe local transport state, not evaluation quality.
        payload = {key: value for key, value in report.items() if not key.startswith("wandb_")}
        payload["wandb_training_run"] = run_path
        atomic_json(json_path, payload)
        html_path = directory / f"{stem}.html"
        html_path.write_text(render([report]), encoding="utf-8")
        index_path = directory / "index.html"
        index_path.write_text(render(candidates), encoding="utf-8")
        text_path = directory / "index.txt"
        text_path.write_text("\n\n".join(
            f'Step {item["step"]}\nPrompt: {sample["prompt"]}\nContinuation: {sample["continuation"]}'
            for item in candidates for sample in item["continuations"]) + "\n", encoding="utf-8")
        for path in (json_path, html_path, index_path, text_path):
            run.upload_file(str(path), root=str(root))
        if train_pid is not None:
            attached = attach_training_logger(train_pid, run_path)
            attached.summary.update({"quality/fixed_prompts": table,
                                     "quality/checkpoint_metrics": metrics_table,
                                     "quality/checkpoint_step": candidates[-1]["step"]})
            report["wandb_native_table_status"] = "published"
        report.update(wandb_training_run=run_path, wandb_status="uploaded",
                      wandb_url=run.url,
                      wandb_qualitative_file="quality/fixed-prompts/index.txt")
        if train_pid is not None:
            report["wandb_qualitative_key"] = "quality/fixed_prompts"
        report.pop("wandb_error", None)
    except Exception as error:
        report.update(wandb_training_run=run_path, wandb_status="failed",
                      wandb_error=f"{type(error).__name__}: {error}")
        print(f"Same-run upload failed; local results preserved: {report['wandb_error']}",
              file=sys.stderr, flush=True)
    finally:
        try:
            atomic_json(output, report)
        finally:
            if upload_lock is not None:
                upload_lock.close()
    return report


def evaluate(checkpoint, config_path, *, base_fixture_path=None, output_path=None,
             wandb_project=None, wandb_entity=None, wandb_run=None, wandb_train_pid=None):
    if wandb_project and wandb_run:
        raise ValueError("Choose a separate evaluation project or an existing training run")
    start_time = time.monotonic()
    settings = load_settings(config_path)
    checkpoint = Path(checkpoint)
    tokenizer_path = checkpoint / "tokenizer.json"
    tokenizer_digest = sha256(tokenizer_path)
    base_path, base_fixture = find_frozen_fixture(
        RUNTIME, settings, tokenizer_digest, requested=base_fixture_path)
    if base_fixture["identity"].get("tokenizer_sha256") != tokenizer_digest:
        raise RuntimeError("Checkpoint tokenizer does not match the frozen v1 fixture")

    versioned_fixture, fixture_path = make_versioned_fixture(
        base_path, base_fixture, settings, RUNTIME / "fixtures-v2")
    fixture_digest = sha256(fixture_path)
    settings_digest = hashlib.sha256(
        json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    import torch
    from transformers import Qwen3ForCausalLM

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    tokenizer = load_ordinary_text_tokenizer(tokenizer_path)
    tokenizer_json = json.loads(tokenizer_path.read_text())
    model_type = json.loads((checkpoint / "config.json").read_text()).get("model_type")
    is_engram = model_type == "engram_qwen3"
    if is_engram:
        from ..training.engram import load_engram_model
        model = load_engram_model(checkpoint).float().eval()
    elif model_type == "qwen3" and not (checkpoint / "engram_config.json").exists():
        model = Qwen3ForCausalLM.from_pretrained(
            checkpoint, local_files_only=True, torch_dtype=torch.float32,
            attn_implementation="sdpa",
        ).to("cpu").eval()
    else:
        raise ValueError(f"Unsupported or inconsistent checkpoint architecture: {model_type}")
    context_length = settings["context_length"]
    if context_length > model.config.max_position_embeddings:
        raise RuntimeError("V2 rolling evaluation exceeds checkpoint context")
    if model.config.bos_token_id is None or model.config.eos_token_id is None:
        raise RuntimeError("Checkpoint must define BOS and EOS token IDs")

    bos_id = int(model.config.bos_token_id)
    eos_id = int(model.config.eos_token_id)
    pad_id = model.config.pad_token_id
    tokenizer_size = tokenizer.get_vocab_size(with_added_tokens=True)
    suppressed_ids = generation_suppression_ids(
        tokenizer_json, tokenizer_size, model.config.vocab_size, eos_id, pad_id)
    model.generation_config.suppress_tokens = suppressed_ids

    text_nll = 0.0
    text_tokens = 0
    text_bytes = 0
    eos_nll = 0.0
    eos_targets = 0
    complete_documents = 0
    probes = []
    continuations = []
    with torch.inference_mode():
        for document in versioned_fixture["samples"]:
            text_ids = encode_ordinary_text(tokenizer, document["text"]).ids
            if not text_ids:
                raise RuntimeError(f"Empty ordinary-text encoding for fixture document {document['id']}")
            if any(token_id in tokenizer.control_ids for token_id in text_ids):
                raise RuntimeError("Control ID appeared in ordinary-text held-out encoding")
            text_nll += target_nll_sum(
                model, [bos_id], text_ids, context_length, settings["rolling_stride"],
                projection_chunk_size=settings["logit_chunk_size"],
            )
            text_tokens += len(text_ids)
            text_bytes += len(document["text"].encode("utf-8"))
            if not document["truncated"]:
                eos_nll += target_nll_sum(
                    model, [bos_id] + text_ids, [eos_id], context_length,
                    settings["rolling_stride"],
                    projection_chunk_size=settings["logit_chunk_size"],
                )
                eos_targets += 1
                complete_documents += 1

        for probe in settings.get("probes", []):
            scores = [score_choice(model, tokenizer, bos_id, probe["prompt"], choice, settings)
                      for choice in probe["choices"]]
            predicted = max(range(len(scores)), key=scores.__getitem__)
            probes.append({**probe, "mean_answer_token_log_probs": scores,
                           "predicted": predicted, "correct": predicted == probe["answer"]})

        for prompt in settings.get("generation_prompts", []):
            prompt_ids = [bos_id] + encode_ordinary_text(tokenizer, prompt).ids
            inputs = torch.tensor([prompt_ids], dtype=torch.long)
            if is_engram:
                generated = greedy_engram(model, prompt_ids, settings["max_new_tokens"],
                                          eos_id, suppressed_ids)
            else:
                generated = model.generate(
                    inputs, max_new_tokens=settings["max_new_tokens"], do_sample=False,
                    use_cache=True, eos_token_id=eos_id, pad_token_id=pad_id,
                )[0, len(prompt_ids):].tolist()
            continuations.append({
                "prompt": prompt,
                "continuation": tokenizer.decode(generated, skip_special_tokens=True),
                "generated_tokens": len(generated),
                "generated_token_ids": generated,
                "repeated_4gram_fraction": repetition_rate(generated),
                "ended_with_eos": bool(generated and generated[-1] == eos_id),
            })

    if text_tokens == 0 or text_bytes == 0:
        raise RuntimeError("Frozen fixture produced no ordinary text to score")
    step = int(checkpoint.name.removeprefix("step-")) if checkpoint.name.startswith("step-") else 0
    output = Path(output_path) if output_path else (
        RUNTIME / "results-v2" / checkpoint.parent.name / f"{checkpoint.name}.json")
    if output.exists():
        previous = json.loads(output.read_text())
        if (previous.get("protocol_version") != PROTOCOL_VERSION
                or previous.get("checkpoint") != str(checkpoint.resolve())
                or previous.get("fixture_sha256") != fixture_digest):
            raise FileExistsError(f"Refusing to replace a different evaluation report: {output}")
    report = {
        "protocol_version": PROTOCOL_VERSION,
        "encoding_policy": settings["encoding_policy"],
        "checkpoint": str(checkpoint.resolve()),
        "step": step,
        "architecture": "engram" if is_engram else "dense",
        "generation_decoding": "greedy",
        "generation_max_new_tokens": settings["max_new_tokens"],
        "model_dtype": "float32",
        "checkpoint_manifest_sha256": sha256(checkpoint / "manifest.json")
            if (checkpoint / "manifest.json").exists() else None,
        "device": "cpu",
        "cpu_threads": 2,
        "fixture_path": str(fixture_path.resolve()),
        "fixture_sha256": fixture_digest,
        "base_fixture_sha256": sha256(base_path),
        "settings_sha256": settings_digest,
        "context_length": context_length,
        "rolling_stride": settings["rolling_stride"],
        "heldout_documents": len(versioned_fixture["samples"]),
        "heldout_text_tokens": text_tokens,
        "heldout_text_utf8_bytes": text_bytes,
        "text_nll_nats": text_nll,
        "text_bits_per_utf8_byte": text_nll / (math.log(2) * text_bytes),
        "complete_documents": complete_documents,
        "eos_targets": eos_targets,
        "eos_nll_nats": eos_nll,
        "eos_mean_nll_nats": eos_nll / eos_targets if eos_targets else None,
        "cloze_accuracy": (sum(item["correct"] for item in probes) / len(probes)) if probes else None,
        "cloze_questions": len(probes),
        "repeated_4gram_fraction": (
            sum(item["repeated_4gram_fraction"] for item in continuations) / len(continuations)
            if continuations else None
        ),
        "probes": probes,
        "continuations": continuations,
        "generation_suppressed_id_count": len(suppressed_ids),
        "generation_suppressed_ids_sha256": hashlib.sha256(
            json.dumps(suppressed_ids, separators=(",", ":")).encode()).hexdigest(),
        "elapsed_seconds": time.monotonic() - start_time,
        "complete": True,
        "wandb_status": "pending" if wandb_project or wandb_run else "not_requested",
    }
    atomic_json(output, report)
    print(json.dumps(report, indent=2), flush=True)

    if wandb_run:
        return publish_to_training_run(report, output, wandb_run, train_pid=wandb_train_pid)
    if wandb_project:
        try:
            import wandb
            run_id = hashlib.sha256((report["checkpoint"] + fixture_digest + settings_digest +
                                     PROTOCOL_VERSION).encode()).hexdigest()[:16]
            with wandb.init(
                project=wandb_project, entity=wandb_entity, id=run_id, resume="allow",
                name=f"light-v2-{checkpoint.name}", job_type="evaluation",
                group="fineweb-edu-338m-8k", dir=str(RUNTIME.resolve()), save_code=False,
                config={"protocol_version": PROTOCOL_VERSION,
                        "encoding_policy": settings["encoding_policy"],
                        "checkpoint": report["checkpoint"], "fixture_sha256": fixture_digest,
                        "settings_sha256": settings_digest},
                settings=wandb.Settings(disable_git=True, disable_code=True, console="off"),
            ) as run:
                run.log({
                    "progress/optimizer_step": step,
                    "quality/v2_text_bits_per_utf8_byte": report["text_bits_per_utf8_byte"],
                    "quality/v2_text_nll_nats": text_nll,
                    "quality/v2_eos_mean_nll_nats": report["eos_mean_nll_nats"],
                    "quality/v2_cloze_accuracy": report["cloze_accuracy"],
                    "quality/v2_repeated_4gram_fraction": sum(
                        item["repeated_4gram_fraction"] for item in continuations
                    ) / len(continuations),
                })
                report["wandb_url"] = run.url
            report["wandb_status"] = "uploaded"
        except Exception as error:
            report["wandb_status"] = "failed"
            report["wandb_error"] = f"{type(error).__name__}: {error}"
            print(f"W&B upload failed; completed local metrics are preserved: {report['wandb_error']}",
                  file=sys.stderr, flush=True)
        atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--base-fixture", type=Path,
                        help="Use this frozen v1 fixture; no source data is opened")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wandb-project", help="Optionally upload scalar v2 metrics as a separate run")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run", help="Attach HTML/JSON to existing ENTITY/PROJECT/RUN_ID; no second logger")
    parser.add_argument("--wandb-train-pid", type=int,
                        help="Publish native Tables through this live Linux trainer's W&B service")
    args = parser.parse_args()
    if args.wandb_run and args.wandb_project:
        parser.error("--wandb-run and --wandb-project are mutually exclusive")
    if args.wandb_train_pid is not None and (not args.wandb_run or args.wandb_train_pid <= 0):
        parser.error("--wandb-train-pid requires --wandb-run and a positive PID")
    if args.wandb_run and args.output and args.output.exists():
        report = json.loads(args.output.read_text())
        settings_digest = hashlib.sha256(json.dumps(
            load_settings(args.config), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (report.get("complete") is True and report.get("protocol_version") == PROTOCOL_VERSION
                and report.get("checkpoint") == str(args.checkpoint.resolve())
                and report.get("settings_sha256") == settings_digest):
            report = publish_to_training_run(report, args.output, args.wandb_run, train_pid=args.wandb_train_pid)
            raise SystemExit(0 if report["wandb_status"] == "uploaded" else 1)
    report = evaluate(args.checkpoint, args.config, base_fixture_path=args.base_fixture,
             output_path=args.output, wandb_project=args.wandb_project,
             wandb_entity=args.wandb_entity, wandb_run=args.wandb_run, wandb_train_pid=args.wandb_train_pid)
    if args.wandb_run and report["wandb_status"] != "uploaded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
