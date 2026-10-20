"""Single-node DDP pretraining; global token batches and rank-local restart state."""

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from ..common.paths import configure_storage
from ..data_io.blocks import DataWaitStopped, GrowingTokenBlocks, TokenBlocks
from ..data_io.download import sha256
from ..data_io.transfer import atomic_json
from .continuation import validate_config, validate_handoff
from .cuda import build_model, learning_rate, save_checkpoint
from .objective import LOSS_POLICY, chunked_loss, normalize_accumulated_gradients, target_mask
from .optimizers import build_optimizer
from .screen import calibrate_memory, evaluate_branch, prune_checkpoints


class LossModel(nn.Module):
    """Enter DDP through forward even though vocabulary loss uses the hidden-state API."""
    def __init__(self, model, chunk_size):
        super().__init__()
        self.network = model
        self.chunk_size = chunk_size

    def forward(self, x, y, mask):
        if hasattr(self.network, "training_loss"):
            return self.network.training_loss(x, y, self.chunk_size, mask)
        return chunked_loss(self.network, x, y, self.chunk_size, loss_mask=mask, reduction="sum")


def local_accumulation(global_accumulation, world_size):
    if world_size < 1 or global_accumulation < 1 or global_accumulation % world_size:
        raise ValueError("Global accumulation must be divisible by GPU count")
    return global_accumulation // world_size


def local_rows(values, rank, world_size, micro_batch):
    if len(values) != world_size * micro_batch or not 0 <= rank < world_size:
        raise ValueError("Global microbatch/rank mismatch")
    return values[rank * micro_batch:(rank + 1) * micro_batch]


def fixed_validation_config(config):
    """Score the same frozen blocks regardless of training batch partitioning."""
    return {**config, "profile": "engram-confirmation", "micro_batch_size": 1,
            "eval_batches": max(128, config["eval_batches"])}


def compute_context(device):
    if device.type == "cpu":
        return nullcontext(), nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return torch.autocast("cuda", dtype=torch.bfloat16), sdpa_kernel(
        [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])


def accumulate_gradients(wrapper, model, batches, device):
    """DDP averages summed NLL gradients; divide by global targets / world size."""
    world = dist.get_world_size() if dist.is_initialized() else 1
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    for index, values in enumerate(batches):
        batch = torch.as_tensor(values, device=device)
        x, y = batch[:, :-1], batch[:, 1:]
        mask = target_mask(x, y, model.config.eos_token_id, model.config.pad_token_id)
        sync = wrapper.no_sync() if isinstance(wrapper, DDP) and index < len(batches) - 1 else nullcontext()
        autocast, attention = compute_context(device)
        with sync:
            with autocast, attention:
                result = wrapper(x, y, mask)
                loss, primary = result if isinstance(result, tuple) else (result, result.detach())
            loss.backward()
        totals[0] += loss.detach().double()
        totals[1] += primary.detach().double()
        totals[2] += mask.sum()
    if dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    if not torch.isfinite(totals).all() or totals[2] < 1:
        raise RuntimeError("Invalid globally accumulated loss/target count")
    normalize_accumulated_gradients(model, float(totals[2]) / world)
    model.last_objective_loss = float(totals[0]) / float(totals[2])
    return float(totals[1]) / float(totals[2]), int(totals[2])


def inspect_corpus(token_dir, sequence_length, tokenizer_hash, eos=7, pad=0):
    """Verify immutable payloads and count usable, boundary-masked targets in one pass."""
    summary = json.loads((token_dir / "manifest.json").read_text())
    if summary.get("complete") is not True or summary.get("status") == "failed":
        raise ValueError("Distributed training requires a completed prepared corpus")
    if summary["tokenizer_sha256"] != tokenizer_hash:
        raise ValueError("Prepared corpus tokenizer mismatch")
    records, total = {}, 0
    for path in sorted(token_dir.glob("*/manifest.json")):
        if path.parent.name.endswith(".partial"):
            continue
        meta = json.loads(path.read_text())
        if meta["tokenizer_sha256"] != tokenizer_hash or meta["source_revision"] != summary["source_revision"]:
            raise ValueError("Mixed corpus identity")
        for split in ("train", "validation"):
            binary = path.parent / f"{split}.bin"
            if binary.stat().st_size != 2 * meta["splits"][split]["tokens"] or sha256(binary) != meta["splits"][split]["sha256"]:
                raise ValueError(f"Corrupt prepared payload: {binary}")
        length = meta["splits"]["train"]["tokens"]
        blocks = max(0, (length - 1) // sequence_length)
        if blocks:
            values = np.memmap(path.parent / "train.bin", dtype="<u2", mode="r")
            # Limit temporary boolean arrays to approximately 16 MiB each.
            counts = []
            batch_blocks = max(1, 8_388_608 // sequence_length)
            for first in range(0, blocks, batch_blocks):
                a, b = first * sequence_length, min(blocks, first + batch_blocks) * sequence_length
                x, y = values[a:b], values[a + 1:b + 1]
                mask = (x != eos) & (x != pad) & (y != pad)
                counts.extend(mask.reshape(-1, sequence_length).sum(1).tolist())
            count = sum(counts)
        else:
            counts, count = [], 0
        total += count
        records[path.parent.name] = {"manifest_sha256": sha256(path), "valid_targets": count}
    if not records:
        raise ValueError("No prepared chunks")
    return {"valid_targets_per_pass": total, "chunks": records,
            "summary_sha256": sha256(token_dir / "manifest.json"), "raw_train_tokens": summary["train_tokens"]}


def enforce_coverage(target_tokens, inventory):
    """Only the nominal 10BT starting stage permits a small, explicitly reported tail replay."""
    coverage = inventory["valid_targets_per_pass"]
    limit = int(coverage * 1.03) if target_tokens <= 10_000_000_000 else coverage
    if target_tokens > limit:
        raise ValueError(f"Target {target_tokens:,} exceeds prepared one-pass coverage {coverage:,}; "
                         "prepare/admit more data before growing the run. The 10BT sample is not a 50B corpus.")
    return {"one_pass_valid_targets": coverage, "nominal_tail_replay_targets": max(0, target_tokens - coverage),
            "content_deduplicated": False}


def restart_rng(device):
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def validate_mixture_checkpoint(state,config,*,engram,world_size):
    """Keep the authorized same-context Adam/RNG recipe through mixture handoff."""
    if not engram or world_size!=4 or state.get('distributed',{}).get('world_size')!=world_size:
        raise ValueError('Mixture continuation requires the same four Engram ranks')
    if config['sequence_length']!=8192 or config['target_tokens']!=50_000_000_000 or (
            config['learning_rate']!=.0003 or config.get('schedule')!='constant-after-warmup-v1'):
        raise ValueError('Mixture continuation requires the held-rate 8K cumulative 50B recipe')
    preserve = ('seed','sequence_length','target_tokens','learning_rate','min_learning_rate',
                'warmup_tokens','weight_decay','betas','grad_clip','precision',
                'gradient_checkpointing','loss_policy','schedule')
    if any(state['config'].get(k)!=config.get(k) for k in preserve) or (
            state['config']['micro_batch_size']*state['config']['gradient_accumulation_steps'] !=
            config['micro_batch_size']*config['gradient_accumulation_steps']):
        raise ValueError('Mixture continuation changed the optimizer or global batch recipe')
    slots = state['optimizer']['state']
    if not slots:
        raise ValueError('Mixture continuation requires saved Adam moments')
    for slot in slots.values():
        if not {'step','exp_avg','exp_avg_sq'}.issubset(slot):
            raise ValueError('Mixture continuation requires complete saved Adam moments')
        if not all(torch.is_tensor(slot[k]) for k in ('exp_avg','exp_avg_sq')) or (
                slot['exp_avg'].shape!=slot['exp_avg_sq'].shape):
            raise ValueError('Mixture continuation requires matched Adam moment tensors')
        if any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in slot.values()):
            raise ValueError('Nonfinite saved Adam state')
        if torch.is_tensor(slot['step']) and slot['step'].numel()!=1 or (
                not math.isfinite(float(slot['step'])) or float(slot['step'])<1):
            raise ValueError('Invalid saved Adam update counter')
    rank_rng = state['distributed'].get('rank_rng',[])
    if len(rank_rng)!=world_size or any(
            not isinstance(r,dict) or any(not torch.is_tensor(r.get(k)) or r[k].dtype!=torch.uint8 or
                r[k].ndim!=1 or not r[k].numel() for k in ('torch','cuda')) for r in rank_rng):
        raise ValueError('Mixture continuation requires all saved rank RNG payloads')


def restore_growth_reader(reader,data,tokens,target_tokens,*,continuation=False,
                          parent_catalog=None,parent_catalog_sha256=None,minimum_ready=1_000_000):
    """Restore historical exposure before checking remaining acquisition coverage."""
    if parent_catalog is not None:
        if not continuation or data.get('mode')!='mixture-growth-v1':
            raise ValueError('Mixture migration requires explicit continuation of mixture state')
        reader.migrate_mixture(data,parent_catalog,parent_catalog_sha256=parent_catalog_sha256)
    elif data.get('mode')=='growing':
        if not continuation:
            raise ValueError('Initial growth cursor migration must be an explicit continuation')
        reader.migrate(data,tokens)
    else:
        reader.load_state_dict(data)
    if reader.parent_tokens+sum(reader.targets.values())!=tokens:
        raise ValueError('Growth source/historical token counters do not reconcile')
    return reader.coverage(target_tokens,minimum_ready=minimum_ready)


def prepare_growth_validation(catalog,config,saved,*,new_sources=(),require_saved=False):
    """Preserve frozen slices and add fixtures only for explicitly added sources."""
    expected = set(catalog['sources'])
    new_sources = set(new_sources)
    if not new_sources.issubset(expected):
        raise ValueError('New validation source is absent from the catalog')
    if saved is None:
        if require_saved:
            raise ValueError('Missing saved frozen validation fixtures and baselines')
        result = {'sources':{},'baseline':None}
        select = expected
    else:
        result = json.loads(json.dumps(saved))
        previous = expected-new_sources
        if set(result['sources'])!=previous or not isinstance(result.get('baseline'),dict) or (
                set(result['baseline'])!={'original',*previous}):
            raise ValueError('Saved frozen validation source or baseline set mismatch')
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in result['baseline'].values()):
            raise ValueError('Nonfinite or invalid saved validation baseline')
        select = new_sources
    required = config['eval_batches']*config['micro_batch_size']
    for source in sorted(select):
        entry = catalog['sources'][source]
        selected,blocks = {},0
        for name,chunk in sorted(entry['chunks'].items()):
            meta = json.loads((Path(entry['directory'])/name/'manifest.json').read_text())
            count = max(0,(meta['splits']['validation']['tokens']-1)//config['sequence_length'])
            if count:
                selected[name] = chunk['manifest_sha256']
                blocks += count
            if blocks>=required:
                break
        if blocks<required:
            raise ValueError(f'Insufficient frozen {source} validation')
        result['sources'][source] = selected
    for source,selected in result['sources'].items():
        entry = catalog['sources'][source]
        blocks = 0
        for name,hash_ in selected.items():
            if Path(name).name!=name or name.endswith('.partial') or (
                    entry['chunks'].get(name,{}).get('manifest_sha256')!=hash_):
                raise ValueError('Frozen source validation inventory changed')
            directory = Path(entry['directory'])/name
            manifest = directory/'manifest.json'
            meta = json.loads(manifest.read_text())
            info = meta['splits']['validation']
            if sha256(manifest)!=hash_ or meta['source_revision']!=catalog['recipe']['sources'][source]['revision'] or (
                    meta['tokenizer_sha256']!=catalog['recipe']['tokenizer_sha256'] or
                    (directory/'validation.bin').stat().st_size!=info['tokens']*2 or
                    sha256(directory/'validation.bin')!=info['sha256']):
                raise ValueError('Frozen source validation changed')
            blocks += max(0,(info['tokens']-1)//config['sequence_length'])
        if blocks<required:
            raise ValueError(f'Insufficient frozen {source} validation')
    return result


def update_growth_baseline(validation,original,losses,*,new_sources=()):
    expected = set(validation['sources'])
    if set(losses)!=expected or any(not math.isfinite(v) or v<=0 for v in [original,*losses.values()]):
        raise ValueError('Missing or nonfinite growth validation loss')
    if validation['baseline'] is None:
        validation['baseline'] = {'original':original,**losses}
    else:
        missing = {'original',*expected}-set(validation['baseline'])
        if missing-set(new_sources) or 'original' in missing:
            raise ValueError('Cannot rebase existing validation slices')
        for source in missing:
            validation['baseline'][source] = losses[source]


def evaluate_growth_sources(model,catalog,config,validation,device):
    # Validation must not advance the saved CPU/CUDA training RNG streams.
    devices = [device.index] if device.type=='cuda' else []
    with torch.random.fork_rng(devices=devices):
        return {source:evaluate_branch(model,Path(catalog['sources'][source]['directory']),
                    config,list(selected),device.type)['loss']
                for source,selected in validation['sources'].items()}


def attach_memory(dense, tokenizer_json):
    from .engram import EngramQwen3ForCausalLM, LOOKUP_V2, build_token_id_map
    return EngramQwen3ForCausalLM.from_dense(dense, residual_scale=1.0,
        lookup_version=LOOKUP_V2, hash_heads=4,
        token_id_map=build_token_id_map(tokenizer_json, dense.config.vocab_size))


def verify_checkpoint(path, manifest_path=None):
    manifest_path = manifest_path or path / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    required = {"model.safetensors", "training_state.pt", "config.json", "run_config.json", "tokenizer.json"}
    if manifest.get("schema_version") != 1 or not required.issubset(manifest["files"]):
        raise ValueError("Checkpoint manifest does not cover the restart payload")
    for name, identity in manifest["files"].items():
        file = path / name
        if Path(name).name != name or file.stat().st_size != identity["bytes"] or sha256(file) != identity["sha256"]:
            raise ValueError(f"Checkpoint publication failed verification: {name}")
    return sha256(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, required=True)
    parser.add_argument("--growth-catalog", type=Path)
    parser.add_argument("--mixture-parent-catalog", type=Path)
    parser.add_argument("--mixture-parent-catalog-sha256")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    transition = parser.add_mutually_exclusive_group()
    transition.add_argument("--resume", type=Path)
    transition.add_argument("--continue-from", type=Path)
    transition.add_argument("--init-from", type=Path)
    parser.add_argument("--parent-manifest", type=Path, help="Source-side checksum export for a legacy parent without publication metadata")
    parser.add_argument("--engram", action="store_true", help="Explicitly enable the unproven normalized multi-head candidate")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-keep", type=int, default=2)
    parser.add_argument("--checkpoint-reserve-gib", type=int, default=32,
                        help="Free write buffer after checkpoint staging (minimum 8 GiB)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    if bool(args.mixture_parent_catalog)!=bool(args.mixture_parent_catalog_sha256):
        parser.error('Use both mixture parent catalog flags')
    if args.mixture_parent_catalog:
        if not args.growth_catalog or not args.continue_from or not args.engram:
            parser.error('Mixture migration requires growth catalog, continue-from and Engram')
        if len(args.mixture_parent_catalog_sha256)!=64 or any(
                c not in '0123456789abcdef' for c in args.mixture_parent_catalog_sha256):
            parser.error('Mixture parent catalog SHA-256 must be 64 lowercase hexadecimal characters')
    if args.growth_catalog and (args.init_from or not (args.resume or args.continue_from)):
        parser.error("Growth requires state-preserving continuation/resume")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("Positive new-token cap required")
    if args.parent_manifest and not (args.resume or args.continue_from or args.init_from):
        parser.error("A parent manifest requires a checkpoint transition")
    if args.checkpoint_keep < 2 or args.checkpoint_reserve_gib < 8 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Keep at least two checkpoints and use a positive step cap")
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config.get("loss_policy") != LOSS_POLICY or config["precision"] != "bf16":
        raise ValueError("Cloud path requires boundary-v2 and BF16 compute")
    world, rank, local_rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    accumulation = local_accumulation(config["gradient_accumulation_steps"], world)
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        raise RuntimeError("Requested GPU count unavailable")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 GPU required")
    torch.set_num_threads(2)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    test_sum = torch.ones(1, device=device)
    dist.all_reduce(test_sum)
    if test_sum.item() != world:
        raise RuntimeError("NCCL rank participation check failed")
    root = configure_storage()
    logs = root / "logs" / config["run_name"]
    logs.mkdir(parents=True, exist_ok=True)
    status, tracker, step, tokens, positions = {"phase": "preflight", "rank": rank}, None, 0, 0, 0
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    def report(**values):
        status.update(values, updated_unix=time.time())
        if rank == 0:
            atomic_json(logs / "status.json", status)
    def track(values):
        nonlocal tracker
        if tracker:
            try:
                # Training, validation and sample tables share an optimizer
                # step. Commit when the next step arrives, not on the first
                # metric, or W&B drops the later same-step records.
                tracker.log(values, step=step, commit=False)
            except Exception as error:
                report(wandb_error=str(error))
                tracker = None
    try:
        if rank == 0 and (args.checkpoint_dir / config["run_name"] / "latest.json").exists() and not args.resume:
            raise ValueError("Existing run checkpoint: resume explicitly or choose a new run name")
        report(pid=os.getpid(), world_size=world, architecture="engram" if args.engram else "dense", config=config)
        tokenizer = json.loads((root / "tokenizer/source_manifest.json").read_text())
        if sha256(root / "tokenizer/tokenizer.json") != tokenizer["tokenizer_sha256"]:
            raise ValueError("Tokenizer payload checksum mismatch")
        if args.growth_catalog:
            from ..data_io.growth_reader import MixtureTokenBlocks
            reader = MixtureTokenBlocks(args.growth_catalog, config["sequence_length"], seed=config["seed"])
            validation_dir = Path(reader.catalog['recipe']['legacy']['root'])
            if reader.catalog['recipe']['tokenizer_sha256'] != tokenizer['tokenizer_sha256']:
                raise ValueError('Growth tokenizer mismatch')
        else:
            validation_dir = args.token_dir
        inventory = inspect_corpus(validation_dir, config["sequence_length"], tokenizer["tokenizer_sha256"],
                                   tokenizer["eos_token_id"], tokenizer["pad_token_id"]) if rank == 0 else None
        payload = [inventory]
        dist.broadcast_object_list(payload, src=0)
        inventory = payload[0]
        if args.growth_catalog:
            legacy = reader.catalog['recipe']['legacy']
            if sha256(Path(legacy['inventory'])) != legacy['inventory_sha256'] or json.loads(Path(legacy['inventory']).read_text()) != inventory:
                raise ValueError('Growth legacy receipt differs from verified corpus')
        # A new mixture catalog covers the prospective remainder, not a fresh
        # 40B stage. Restore historical counters before checking its coverage.
        coverage = None if args.growth_catalog else enforce_coverage(config["target_tokens"], inventory)
        if not args.growth_catalog:
            reader = GrowingTokenBlocks(args.token_dir, config["sequence_length"], seed=config["seed"])
        bound_chunks = {k:v["manifest_sha256"] for k,v in inventory["chunks"].items()}
        if not args.growth_catalog and reader.known != bound_chunks:
            raise ValueError("Reader inventory differs from verified corpus")
        if rank == 0:
            atomic_json(logs / "inventory.json", inventory)
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        mc = json.loads(Path(config["model_config"]).read_text())
        custom = mc.get("model_type") in ("microllama_hybrid_engram", "microllama_dense_engram")
        architecture = mc["model_type"] if custom else ("engram" if args.engram else "dense")
        report(architecture=architecture)
        if custom and not args.engram:
            raise ValueError("This experiment configuration requires explicit --engram")
        if mc["max_position_embeddings"] < config["sequence_length"] or any(mc.get(k) != tokenizer[k] for k in ("eos_token_id", "pad_token_id", "bos_token_id")):
            raise ValueError("Model context or tokenizer controls differ")
        if mc["vocab_size"] <= tokenizer["max_token_id"]:
            raise ValueError("Model vocabulary too small")
        if args.engram and config["gradient_checkpointing"]:
            raise ValueError("Engram decoder checkpointing is not qualified; use the tested 8K shape")
        torch.manual_seed(config["seed"])
        parent = args.resume or args.continue_from or args.init_from
        state, lineage, calibration, new_memory = None, None, None, False
        growth_validation = None
        new_growth_sources = set()
        growth_validation_required = False
        if parent:
            if rank == 0:
                report(parent_manifest_sha256=verify_checkpoint(parent, args.parent_manifest))
            saved_mc = json.loads((parent / "config.json").read_text())
            saved_engram = saved_mc.get("model_type") in (
                "engram_qwen3", "microllama_hybrid_engram", "microllama_dense_engram")
            if custom and saved_mc.get("model_type") != architecture:
                raise ValueError("Checkpoint architecture differs from the experiment configuration")
            if saved_engram != args.engram and not (args.init_from and args.engram and not saved_engram):
                raise ValueError("Architecture changes require a separate experiment; --engram must match checkpoint")
            if saved_engram and saved_mc.get("engram_lookup_version") != "normalized-multihead-v2":
                raise ValueError("Cloud Engram requires the revised, qualified lookup architecture")
            if sha256(parent / "tokenizer.json") != tokenizer["tokenizer_sha256"]:
                raise ValueError("Checkpoint tokenizer mismatch")
            for key, value in mc.items():
                if key not in ("model_type", "architectures") and saved_mc.get(key) != value:
                    raise ValueError(f"Checkpoint model mismatch: {key}")
            if custom:
                from .hybrid import load_model
                model = load_model(parent)
            elif saved_engram:
                from .engram import load_engram_model
                model = load_engram_model(parent)
            else:
                from transformers import Qwen3ForCausalLM
                model = Qwen3ForCausalLM.from_pretrained(parent, local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation="sdpa")
                if args.engram:
                    model = attach_memory(model, root / "tokenizer/tokenizer.json")
                    new_memory = True
            if not args.init_from:
                state = torch.load(parent / "training_state.pt", map_location="cpu", weights_only=True)
                if args.mixture_parent_catalog or state['data'].get('mixture_transition'):
                    validate_mixture_checkpoint(state,config,engram=args.engram,world_size=world)
                if args.resume and state["config"] != config:
                    raise ValueError("Exact resume config mismatch; use an explicit continuation for changes")
                if args.continue_from:
                    if state["config"].get("profile") == "engram-confirmation":
                        from .continuation import promote_confirmation_state
                        state, changes = promote_confirmation_state(state, config, saved_mc, mc)
                    else:
                        changes = validate_handoff(state["config"], config, saved_mc, mc)
                    lineage = {"parent_checkpoint": str(parent), "parent_step": state["step"], "parent_tokens": state["tokens"],
                               "changes": changes, "historical_loss_policy": state["config"].get("loss_policy", "legacy-unmasked-v1"),
                               "world_size_transition": [state.get("distributed", {}).get("world_size", 1), world],
                               "note": "Optimizer and global reader preserved; floating-point reduction order changes across GPU counts"}
                else:
                    lineage = state.get("lineage")
                if args.resume and not args.growth_catalog and state["data"]["known"] != reader.known:
                    raise ValueError("Exact resume inventory differs; use continuation for explicitly added chunks")
                if args.growth_catalog:
                    old_sources = set(state['data'].get('sources',{}))
                    growth_validation_required = state['data'].get('mode')!='growing'
                    coverage = restore_growth_reader(reader,state['data'],state['tokens'],config['target_tokens'],
                        continuation=bool(args.continue_from),parent_catalog=args.mixture_parent_catalog,
                        parent_catalog_sha256=args.mixture_parent_catalog_sha256)
                    if args.mixture_parent_catalog:
                        new_growth_sources = set(reader.catalog['sources'])-old_sources
                        lineage['mixture_transition'] = {**reader.mixture_transition,'recipe_id':reader.recipe_id}
                else:
                    reader.load_state_dict(state["data"])
                step, tokens = state["step"], state["tokens"]
                positions = state.get("input_positions", tokens)
                if args.max_steps is not None and args.max_steps <= step:
                    raise ValueError("Total-step cap must exceed the starting checkpoint step")
                calibration = state.get("distributed", {}).get("memory_calibration") or state.get("provenance", {}).get("memory_calibration")
            else:
                lineage = {"parent_checkpoint": str(parent), "weights_only": True, "optimizer_and_cursor_reset": True}
            if lineage is not None and not args.resume:
                lineage["parent_manifest_sha256"] = sha256(args.parent_manifest or parent / "checkpoint_manifest.json")
        else:
            if custom:
                from .hybrid import build_model as build_experiment
                model = build_experiment(mc, root / "tokenizer/tokenizer.json")
                new_memory = True
            else:
                model = build_model(mc, gradient_checkpointing=False)
            if args.engram and not custom:
                model = attach_memory(model, root / "tokenizer/tokenizer.json")
                new_memory = True
        model = model.float().to(device)
        model.config.use_cache = False
        model.config._attn_implementation = "sdpa"
        if mc.get("model_type") == "microllama_hybrid_engram":
            from .hybrid import require_fast_kernels
            require_fast_kernels(model)
        allowance = int(sum(p.numel() for p in model.parameters()) * 12 * 1.1)
        existing_count = sum((p / "checkpoint_manifest.json").is_file()
                             for p in (args.checkpoint_dir / config["run_name"]).glob("step-*")
                             if not p.name.endswith(".partial"))
        additional_copies = args.checkpoint_keep + 1 - min(args.checkpoint_keep, existing_count)
        if shutil.disk_usage(args.checkpoint_dir).free < args.checkpoint_reserve_gib * 2**30 + allowance * additional_copies:
            raise RuntimeError("Insufficient disk space for retained checkpoints, atomic staging and configured reserve")
        if config["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if new_memory:
            cursor = reader.state_dict()
            reference = reader.next_batch(1)
            calibration = calibrate_memory(model, torch.as_tensor(reference[:, :-1], device=device), .005)
            reader.load_state_dict(cursor)
        optimizer = build_optimizer(model, fused=True, adam_lr=config["learning_rate"],
                                    weight_decay=config["weight_decay"], memory_lr_multiplier=5 if args.engram else 1)
        for group in optimizer.param_groups:
            group["betas"] = tuple(config["betas"])
        if state:
            optimizer.load_state_dict(state["optimizer"])
            restart = state.get("distributed")
            if restart and restart["architecture"] != architecture:
                raise ValueError("Restart architecture differs")
            if args.resume and restart and restart["world_size"] != world:
                raise ValueError("Exact resume requires the original GPU count; use continuation")
            if restart and restart["world_size"] == world:
                rng = restart["rank_rng"][rank]
            elif rank == 0:
                rng = {"torch": state["torch_rng"], "cuda": state["cuda_rng"][0]}
            else:
                torch.manual_seed(config["seed"] + rank)
                rng = restart_rng(device)
            shards = state["validation_shards"]
            growth_validation = state.get('growth_validation')
            del state
        else:
            torch.manual_seed(config["seed"] + rank)
            rng = restart_rng(device)
            shards = [name for name in reader.known if reader.metadata[name]["splits"]["validation"]["tokens"] > config["sequence_length"]]
        validation_config = fixed_validation_config(config)
        validation_reader = TokenBlocks(
            validation_dir, config["sequence_length"], split="validation", seed=config["seed"], shards=shards)
        if validation_reader.num_blocks < validation_config["eval_batches"] * validation_config["micro_batch_size"]:
            raise ValueError("Insufficient frozen validation blocks")
        if args.growth_catalog:
            growth_validation = prepare_growth_validation(reader.catalog,validation_config,growth_validation,
                new_sources=new_growth_sources,require_saved=growth_validation_required)
        wrapper = DDP(LossModel(model, config["loss_chunk_size"]), device_ids=[local_rank],
                      broadcast_buffers=False, gradient_as_bucket_view=True)
        restore_rng(rng, device)
        provenance = {"code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "torch": str(torch.__version__), "gpus": [torch.cuda.get_device_name(i) for i in range(world)],
                      "world_size": world, "local_accumulation": accumulation,
                      "global_input_positions_per_update": config["sequence_length"] * config["micro_batch_size"] * config["gradient_accumulation_steps"],
                      "coverage": coverage, "lineage": lineage, "memory_calibration": calibration}
        provenance.update(architecture=architecture, parameters=sum(p.numel() for p in model.parameters()),
                          transformers=__import__("transformers").__version__)
        if rank == 0:
            atomic_json(logs / "provenance.json", provenance)
            if not args.no_wandb:
                try:
                    import wandb
                    tracking_path = logs / "wandb-id.json"
                    if not tracking_path.exists():
                        atomic_json(tracking_path, {"id": uuid.uuid4().hex[:16]})
                    tracking_id = json.loads(tracking_path.read_text())["id"]
                    tracker = wandb.init(project="MicroLlama2", name=config["run_name"], dir=str(logs),
                        config={**config, **provenance}, id=tracking_id,
                        resume="allow", group=os.environ.get("WANDB_RUN_GROUP"),
                        settings=wandb.Settings(disable_git=True, disable_code=True, init_timeout=30))
                    report(wandb_url=tracker.url)
                except Exception as error:
                    report(wandb_error=str(error))
        dist.barrier()
        def evaluate():
            dist.barrier()
            if rank == 0:
                with torch.random.fork_rng(devices=[device.index]):
                    measured = evaluate_branch(model, validation_dir, validation_config, shards, "cuda")
                    with (logs / "validation.jsonl").open("a") as file:
                        file.write(json.dumps({"step": step, "tokens": tokens, "validation": measured}) + "\n")
                    report(validation_loss=measured["loss"])
                    track({"validation/loss": measured["loss"], "tokens": tokens, **{k:measured[k] for k in ("memory_relative_rms", "disabled_minus_enabled_loss") if k in measured}})
                    if custom:
                        from ..evaluation.samples import fixed_samples
                        samples = fixed_samples(model, root / "tokenizer/tokenizer.json",
                                                Path("configs/evaluation/lightweight.json"))
                        quality = logs / "quality" / f"step-{step:08d}.json"
                        quality.parent.mkdir(parents=True, exist_ok=True)
                        atomic_json(quality, {"step":step,"tokens":tokens,"samples":samples})
                        if tracker:
                            import wandb
                            track({"quality/fixed_prompts":wandb.Table(
                                columns=["step","tokens","prompt","continuation"],
                                data=[[step,tokens,row["prompt"],row["continuation"]] for row in samples])})
                    if args.growth_catalog:
                        source_losses = evaluate_growth_sources(model,reader.catalog,validation_config,growth_validation,device)
                        update_growth_baseline(growth_validation,measured['loss'],source_losses,new_sources=new_growth_sources)
                        atomic_json(logs / 'growth-validation.json', {'step':step, 'tokens':tokens,
                            'baseline':growth_validation['baseline'], 'original':measured['loss'], 'sources':source_losses,
                            'frozen_shards':growth_validation['sources']})
                        track({f'validation/{source}/loss':loss for source,loss in source_losses.items()})
            dist.barrier()
        def publish():
            states = [None] * world
            dist.all_gather_object(states, restart_rng(device))
            cursors = [None] * world
            if args.growth_catalog:
                shared_validation = [growth_validation if rank == 0 else None]
                dist.broadcast_object_list(shared_validation, src=0)
                # Rank zero owns global data selection. All ranks receive the same
                # batches; a producer publication cannot race independent readers.
                shared = [reader.state_dict() if rank == 0 else None]
                dist.broadcast_object_list(shared, src=0)
                cursors = [shared[0]] * world
            else:
                dist.all_gather_object(cursors, reader.state_dict())
            if any(cursor != cursors[0] for cursor in cursors):
                raise RuntimeError("Rank global cursors diverged")
            if rank == 0:
                if shutil.disk_usage(args.checkpoint_dir).free < args.checkpoint_reserve_gib * 2**30 + allowance:
                    raise RuntimeError("Insufficient free space for publication plus configured reserve")
                saved = save_checkpoint(root, config["run_name"], model, optimizer, reader, config, step, tokens,
                    shards, args.checkpoint_dir, extra_state={"input_positions": positions, "lineage": lineage,
                        **({'growth_validation':shared_validation[0]} if args.growth_catalog else {}),
                        "distributed": {"schema_version": 1, "architecture": architecture,
                                        "world_size": world, "rank_rng": states, "memory_calibration": calibration}})
                prune_checkpoints(saved.parent, args.checkpoint_keep)
                report(checkpoint=str(saved))
            dist.barrier()
        evaluate()
        model.train()
        torch.cuda.reset_peak_memory_stats(device)
        report(phase="training", step=step, tokens=tokens)
        last_saved = step if args.resume else -1
        updates = 0
        starting_tokens = tokens
        def growth_batch():
            while True:
                flag = torch.tensor(int(stopping), device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if flag.item():
                    raise DataWaitStopped('Stopped while awaiting admitted growth data')
                values = reader.try_batch(world * config['micro_batch_size']) if rank == 0 else None
                ready = torch.tensor(int(values is not None) if rank == 0 else 0, device=device)
                dist.broadcast(ready, src=0)
                if ready.item():
                    batch = torch.as_tensor(values, device=device) if rank == 0 else torch.empty(
                        (world * config['micro_batch_size'], config['sequence_length'] + 1), dtype=torch.long, device=device)
                    dist.broadcast(batch, src=0)
                    return batch
                if rank == 0:
                    report(phase='waiting-for-growth-data')
                time.sleep(2)
        while tokens < config["target_tokens"] and (args.max_steps is None or step < args.max_steps) and (
                args.max_new_tokens is None or tokens - starting_tokens < args.max_new_tokens):
            flag = torch.tensor(int(stopping), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if flag.item():
                break
            started = time.monotonic()
            before_source_targets = sum(reader.targets.values()) if args.growth_catalog and rank == 0 else 0
            before_cursor = reader.state_dict() if args.growth_catalog and rank == 0 else None
            try:
                batches = [local_rows(growth_batch() if args.growth_catalog else reader.next_batch(world * config["micro_batch_size"]), rank, world,
                                      config["micro_batch_size"]) for _ in range(accumulation)]
            except DataWaitStopped:
                if before_cursor is not None:
                    reader.load_state_dict(before_cursor)
                break
            if not args.growth_catalog and reader.known != bound_chunks:
                raise RuntimeError("Corpus inventory changed during training; stop and continue explicitly after growth")
            optimizer.zero_grad(set_to_none=True)
            loss, valid = accumulate_gradients(wrapper, model, batches, device)
            if args.growth_catalog and rank == 0 and sum(reader.targets.values()) - before_source_targets != valid:
                raise ValueError('Mixture source targets differ from globally normalized loss targets')
            if updates == 0:
                for name, parameter in model.named_parameters():
                    if parameter.dtype != torch.float32 or parameter.grad is None or not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"Missing/nonfinite FP32 parameter gradient: {name}")
                    if ".engram." in name and parameter.grad.abs().sum().item() == 0:
                        raise RuntimeError(f"Inactive memory gradient: {name}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            rate = learning_rate(config, tokens + valid)
            for group in optimizer.param_groups:
                group["lr"] = rate * (5 if group.get("group_name") == "memory_tables" else 1)
            optimizer.step()
            step, tokens = step + 1, tokens + valid
            updates += 1
            positions += provenance["global_input_positions_per_update"]
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.monotonic() - started, device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            peak = torch.tensor(torch.cuda.max_memory_allocated(device) / 2**30, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            if rank == 0:
                metric = {"step": step, "tokens": tokens, "input_positions": positions, "loss": loss,
                          "grad_norm": float(norm), "learning_rate": rate, "valid_targets_per_second": valid / float(elapsed),
                          "max_rank_peak_allocated_gib": float(peak)}
                if hasattr(model, "training_loss"):
                    metric.update(objective_loss=model.last_objective_loss,
                                  mtp_weighted_loss=model.last_objective_loss - loss)
                if args.growth_catalog:
                    metric.update({f'sources/{source}/valid_targets': count for source,count in reader.targets.items()})
                    metric['sources/replay_targets'] = 0
                with (logs / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps(metric) + "\n")
                print(json.dumps(metric), flush=True)
                track(metric)
                report(phase='training', step=step, tokens=tokens, input_positions=positions, loss=loss)
                if updates == 3:
                    report(qualification_passed=True, qualification_step=step)
            if step % config["eval_every"] == 0:
                evaluate()
            if step % config["save_every"] == 0:
                publish()
                last_saved = step
        if step != last_saved:
            evaluate()
            publish()
        report(phase="completed" if tokens >= config["target_tokens"] else "stopped", step=step, tokens=tokens)
    except Exception as error:
        report(phase="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if tracker:
            try:
                tracker.log({}, step=step, commit=True)
                tracker.finish(exit_code=0 if status.get("phase") in ("completed", "stopped") else 1)
            except Exception as error:
                report(wandb_error=str(error))
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
