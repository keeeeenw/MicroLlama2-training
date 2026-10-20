"""Deterministically shuffled, memory-mapped token blocks with exact resume."""
import hashlib
import json
import time

import numpy as np


class TokenBlocks:
    def __init__(self, directory, sequence_length, split="train", seed=42, shards=None):
        self.sequence_length = sequence_length
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self.maps = []
        counts = []
        manifests = []
        for path in sorted(directory.glob("*/manifest.json")):
            if path.parent.name.endswith(".partial") or (shards is not None and path.parent.name not in shards):
                continue
            meta = json.loads(path.read_text())
            manifests.append(meta)
            bin_path = path.parent / f"{split}.bin"
            n = meta["splits"][split]["tokens"]
            if bin_path.stat().st_size != n * 2:
                raise RuntimeError(f"Token file length mismatch: {bin_path}")
            count = max(0, (n - 1) // sequence_length)
            if count:
                self.maps.append(np.memmap(bin_path, mode="r", dtype="<u2"))
                counts.append(count)
        if not counts:
            raise RuntimeError(f"No {split} blocks of length {sequence_length}; prepare more data")
        identities = {(m["source_revision"], m["tokenizer_sha256"]) for m in manifests}
        if len(identities) != 1:
            raise RuntimeError("Mixed dataset revisions or tokenizers")
        self.fingerprint = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()
        self.ends = np.cumsum(counts)
        self.num_blocks = int(self.ends[-1])
        self._shuffle()

    def _shuffle(self):
        self.order = np.random.default_rng(self.seed + self.epoch).permutation(self.num_blocks)

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor, "seed": self.seed,
                "sequence_length": self.sequence_length, "fingerprint": self.fingerprint}

    def load_state_dict(self, state):
        for key in ("seed", "sequence_length", "fingerprint"):
            if state[key] != self.state_dict()[key]:
                raise RuntimeError(f"Dataset resume mismatch: {key}")
        self.epoch, self.cursor = state["epoch"], state["cursor"]
        if not 0 <= self.cursor <= self.num_blocks or self.epoch < 0:
            raise RuntimeError("Invalid data cursor")
        self._shuffle()

    def next_batch(self, size):
        rows = []
        for _ in range(size):
            if self.cursor == self.num_blocks:
                self.epoch += 1
                self.cursor = 0
                self._shuffle()
            block = int(self.order[self.cursor])
            self.cursor += 1
            shard = int(np.searchsorted(self.ends, block, side="right"))
            local = block - (int(self.ends[shard - 1]) if shard else 0)
            start = local * self.sequence_length
            rows.append(np.asarray(self.maps[shard][start:start + self.sequence_length + 1], dtype=np.int64))
        return np.stack(rows)


class DataWaitStopped(RuntimeError):
    pass


class GrowingTokenBlocks:
    """Consume each immutable arriving chunk once before any corpus repetition."""
    def __init__(self, directory, sequence_length, seed=42):
        self.directory = directory
        self.sequence_length = sequence_length
        self.seed = seed
        self.epoch = 0
        self.consumed = []
        self.current = None
        self.cursor = 0
        self.known = {}
        self.metadata = {}
        self.values = None
        self._discover()

    def _discover(self):
        summary = json.loads((self.directory / "manifest.json").read_text())
        if summary.get("status") == "failed":
            raise RuntimeError(f"Token producer failed: {summary.get('error')}")
        identity = (summary["source_revision"], summary["tokenizer_sha256"])
        found = set()
        for path in sorted(self.directory.glob("*/manifest.json")):
            if path.parent.name.endswith(".partial"):
                continue
            meta = json.loads(path.read_text())
            if (meta["source_revision"], meta["tokenizer_sha256"]) != identity:
                raise RuntimeError("Streaming corpus identity mismatch")
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            name = path.parent.name
            found.add(name)
            if name in self.known and self.known[name] != fingerprint:
                raise RuntimeError(f"Published chunk changed: {name}")
            self.known[name] = fingerprint
            self.metadata[name] = meta
        if set(self.known) - found:
            raise RuntimeError("Published streaming chunk was removed")
        self.complete = summary["complete"]
        self.available_tokens = sum(m["splits"]["train"]["tokens"] for m in self.metadata.values())
        self.num_blocks = sum(max(0, (m["splits"]["train"]["tokens"] - 1) // self.sequence_length)
                              for m in self.metadata.values())
        if self.complete and not self.num_blocks:
            raise RuntimeError("Completed corpus has no usable training blocks")

    def _open_current(self):
        meta = self.metadata[self.current]
        tokens = meta["splits"]["train"]["tokens"]
        path = self.directory / self.current / "train.bin"
        if path.stat().st_size != tokens * 2:
            raise RuntimeError("Streaming token file size mismatch")
        self.values = np.memmap(path, mode="r", dtype="<u2")
        self.count = max(0, (tokens - 1) // self.sequence_length)
        key = int(hashlib.blake2b(f"{self.seed}:{self.epoch}:{self.current}".encode(), digest_size=8).hexdigest(), 16)
        self.order = np.random.default_rng(key).permutation(self.count)

    def state_dict(self):
        return {"mode": "growing", "seed": self.seed, "sequence_length": self.sequence_length,
                "epoch": self.epoch, "consumed": list(self.consumed), "current": self.current,
                "cursor": self.cursor, "known": dict(self.known)}

    def load_state_dict(self, state):
        if state.get("mode") != "growing" or state["seed"] != self.seed or state["sequence_length"] != self.sequence_length:
            raise RuntimeError("Streaming resume configuration mismatch")
        self._discover()
        for name, fingerprint in state["known"].items():
            if self.known.get(name) != fingerprint:
                raise RuntimeError(f"Required streaming chunk missing or changed: {name}")
        self.epoch = state["epoch"]
        if self.epoch < 0 or not set(state["consumed"]).issubset(self.known):
            raise RuntimeError("Invalid streaming epoch or consumed chunks")
        self.consumed = list(state["consumed"])
        self.current, self.cursor = state["current"], state["cursor"]
        if self.current is not None:
            self._open_current()
            if not 0 <= self.cursor <= self.count:
                raise RuntimeError("Invalid streaming data cursor")

    def next_batch(self, size, stop_requested=None):
        rows = []
        waiting_logged = False
        while len(rows) < size:
            if self.current is not None and self.cursor == self.count:
                self.consumed.append(self.current)
                self.current = None
                self.values = None
            if self.current is None:
                self._discover()
                candidates = sorted(set(self.known) - set(self.consumed))
                if not candidates and self.complete and self.known:
                    self.epoch += 1
                    self.consumed = []
                    candidates = sorted(self.known)
                if not candidates:
                    if stop_requested and stop_requested():
                        raise DataWaitStopped("Stopped while waiting for prepared data")
                    if not waiting_logged:
                        print("Waiting for the next completed token chunk; no early-shard repetition.", flush=True)
                        waiting_logged = True
                    time.sleep(2)
                    continue
                self.current = candidates[0]
                self.cursor = 0
                self._open_current()
                if not self.count:
                    self.consumed.append(self.current)
                    self.current = None
                    continue
            start = int(self.order[self.cursor]) * self.sequence_length
            self.cursor += 1
            rows.append(np.asarray(self.values[start:start + self.sequence_length + 1], dtype=np.int64))
        return np.stack(rows)
