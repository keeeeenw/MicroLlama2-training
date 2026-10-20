"""Frozen XMemTransfer donor memory and a small target-model reader.

The addressing and reader equations follow OLAResearch/XMemTransfer, Apache-2.0,
revision ``4e530863f6c267674670147e8548654f8840c973``:
https://github.com/OLAResearch/XMemTransfer/tree/4e530863f6c267674670147e8548654f8840c973
Specifically ``engram/canonicalization.py``, ``hashing.py``, ``memory.py`` and
``adaptor.py``. This implementation adds explicit CPU preprocessing, control-token
resets, strict tensor-only loading, frozen tables, and residual calibration.

Compatibility is deliberately with the released bank's *raw tokenizer surface*
canonicalization. ByteLevel spellings such as ``Ã``/``©`` are concatenated before
normalization; decoding partial UTF-8 tokens independently would change donor
row identities. Only leading Ġ/▁ is a word boundary in the published algorithm.
It does not interpret embedded byte whitespace markers as linguistic boundaries.
Never substitute our existing Engram token-ID hashes for these CRC32 addresses.
Only already completed words participate, so later tokens cannot change an
earlier lookup. The current incomplete word is intentionally excluded.

This module does not alter a backbone, load remote code, or launch training.
The experiment runner owns the layer hook and checkpoint publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Mapping
import unicodedata
import zlib

import torch
from torch import nn


SOURCE_REVISION = "4e530863f6c267674670147e8548654f8840c973"
DONOR_REPOSITORY = "OLAResearchX/xmemtransfer-pythia160m-source-50m-word-boundary"


@dataclass(frozen=True)
class XMemConfig:
    """Released eight-head, 33,554,432-parameter donor-bank layout."""

    max_ngram: int = 3
    heads_per_order: int = 4
    table_size: int = 65_536
    d_head: int = 64
    hash_seed: int = 42
    canon_mode: str = "word_boundary"

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in
               (self.max_ngram, self.heads_per_order, self.table_size, self.d_head, self.hash_seed)):
            raise ValueError("XMemTransfer dimensions and hash seed must be integers")
        if self.canon_mode != "word_boundary":
            raise ValueError("XMemTransfer transfer requires the word_boundary donor bank")
        if self.max_ngram < 2 or self.heads_per_order < 1 or self.table_size < 2 or self.d_head < 1:
            raise ValueError("Invalid XMemTransfer memory dimensions")

    @property
    def total_heads(self) -> int:
        return (self.max_ngram - 1) * self.heads_per_order

    @property
    def d_mem(self) -> int:
        return self.total_heads * self.d_head

    @property
    def total_params(self) -> int:
        return self.total_heads * self.table_size * self.d_head

    @classmethod
    def from_json(cls, path: str | Path) -> "XMemConfig":
        document = json.loads(Path(path).read_text())
        required = ("max_ngram", "heads_per_order", "table_size", "d_head", "hash_seed", "canon_mode")
        if any(key not in document for key in required):
            raise ValueError("Donor memory configuration is incomplete")
        config = cls(**{key: document[key] for key in required})
        for key in ("d_mem", "total_params"):
            if key in document and document[key] != getattr(config, key):
                raise ValueError(f"Donor {key} disagrees with memory dimensions")
        # An arbitrary bank is unsuitable for this specific pretrained experiment.
        if config != cls():
            raise ValueError("Donor configuration differs from the pinned Pythia word-boundary release")
        return config


def normalize_word(surface: str) -> str:
    """Exact source normalization; notably lower(), rather than casefold()."""
    surface = unicodedata.normalize("NFKC", surface).lower()
    surface = unicodedata.normalize("NFD", surface)
    surface = "".join(char for char in surface if not unicodedata.combining(char))
    return " ".join(surface.split())


class WordBoundaryAddressing:
    """Precompute causal donor addresses from CPU Liquid ByteLevel token IDs.

    Each call starts independent block histories, as in source training. EOS,
    PAD, and the tokenizer's other explicit controls terminate history. Packed
    blocks must not carry words across documents. No GPU work or persistent
    mutable history is hidden here. For cached generation, callers can supply
    the full causal prefix and select its last address.
    """

    BOS_WORD = "<bos>"

    def __init__(
        self,
        token_surfaces: Mapping[int, str],
        config: XMemConfig | None = None,
        boundary_ids: set[int] | frozenset[int] = frozenset({0, 7}),
    ) -> None:
        self.config = config or XMemConfig()
        self.token_surfaces = dict(token_surfaces)
        self.boundary_ids = frozenset(boundary_ids)
        rng = random.Random(self.config.hash_seed)
        # WordNgramHasher draws salts directly, unlike the token-ID hasher.
        self.head_salts = tuple(rng.randrange(0, 2**31) for _ in range(self.config.total_heads))

    @classmethod
    def from_tokenizer_json(
        cls, path: str | Path, config: XMemConfig | None = None,
    ) -> "WordBoundaryAddressing":
        document = json.loads(Path(path).read_text())
        vocab = document.get("model", {}).get("vocab")
        if document.get("model", {}).get("type") != "BPE" or not isinstance(vocab, dict):
            raise ValueError("Expected the pinned Liquid BPE tokenizer JSON")
        decoder = document.get("decoder", {})
        decoders = decoder.get("decoders", []) if decoder.get("type") == "Sequence" else [decoder]
        if not any(part.get("type") == "ByteLevel" for part in decoders):
            raise ValueError("Raw donor addressing requires a ByteLevel tokenizer")
        surfaces = {int(index): str(surface) for surface, index in vocab.items()}
        controls = {0, 7}
        for token in document.get("added_tokens", []):
            surfaces[int(token["id"])] = token["content"]
            if token.get("special", False):
                controls.add(int(token["id"]))
        return cls(surfaces, config, controls)

    def _completed_addresses(self, completed: list[str]) -> list[int]:
        addresses = []
        head = 0
        for order in range(2, self.config.max_ngram + 1):
            words = ([self.BOS_WORD] * max(0, order - len(completed)) + completed)[-order:]
            base_hash = zlib.crc32("|".join(words).encode("utf-8")) & 0xFFFFFFFF
            for _ in range(self.config.heads_per_order):
                addresses.append((base_hash ^ self.head_salts[head]) % self.config.table_size)
                head += 1
        return addresses

    def encode_with_mask(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return CPU long [B,T,8] addresses and boolean [B,T] active mask."""
        if input_ids.device.type != "cpu":
            raise ValueError("Precompute XMemTransfer addresses on CPU before moving the block to GPU")
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint16):
            raise ValueError("XMemTransfer addressing expects an integer [batch, sequence] tensor")
        batch, length = input_ids.shape
        if not batch or not length:
            return (torch.empty((batch, length, self.config.total_heads), dtype=torch.long),
                    torch.empty((batch, length), dtype=torch.bool))
        flat_addresses: list[int] = []
        active: list[bool] = []
        for sequence in input_ids.tolist():
            completed = [self.BOS_WORD]
            parts: list[str] = []
            cached = self._completed_addresses(completed)
            for token_id in sequence:
                if token_id in self.boundary_ids:
                    completed = [self.BOS_WORD]
                    parts = []
                    cached = self._completed_addresses(completed)
                    active.append(False)
                else:
                    if token_id not in self.token_surfaces:
                        raise ValueError(f"Unknown tokenizer ID {token_id}")
                    surface = self.token_surfaces[token_id]
                    is_boundary = surface.startswith(("Ġ", "▁"))
                    if is_boundary and parts:
                        word = normalize_word("".join(parts))
                        if word:
                            completed.append(word)
                            completed = completed[-self.config.max_ngram:]
                        parts = []
                        cached = self._completed_addresses(completed)
                    if is_boundary:
                        surface = surface[1:]
                    if surface:
                        parts.append(surface)
                    active.append(True)
                flat_addresses.extend(cached)
        return (torch.tensor(flat_addresses, dtype=torch.long).reshape(batch, length, self.config.total_heads),
                torch.tensor(active, dtype=torch.bool).reshape(batch, length))

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.encode_with_mask(input_ids)[0]


class FrozenMemory(nn.Module):
    """Strict frozen per-head embeddings; safe CPU tensor-only donor loading."""

    def __init__(self, config: XMemConfig, weights: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.config = config
        expected = {f"tables.{head}.weight" for head in range(config.total_heads)}
        if set(weights) != expected:
            raise ValueError("Donor state dict must contain exactly the expected embedding tables")
        tables = []
        for head in range(config.total_heads):
            weight = weights[f"tables.{head}.weight"]
            if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
                    or weight.shape != (config.table_size, config.d_head)
                    or not weight.is_floating_point() or weight.layout != torch.strided):
                raise ValueError(f"Invalid donor embedding tensor for head {head}")
            if not torch.isfinite(weight).all():
                raise ValueError(f"Nonfinite donor embedding tensor for head {head}")
            tables.append(nn.Embedding.from_pretrained(weight.detach(), freeze=True))
        self.tables = nn.ModuleList(tables)
        self.train(False)

    @classmethod
    def from_directory(
        cls, path: str | Path, config: XMemConfig | None = None,
        memory_file: str = "memory_best.pt",
    ) -> "FrozenMemory":
        directory = Path(path)
        if Path(memory_file).name != memory_file:
            raise ValueError("Donor filename must name one file in the donor directory")
        saved_config = XMemConfig.from_json(directory / "memory_config.json")
        if config is not None and config != saved_config:
            raise ValueError("Requested memory configuration differs from the published bank")
        config = saved_config
        weights = torch.load(directory / memory_file, map_location="cpu", weights_only=True)
        if not isinstance(weights, Mapping):
            raise ValueError("Donor memory file is not a tensor state dict")
        return cls(config, weights)

    @property
    def d_mem(self) -> int:
        return self.config.d_mem

    def train(self, mode: bool = True) -> "FrozenMemory":
        # A parent model's .train() must never make the donor trainable.
        super().train(False)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if indices.ndim != 3 or indices.shape[-1] != self.config.total_heads or indices.dtype != torch.long:
            raise ValueError("Expected long donor addresses [batch, sequence, total_heads]")
        return torch.cat([table(indices[..., head]) for head, table in enumerate(self.tables)], dim=-1)


class TargetReader(nn.Module):
    """One-site trainable key/value reader with a context-dependent sigmoid gate.

    Return a residual contribution and gates; the caller adds the contribution
    at its explicitly selected decoder layer. The frozen 512-wide vectors are
    projected to the 1024-wide student without changing its existing parameters.
    A small, nonzero fixed scale preserves the parent function while allowing
    first-step gradients to all reader projections and normalization weights.
    ``calibrate_residual_scale`` can match initial residual RMS across conditions.
    """

    def __init__(
        self, hidden_size: int = 1024, memory_dim: int = 512,
        residual_scale: float = 1e-3, gate_bias_init: float = -2.0,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or memory_dim < 1 or not 0 < residual_scale <= 1:
            raise ValueError("Reader dimensions and nonzero residual scale must be valid")
        self.hidden_size = hidden_size
        self.w_k = nn.Linear(memory_dim, hidden_size, bias=False)
        self.w_v = nn.Linear(memory_dim, hidden_size, bias=False)
        self.norm_h = nn.RMSNorm(hidden_size, eps=1e-6)
        self.norm_k = nn.RMSNorm(hidden_size, eps=1e-6)
        self.gate_bias = nn.Parameter(torch.tensor(float(gate_bias_init)))
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        nn.init.normal_(self.w_k.weight, mean=0, std=0.02)
        nn.init.normal_(self.w_v.weight, mean=0, std=0.02)

    def _unscaled(
        self, hidden: torch.Tensor, memory: torch.Tensor, active: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[:-1] != memory.shape[:-1] or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Hidden states and memory vectors must share batch/sequence dimensions")
        key = self.w_k(memory)
        value = self.w_v(memory)
        logit = (self.norm_h(hidden).float() * self.norm_k(key).float()).sum(-1)
        gate = torch.sigmoid(logit / math.sqrt(self.hidden_size) + self.gate_bias)
        if active is not None:
            if active.shape != hidden.shape[:-1] or active.dtype != torch.bool:
                raise ValueError("Memory active mask must be boolean [batch, sequence]")
            gate = gate * active
        return (gate.unsqueeze(-1).to(value.dtype) * value).to(hidden.dtype), gate

    def forward(
        self, hidden: torch.Tensor, memory: torch.Tensor, active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contribution, gate = self._unscaled(hidden, memory, active)
        return contribution * self.residual_scale, gate

    @torch.no_grad()
    def calibrate_residual_scale(
        self, hidden: torch.Tensor, memory: torch.Tensor,
        active: torch.Tensor | None = None, target_ratio: float = 1e-3,
    ) -> float:
        """Match actual residual RMS, allowing for the bank's small vector units.

        The scalar need not be below one: released vectors have RMS around 0.02,
        while trained hidden states may be much larger. The measured RMS ratio
        is the stability constraint. Reject extreme amplification instead of
        silently accepting a nearly zero or incompatible memory contribution.
        """
        if not 0 < target_ratio <= 0.1:
            raise ValueError("Calibration target must be positive and preserve the parent function")
        contribution, _ = self._unscaled(hidden, memory, active)
        if active is not None:
            hidden, contribution = hidden[active], contribution[active]
        if hidden.numel() == 0:
            raise ValueError("Cannot calibrate on a block containing only controls")
        hidden_rms = hidden.float().square().mean().sqrt()
        added_rms = contribution.float().square().mean().sqrt()
        if not torch.isfinite(hidden_rms + added_rms) or min(hidden_rms, added_rms) <= 0:
            raise ValueError("Calibration requires finite nonzero hidden and contribution RMS")
        scale = float((target_ratio * hidden_rms / added_rms).item())
        if not math.isfinite(scale) or not 0 < scale <= 100:
            raise ValueError("Calibration target requires a residual scale outside (0, 100]")
        self.residual_scale.fill_(scale)
        return scale
