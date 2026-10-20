"""Small, explicit Engram-style conditional-memory prototype for Qwen3.

Provenance: https://arxiv.org/abs/2601.07372 and the official illustrative
implementation at https://github.com/deepseek-ai/Engram. This prototype keeps
the paper's hashed suffix n-gram lookup, context-aware key/value gating, and
sparse layer placement. The legacy profile uses one table per order; the
confirmation profile adds normalized lookup identities and independently
hashed prime-sized heads at approximately the same parameter budget. Both
share gate/value projections across selected layers and omit causal short
convolution, multi-branch/HConv integration, sharding and host/NVMe offload. The official demo
itself describes its backbone components as mocks; this module instead wraps
the pinned Transformers Qwen3 layers directly.

The legacy default 1024-wide setting adds 50,190,337 trainable parameters. Its
residual starts at a small nonzero scale, so the dense function is nearly
preserved while table, key, and value parameters all receive gradients on the
first update. EOS and PAD terminate the local n-gram history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional
import unicodedata

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    create_causal_mask,
    create_sliding_window_causal_mask,
)


LOOKUP_V2 = "normalized-multihead-v2"


def token_map_digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def canonical_token_map(decoded: list[str], protected: set[int], model_vocab: int) -> list[int]:
    """Normalize only memory identities; isolate controls, partial UTF-8 and padding IDs."""
    if model_vocab < len(decoded):
        raise ValueError("Model vocabulary is smaller than the tokenizer")
    mapping = list(range(model_vocab))
    canonical = {}
    for index, text in enumerate(decoded):
        if index in protected or "\ufffd" in text or not text.strip():
            continue
        key = unicodedata.normalize("NFKC", text).casefold().strip()
        if key:
            mapping[index] = canonical.setdefault(key, index)
    return mapping


def build_token_id_map(tokenizer_json: str | Path, model_vocab: int) -> list[int]:
    """Read the pinned tokenizer without changing it, producing a checkpointed lookup map."""
    from tokenizers import Tokenizer
    path = Path(tokenizer_json)
    document = json.loads(path.read_text())
    tokenizer = Tokenizer.from_file(str(path))
    protected = {token["id"] for token in document.get("added_tokens", [])} | {0, 1, 7}
    decoded = [tokenizer.decode([i], skip_special_tokens=False)
               for i in range(tokenizer.get_vocab_size(with_added_tokens=True))]
    return canonical_token_map(decoded, protected, model_vocab)


def prime_table_sizes(start: int, count: int) -> list[int]:
    """Distinct deterministic prime moduli without a runtime dependency on SymPy."""
    result = []
    candidate = max(2, start)
    while len(result) < count:
        if all(candidate % divisor for divisor in range(2, math.isqrt(candidate) + 1)):
            result.append(candidate)
        candidate += 1
    return result


@dataclass
class EngramState:
    """Explicit suffix history for chunked input and cached decoding.

    ``history_ids`` is right aligned and has shape ``[batch, 2]``. A negative
    entry means that no in-document token is available at that position.
    Callers own this state; the model keeps no hidden mutable generation cache.
    """

    history_ids: torch.LongTensor

    def to(self, device: torch.device | str) -> "EngramState":
        return EngramState(self.history_ids.to(device=device))

    def detach(self) -> "EngramState":
        return EngramState(self.history_ids.detach())


@dataclass
class EngramBaseModelOutputWithPast(BaseModelOutputWithPast):
    engram_state: Optional[EngramState] = None


@dataclass
class EngramCausalLMOutputWithPast(CausalLMOutputWithPast):
    engram_state: Optional[EngramState] = None


class EngramQwen3Config(Qwen3Config):
    """Qwen3 config that makes Engram checkpoints distinguishable from dense."""

    model_type = "engram_qwen3"

    def __init__(
        self,
        *args: Any,
        engram_table_size: int = 194_000,
        engram_embedding_dim: int = 128,
        engram_layers: tuple[int, ...] | list[int] = (3, 11),
        engram_residual_scale: float = 1e-3,
        engram_lookup_version: str = "single-head-v1",
        engram_hash_heads: int = 1,
        engram_token_id_map: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("model_type", None)
        kwargs.pop("architectures", None)
        super().__init__(*args, **kwargs)
        if engram_table_size < 2 or engram_embedding_dim < 1:
            raise ValueError("Engram table size and embedding dimension must be positive")
        layers = tuple(int(layer) for layer in engram_layers)
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("Engram layers must be a nonempty list of unique layer indices")
        if any(layer < 0 or layer >= self.num_hidden_layers for layer in layers):
            raise ValueError("Engram layer index is outside the Qwen3 backbone")
        if not 0.0 < engram_residual_scale <= 1.0:
            raise ValueError("Engram residual scale must be in (0, 1]")
        self.engram_table_size = int(engram_table_size)
        self.engram_embedding_dim = int(engram_embedding_dim)
        self.engram_layers = layers
        self.engram_residual_scale = float(engram_residual_scale)
        if engram_lookup_version not in ("single-head-v1", LOOKUP_V2):
            raise ValueError("Unsupported memory lookup version")
        if engram_hash_heads not in (1, 2, 4, 8) or engram_embedding_dim % engram_hash_heads:
            raise ValueError("Memory heads must divide the per-order embedding dimension")
        if engram_lookup_version == "single-head-v1":
            if engram_hash_heads != 1 or engram_token_id_map is not None:
                raise ValueError("Legacy lookup requires one head and unmodified IDs")
        elif (engram_token_id_map is None or len(engram_token_id_map) != self.vocab_size or
              any(type(i) is not int or not 0 <= i < self.vocab_size for i in engram_token_id_map)):
            raise ValueError("Revised memory requires a complete valid token identity map")
        if engram_token_id_map is not None:
            for special in (self.eos_token_id, self.pad_token_id, self.bos_token_id):
                if special is not None and engram_token_id_map[special] != special:
                    raise ValueError("Memory normalization cannot merge document controls")
        self.engram_lookup_version = engram_lookup_version
        self.engram_hash_heads = engram_hash_heads
        self.engram_token_id_map = engram_token_id_map


class EngramMemory(nn.Module):
    """Hashed bigram/trigram lookup with context-aware gated residual fusion."""

    _BIGRAM_MULTIPLIERS = (6364136223846793005, 1442695040888963407)
    _TRIGRAM_MULTIPLIERS = (2862933555777941757, 3202034522624059733, 3935559000370003845)

    def __init__(self, config: EngramQwen3Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.eos_token_id = config.eos_token_id
        self.pad_token_id = config.pad_token_id
        table_size = config.engram_table_size
        dim = config.engram_embedding_dim
        self.revised = config.engram_lookup_version == LOOKUP_V2
        self.heads = config.engram_hash_heads
        if self.revised:
            sizes = prime_table_sizes(table_size, 2 * self.heads)
            self.bigram_tables = nn.ModuleList(nn.Embedding(size, dim // self.heads) for size in sizes[:self.heads])
            self.trigram_tables = nn.ModuleList(nn.Embedding(size, dim // self.heads) for size in sizes[self.heads:])
            self.register_buffer("canonical_ids", torch.tensor(config.engram_token_id_map, dtype=torch.long))
        else:
            self.bigram_table = nn.Embedding(table_size, dim)
            self.trigram_table = nn.Embedding(table_size, dim)
        joined_dim = 2 * dim
        self.key_proj = nn.Linear(joined_dim, self.hidden_size, bias=False)
        self.value_proj = nn.Linear(joined_dim, self.hidden_size, bias=False)
        self.query_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.key_norm = nn.RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.residual_scale = nn.Parameter(torch.tensor(config.engram_residual_scale, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Small, nonzero values retain the dense backbone's behavior closely
        # without cutting off first-step gradients to the tables or gate.
        for table in self.tables():
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.25)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.25)
        nn.init.ones_(self.query_norm.weight)
        nn.init.ones_(self.key_norm.weight)

    def tables(self):
        if self.revised:
            return list(self.bigram_tables) + list(self.trigram_tables)
        return [self.bigram_table, self.trigram_table]

    @staticmethod
    def empty_state(batch_size: int, device: torch.device) -> EngramState:
        return EngramState(torch.full((batch_size, 2), -1, dtype=torch.long, device=device))

    def _is_boundary(self, ids: torch.LongTensor) -> torch.BoolTensor:
        boundary = torch.zeros_like(ids, dtype=torch.bool)
        if self.pad_token_id is not None:
            boundary = boundary | ids.eq(self.pad_token_id)
        if self.eos_token_id is not None:
            boundary = boundary | ids.eq(self.eos_token_id)
        return boundary

    def _indices(
        self, input_ids: torch.LongTensor, state: EngramState
    ) -> tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor, torch.BoolTensor]:
        batch, length = input_ids.shape
        if state.history_ids.shape != (batch, 2):
            raise ValueError(f"Engram state must have shape {(batch, 2)}, got {tuple(state.history_ids.shape)}")
        if state.history_ids.device != input_ids.device:
            raise ValueError("Engram state and input_ids must be on the same device")
        joined = torch.cat((state.history_ids, input_ids), dim=1)
        prev1 = joined[:, 1 : length + 1]
        prev2 = joined[:, :length]
        current = input_ids
        boundary = self._is_boundary(current)
        current_valid = ~boundary
        prev1_valid = prev1.ge(0) & ~self._is_boundary(prev1)
        prev2_valid = prev2.ge(0) & ~self._is_boundary(prev2)
        bigram_valid = current_valid & prev1_valid
        trigram_valid = bigram_valid & prev2_valid

        if self.revised:
            current = self.canonical_ids[current]
            prev1 = self.canonical_ids[prev1.clamp_min(0)]
            prev2 = self.canonical_ids[prev2.clamp_min(0)]
            bigram_indices, trigram_indices = [], []
            for head in range(self.heads):
                salt = (head + 1) * 0x1E3779B97F4A7C15
                def coefficients(base):
                    return [((value ^ (salt * (i + 1))) & (2**63 - 1)) | 1
                            for i, value in enumerate(base)]
                b0, b1 = coefficients(self._BIGRAM_MULTIPLIERS)
                t0, t1, t2 = coefficients(self._TRIGRAM_MULTIPLIERS)
                bigram_indices.append(torch.remainder((prev1 * b0) ^ (current * b1),
                                       self.bigram_tables[head].num_embeddings))
                trigram_indices.append(torch.remainder((prev2 * t0) ^ (prev1 * t1) ^ (current * t2),
                                        self.trigram_tables[head].num_embeddings))
            return (torch.stack(bigram_indices, dim=-1), torch.stack(trigram_indices, dim=-1),
                    bigram_valid, trigram_valid)

        # Integer overflow is intentional: two's-complement int64 arithmetic
        # followed by remainder gives deterministic indices on CPU and CUDA.
        b0, b1 = self._BIGRAM_MULTIPLIERS
        bigram_hash = (prev1 * b0) ^ (current * b1)
        t0, t1, t2 = self._TRIGRAM_MULTIPLIERS
        trigram_hash = (prev2 * t0) ^ (prev1 * t1) ^ (current * t2)
        bigram_idx = torch.remainder(bigram_hash, self.bigram_table.num_embeddings)
        trigram_idx = torch.remainder(trigram_hash, self.trigram_table.num_embeddings)
        return bigram_idx, trigram_idx, bigram_valid, trigram_valid

    def _next_state(self, input_ids: torch.LongTensor, state: EngramState) -> EngramState:
        batch, _ = input_ids.shape
        boundary = self._is_boundary(input_ids)
        prior = state.history_ids
        all_ids = torch.cat((prior, input_ids), dim=1)
        all_valid = torch.cat((prior.ge(0), ~boundary), dim=1)
        positions = torch.arange(all_ids.shape[1], device=input_ids.device).expand(batch, -1)
        boundary_positions = torch.where(~all_valid, positions, -1)
        last_boundary = boundary_positions.max(dim=1, keepdim=True).values
        after_boundary = all_valid & positions.gt(last_boundary)
        last = torch.where(after_boundary, positions, -1).max(dim=1).values
        second = torch.where(after_boundary & positions.lt(last[:, None]), positions, -1).max(dim=1).values
        safe_last = last.clamp_min(0)
        safe_second = second.clamp_min(0)
        gathered = torch.stack((all_ids.gather(1, safe_second[:, None]).squeeze(1),
                                all_ids.gather(1, safe_last[:, None]).squeeze(1)), dim=1)
        gathered = torch.where(torch.stack((second.ge(0), last.ge(0)), dim=1), gathered, -1)
        return EngramState(gathered)

    def forward(self, hidden: torch.Tensor, input_ids: torch.LongTensor, state: EngramState) -> tuple[torch.Tensor, EngramState]:
        bigram_idx, trigram_idx, bigram_valid, trigram_valid = self._indices(input_ids, state)
        if self.revised:
            bigram = torch.cat([table(bigram_idx[..., head])
                                for head, table in enumerate(self.bigram_tables)], dim=-1)
            trigram = torch.cat([table(trigram_idx[..., head])
                                 for head, table in enumerate(self.trigram_tables)], dim=-1)
        else:
            bigram = self.bigram_table(bigram_idx)
            trigram = self.trigram_table(trigram_idx)
        bigram = bigram * bigram_valid.unsqueeze(-1).to(hidden.dtype)
        trigram = trigram * trigram_valid.unsqueeze(-1).to(hidden.dtype)
        memory = torch.cat((bigram, trigram), dim=-1)
        key = self.key_norm(self.key_proj(memory))
        query = self.query_norm(hidden)
        gate = torch.sigmoid((query * key).sum(dim=-1, keepdim=True) / (self.hidden_size ** 0.5))
        value = self.value_proj(memory)
        current_valid = ~self._is_boundary(input_ids)
        output = hidden + (gate * value * current_valid.unsqueeze(-1).to(value.dtype) * self.residual_scale)
        return output, self._next_state(input_ids, state)


class EngramQwen3Model(Qwen3Model):
    """Qwen3 backbone with explicit memory injection after selected layers."""

    def __init__(self, config: EngramQwen3Config) -> None:
        super().__init__(config)
        self.engram = EngramMemory(config)
        self.engram_layers = tuple(config.engram_layers)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        engram_state: Optional[EngramState] = None,
        **kwargs: Any,
    ) -> EngramBaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if input_ids is None:
            raise ValueError("Engram requires input_ids so suffix n-grams can be retrieved")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch, length = input_ids.shape
        state = engram_state or self.engram.empty_state(batch, input_ids.device)

        if use_cache and past_key_values is None:
            from transformers.cache_utils import DynamicCache
            past_key_values = DynamicCache(config=self.config)
        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen, past_seen + length, device=input_ids.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_masks = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_masks = attention_mask

        hidden = inputs_embeds
        position_embeddings = self.rotary_emb(hidden, position_ids)
        for index, layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden = layer(
                hidden,
                attention_mask=causal_masks[layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if index in self.engram_layers:
                # Every insertion sees the same suffix of preceding input. The
                # caller-visible state advances once per model call, not once
                # per memory-bearing layer.
                hidden, _ = self.engram(hidden, input_ids, state)
        hidden = self.norm(hidden)
        next_state = self.engram._next_state(input_ids, state)
        return EngramBaseModelOutputWithPast(
            last_hidden_state=hidden,
            past_key_values=past_key_values if use_cache else None,
            engram_state=next_state,
        )


class EngramQwen3ForCausalLM(Qwen3ForCausalLM):
    """Qwen3 causal LM whose saved config requires the Engram-aware class."""

    config_class = EngramQwen3Config

    def __init__(self, config: EngramQwen3Config) -> None:
        # Reuse Qwen3's pretrained-model initialization without constructing a
        # throwaway dense backbone; this allocates only one 338M-class model.
        super(Qwen3ForCausalLM, self).__init__(config)
        self.model = EngramQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if config.tie_word_embeddings:
            self.tie_weights()
        self.config.architectures = [self.__class__.__name__]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        engram_state: Optional[EngramState] = None,
        **kwargs: Any,
    ) -> EngramCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            engram_state=engram_state,
            **kwargs,
        )
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(outputs.last_hidden_state[:, selection, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        return EngramCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            engram_state=outputs.engram_state,
        )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "HF generate() does not carry EngramState between decoding calls. "
            "Use engram_prefill() and engram_decode_step() with the returned "
            "past_key_values and engram_state."
        )

    def save_pretrained(self, *args: Any, **kwargs: Any) -> Any:
        result = super().save_pretrained(*args, **kwargs)
        if args:
            directory = Path(args[0])
        else:
            directory = Path(kwargs["save_directory"])
        (directory / "engram_config.json").write_text(
            __import__("json").dumps(self.engram_metadata(), indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
        return_info = kwargs.pop("output_loading_info", False)
        kwargs["output_loading_info"] = True
        model, info = super().from_pretrained(*args, **kwargs)
        if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
            raise RuntimeError(
                "Engram checkpoint did not restore strictly: "
                f"missing={info.get('missing_keys')}, unexpected={info.get('unexpected_keys')}, "
                f"mismatched={info.get('mismatched_keys')}"
            )
        return (model, info) if return_info else model

    @classmethod
    def from_dense(
        cls,
        dense_model: Qwen3ForCausalLM,
        *,
        table_size: int = 194_000,
        embedding_dim: int = 128,
        layers: tuple[int, ...] = (3, 11),
        residual_scale: float = 1e-3,
        lookup_version: str = "single-head-v1",
        hash_heads: int = 1,
        token_id_map: Optional[list[int]] = None,
    ) -> "EngramQwen3ForCausalLM":
        """Copy a dense Qwen3 checkpoint and attach a near-neutral memory path."""
        config_values = dense_model.config.to_dict()
        for key in ("model_type", "architectures", "transformers_version", "_name_or_path"):
            config_values.pop(key, None)
        config = EngramQwen3Config(
            **config_values,
            engram_table_size=table_size,
            engram_embedding_dim=embedding_dim,
            engram_layers=layers,
            engram_residual_scale=residual_scale,
            engram_lookup_version=lookup_version,
            engram_hash_heads=hash_heads,
            engram_token_id_map=token_id_map,
        )
        model = cls(config)
        result = model.load_state_dict(dense_model.state_dict(), strict=False)
        expected = {key for key in model.state_dict() if key.startswith("model.engram.")}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError(f"Dense-to-Engram restore mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
        if config.tie_word_embeddings and model.lm_head.weight is not model.model.embed_tokens.weight:
            raise RuntimeError("Qwen3 tied embedding was not preserved while attaching Engram")
        return model

    def engram_parameter_count(self) -> int:
        return sum(parameter.numel() for name, parameter in self.named_parameters() if ".engram." in name)

    def engram_metadata(self) -> dict[str, Any]:
        metadata = {
            "architecture": self.__class__.__name__,
            "provenance": {
                "paper": "https://arxiv.org/abs/2601.07372",
                "official_demo": "https://github.com/deepseek-ai/Engram",
            },
            "implementation": "single hash head per order; shared key/value gate across layers; no compression, convolution, sharding, or offload",
            "layers_zero_based": list(self.config.engram_layers),
            "table_size_per_order": self.config.engram_table_size,
            "embedding_dim_per_order": self.config.engram_embedding_dim,
            "added_trainable_parameters": self.engram_parameter_count(),
            "initial_residual_scale": self.config.engram_residual_scale,
            "learned_residual_scale": float(self.model.engram.residual_scale.detach().float().cpu()),
            "eos_token_id": self.config.eos_token_id,
            "pad_token_id": self.config.pad_token_id,
        }
        if self.config.engram_lookup_version == LOOKUP_V2:
            metadata.update(
                implementation="normalized multi-head lookup; shared key/value gate across layers; no convolution, multi-branch, sharding, or offload",
                lookup_version=LOOKUP_V2, hash_heads_per_order=self.config.engram_hash_heads,
                table_sizes=[table.num_embeddings for table in self.model.engram.tables()],
                normalization_policy="nfkc-casefold-strip-v1; controls/partial UTF-8/whitespace isolated",
                token_map_sha256=token_map_digest(self.config.engram_token_id_map),
                distinct_lookup_ids=len(set(self.config.engram_token_id_map)),
            )
        return metadata


def build_engram_model(
    model_config: dict[str, Any],
    *,
    dense_model: Optional[Qwen3ForCausalLM] = None,
    table_size: int = 194_000,
    embedding_dim: int = 128,
    layers: tuple[int, ...] = (3, 11),
    residual_scale: float = 1e-3,
    lookup_version: str = "single-head-v1",
    hash_heads: int = 1,
    token_id_map: Optional[list[int]] = None,
) -> EngramQwen3ForCausalLM:
    """Build from a model JSON config, optionally copying an in-memory dense model."""
    values = dict(model_config)
    for key in ("model_type", "architectures"):
        values.pop(key, None)
    config = EngramQwen3Config(
        **values,
        engram_table_size=table_size,
        engram_embedding_dim=embedding_dim,
        engram_layers=layers,
        engram_residual_scale=residual_scale,
        engram_lookup_version=lookup_version,
        engram_hash_heads=hash_heads,
        engram_token_id_map=token_id_map,
    )
    model = EngramQwen3ForCausalLM(config)
    if dense_model is not None:
        result = model.load_state_dict(dense_model.state_dict(), strict=False)
        expected = {key for key in model.state_dict() if key.startswith("model.engram.")}
        if set(result.missing_keys) != expected or result.unexpected_keys:
            raise RuntimeError(f"Dense-to-Engram restore mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    return model


def save_engram_model(model: EngramQwen3ForCausalLM, directory: str | Path) -> Path:
    """Save a complete Engram checkpoint with an architecture marker.

    Its ``model_type`` is ``engram_qwen3`` and architecture is
    ``EngramQwen3ForCausalLM``. A stock Qwen3 loader therefore rejects the
    unknown config instead of silently ignoring memory weights. Import this
    module before using Transformers Auto classes in a fresh process.
    """
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.config.architectures = [EngramQwen3ForCausalLM.__name__]
    model.save_pretrained(destination, safe_serialization=True)
    (destination / "engram_metadata.json").write_text(
        __import__("json").dumps(model.engram_metadata(), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_engram_model(directory: str | Path, *, map_location: str | torch.device = "cpu") -> EngramQwen3ForCausalLM:
    """Restore the explicit Engram class; plain Qwen3 loading is not supported."""
    path = Path(directory)
    if not (path / "engram_config.json").is_file():
        raise RuntimeError(f"Missing Engram marker in {path}; refusing to load as a dense Qwen3 checkpoint")
    config = EngramQwen3Config.from_pretrained(path, local_files_only=True)
    model = EngramQwen3ForCausalLM.from_pretrained(path, config=config, local_files_only=True, torch_dtype=None).to(map_location)
    metadata = __import__("json").loads((path / "engram_config.json").read_text(encoding="utf-8"))
    if model.config.engram_lookup_version == LOOKUP_V2:
        if model.model.engram.canonical_ids.cpu().tolist() != model.config.engram_token_id_map:
            raise RuntimeError("Checkpoint lookup buffer does not match its configured token map")
    if metadata != model.engram_metadata():
        raise RuntimeError("Engram metadata does not match the reconstructed model")
    return model


def engram_prefill(
    model: EngramQwen3ForCausalLM,
    input_ids: torch.LongTensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
) -> EngramCausalLMOutputWithPast:
    """Run a prompt and return explicit Engram suffix state plus KV cache."""
    return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)


def engram_decode_step(
    model: EngramQwen3ForCausalLM,
    input_ids: torch.LongTensor,
    *,
    past_key_values: Any,
    engram_state: EngramState,
    attention_mask: Optional[torch.Tensor] = None,
) -> EngramCausalLMOutputWithPast:
    """Decode a token fragment using caller-owned KV and Engram state."""
    return model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        engram_state=engram_state,
        use_cache=True,
    )


# Register only after defining the custom config/class. Without this module
# import, an Auto loader sees the unknown `engram_qwen3` model type and fails
# instead of constructing a dense Qwen3 model that drops memory weights.
try:
    AutoConfig.register(EngramQwen3Config.model_type, EngramQwen3Config)
except ValueError:
    pass
try:
    AutoModelForCausalLM.register(EngramQwen3Config, EngramQwen3ForCausalLM)
except ValueError:
    pass
