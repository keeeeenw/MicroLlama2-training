"""Compact Qwen3.5/3.8 text blocks, native Engram, and a shared MTP predictor.

Requires the isolated Transformers 5.3 environment. Decoder blocks are upstream
implementations; hooks insert our existing memory after specified layers without
reimplementing DeltaNet. MTP follows vLLM's Qwen3.5 predictor structure, with a
declared local training recipe (Qwen's auxiliary loss recipe is unpublished).
"""

from copy import deepcopy
import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3Config
from transformers import initialization as init
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3Model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForCausalLM, Qwen3_5TextModel, Qwen3_5DecoderLayer, Qwen3_5RMSNorm, Qwen3_5GatedDeltaNet,
    create_causal_mask,
)

from .engram import EngramMemory, EngramQwen3Config, LOOKUP_V2
from .objective import hidden_nll, target_mask


def memory_config(config):
    """Reuse the qualified memory validation and identity, independent of decoder."""
    return EngramQwen3Config(
        vocab_size=config.vocab_size, hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers, rms_norm_eps=config.rms_norm_eps,
        eos_token_id=config.eos_token_id, pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id, engram_table_size=config.engram_table_size,
        engram_embedding_dim=config.engram_embedding_dim,
        engram_layers=config.engram_layers, engram_residual_scale=config.engram_residual_scale,
        engram_lookup_version=LOOKUP_V2, engram_hash_heads=4,
        engram_token_id_map=config.engram_token_id_map,
    )


class MemoryConfigMixin:
    def __init__(self, *args, engram_table_size=194_000, engram_embedding_dim=128,
                 engram_layers=(3, 11), engram_residual_scale=1.0,
                 engram_token_id_map=None, engram_init_seed=1042, **kwargs):
        kwargs.pop("model_type", None)
        super().__init__(*args, **kwargs)
        self.engram_table_size = engram_table_size
        self.engram_embedding_dim = engram_embedding_dim
        self.engram_layers = list(engram_layers)
        self.engram_residual_scale = engram_residual_scale
        self.engram_token_id_map = (list(range(self.vocab_size)) if engram_token_id_map is None
                                    else engram_token_id_map)
        self.engram_lookup_version = LOOKUP_V2
        self.engram_hash_heads = 4
        self.engram_init_seed = int(engram_init_seed)
        memory_config(self)  # Reject invalid placement, controls and lookup identity.


def reset_memory(model):
    """Identical initial memory weights in both arms despite different decoders."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(model.config.engram_init_seed)
        model.model.engram.reset_parameters()


class HybridConfig(MemoryConfigMixin, Qwen3_5TextConfig):
    model_type = "microllama_hybrid_engram"

    def __init__(self, *args, mtp_steps=2, mtp_loss_weight=0.1,
                 delta_init="fla-loguniform-v1", mtp_activation_checkpointing=True, **kwargs):
        super().__init__(*args, **kwargs)
        if type(mtp_steps) is not int or not 0 <= mtp_steps <= 3:
            raise ValueError("MTP supports zero to three auxiliary prediction steps")
        if not 0 < mtp_loss_weight <= 1:
            raise ValueError("MTP loss weight must be in (0, 1]")
        self.mtp_steps = mtp_steps
        self.mtp_loss_weight = mtp_loss_weight
        self.mtp_num_hidden_layers = 1 if mtp_steps else 0
        if delta_init != "fla-loguniform-v1":
            raise ValueError("Unknown from-scratch DeltaNet initialization")
        self.delta_init = delta_init
        self.mtp_activation_checkpointing = bool(mtp_activation_checkpointing)


class DenseControlConfig(MemoryConfigMixin, Qwen3Config):
    model_type = "microllama_dense_engram"


class MemoryDecoderMixin:
    """Synchronous memory hooks with call-local IDs; no decoder recomputation.

    All insertion layers receive the same preceding suffix. The recurrent
    suffix advances once per model call, never once per insertion. Cached
    decoding is rejected until the combined DeltaNet/Engram cache is qualified.
    """
    def __init__(self, config):
        super().__init__(config)
        self.engram = EngramMemory(memory_config(config))
        self._memory_context = None
        for index in config.engram_layers:
            self.layers[index].register_forward_hook(self._insert_memory)

    def _insert_memory(self, module, inputs, output):
        ids, state = self._memory_context
        if isinstance(output, tuple):
            hidden, _ = self.engram(output[0], ids, state)
            return (hidden, *output[1:])
        hidden, _ = self.engram(output, ids, state)
        return hidden

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None or kwargs.get("inputs_embeds") is not None:
            raise ValueError("Engram requires explicit input IDs")
        if kwargs.get("use_cache", self.config.use_cache) or kwargs.get("past_key_values") is not None:
            raise ValueError("Combined cached decoding is unqualified; set use_cache=False")
        if self.gradient_checkpointing or self._memory_context is not None:
            raise ValueError("Memory hooks require non-reentrant, non-checkpointed decoder execution")
        self._memory_context = (input_ids, self.engram.empty_state(input_ids.shape[0], input_ids.device))
        try:
            return super().forward(input_ids=input_ids, **kwargs)
        finally:
            self._memory_context = None


def initialize_delta_time(module):
    """FLA's from-scratch timescales, rather than HF's pretrained placeholders."""
    if isinstance(module, Qwen3_5GatedDeltaNet):
        dt = torch.exp(torch.rand_like(module.dt_bias) * (math.log(.1)-math.log(.001)) + math.log(.001))
        # HF's helper skips checkpoint-loaded parameters. Raw copy_ here would
        # overwrite trained decay biases during from_pretrained's finalization.
        init.copy_(module.dt_bias, dt + torch.log(-torch.expm1(-dt)))


class HybridTextModel(MemoryDecoderMixin, Qwen3_5TextModel):
    def _init_weights(self, module):
        super()._init_weights(module)
        initialize_delta_time(module)


class DenseControlModel(MemoryDecoderMixin, Qwen3Model):
    pass


class MultiTokenPredictor(nn.Module):
    """One full gated-attention block reused for successive future positions.

    Shared embedding and LM head are passed by the caller, never duplicated.
    Source structure: vllm/model_executor/models/qwen3_5_mtp.py (Apache-2.0).
    """
    def __init__(self, config):
        super().__init__()
        self.pre_fc_norm_embedding = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        auxiliary = deepcopy(config)
        auxiliary.layer_types = ["full_attention"]
        auxiliary.num_hidden_layers = 1
        self.layer = Qwen3_5DecoderLayer(auxiliary, 0)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, embeddings, position_embeddings, mask):
        joined = torch.cat((self.pre_fc_norm_embedding(embeddings),
                            self.pre_fc_norm_hidden(hidden)), dim=-1)
        return self.norm(self.layer(self.fc(joined), position_embeddings=position_embeddings,
                                   attention_mask=mask, use_cache=False))


class HybridForCausalLM(Qwen3_5ForCausalLM):
    config_class = HybridConfig
    _keys_to_ignore_on_load_unexpected = []

    def _init_weights(self, module):
        super()._init_weights(module)
        initialize_delta_time(module)

    def __init__(self, config):
        # Let upstream initialize all decoder-specific parameters, then replace
        # only the decoder class to add memory hooks. No copied decoder code.
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5PreTrainedModel
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = HybridTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mtp = MultiTokenPredictor(config) if config.mtp_steps else None
        self.post_init()
        reset_memory(self)

    def training_loss(self, x, y, chunk_size, loss_mask):
        """Return objective sum and primary NLL sum for distinct logging.

        Auxiliary examples never cross EOS/PAD. All loss sums use the global
        primary-target denominator, preserving accumulation/DDP partitioning.
        Count base targets once for the training budget.
        """
        hidden = self.model(input_ids=x, use_cache=False).last_hidden_state
        base = hidden_nll(hidden, self.lm_head.weight, y, chunk_size, loss_mask)
        if self.mtp is None:
            return base, base.detach()
        auxiliary = hidden.new_zeros((), dtype=torch.float32)
        current = hidden
        mask = loss_mask
        for horizon in range(1, self.config.mtp_steps + 1):
            if x.shape[1] <= horizon:
                raise ValueError("Sequence too short for the configured MTP horizons")
            current = current[:, :-1]
            mask = mask[:, :-1] & loss_mask[:, horizon:]
            consumed = y[:, horizon - 1:-1]
            embeddings = self.model.embed_tokens(consumed)
            positions = torch.arange(horizon, x.shape[1], device=x.device).view(1, -1).expand(x.shape[0], -1)
            positional = self.model.rotary_emb(current, positions)
            causal = create_causal_mask(config=self.config, inputs_embeds=current,
                attention_mask=None, cache_position=torch.arange(current.shape[1], device=x.device), past_key_values=None,
                position_ids=positions)
            if self.config.mtp_activation_checkpointing and torch.is_grad_enabled():
                current = checkpoint(self.mtp,current,embeddings,positional,causal,use_reentrant=False)
            else:
                current = self.mtp(current, embeddings, positional, causal)
            nll = hidden_nll(current, self.lm_head.weight, y[:, horizon:], chunk_size, mask)
            auxiliary = auxiliary + nll
        auxiliary = auxiliary * (self.config.mtp_loss_weight / self.config.mtp_steps)
        return base + auxiliary, base.detach()


class DenseControlForCausalLM(Qwen3ForCausalLM):
    config_class = DenseControlConfig

    def __init__(self, config):
        from transformers.models.qwen3.modeling_qwen3 import Qwen3PreTrainedModel
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = DenseControlModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        reset_memory(self)


def require_fast_kernels(model):
    """Fail rather than accidentally run the slow Python DeltaNet fallback."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as upstream
    if not upstream.is_fast_path_available:
        raise RuntimeError("Hybrid training requires FLA and causal-conv1d fast kernels")
    for layer in model.model.layers:
        if layer.layer_type == "linear_attention":
            if (layer.linear_attn.chunk_gated_delta_rule is not upstream.chunk_gated_delta_rule or
                    layer.linear_attn.causal_conv1d_fn is not upstream.causal_conv1d_fn):
                raise RuntimeError("Unexpected DeltaNet kernel fallback")


def use_reference_delta(model):
    """Explicit FP32 likelihood evaluation; FLA's chunk kernel requires BF16.

    Never called by the trainer. This preserves the existing public suite's
    FP32 scoring protocol rather than silently evaluating the hybrid in BF16.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    for layer in model.model.layers:
        if getattr(layer, 'layer_type', None) == 'linear_attention':
            layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule


MODEL_TYPES = {HybridConfig.model_type: (HybridConfig, HybridForCausalLM),
               DenseControlConfig.model_type: (DenseControlConfig, DenseControlForCausalLM)}
for kind, (configuration, network) in MODEL_TYPES.items():
    AutoConfig.register(kind, configuration)
    AutoModelForCausalLM.register(configuration, network)


def build_model(config, tokenizer_json=None):
    from .engram import build_token_id_map
    values = dict(config)
    if tokenizer_json is not None:
        values["engram_token_id_map"] = build_token_id_map(tokenizer_json, values["vocab_size"])
    configuration, network = MODEL_TYPES[values["model_type"]]
    c = configuration(**values)
    c.use_cache = False
    c._attn_implementation = "sdpa"
    return network(c).float()


def load_model(path):
    import json
    from pathlib import Path
    kind = json.loads((Path(path) / "config.json").read_text())["model_type"]
    return MODEL_TYPES[kind][1].from_pretrained(path, local_files_only=True,
                                              dtype=torch.float32, attn_implementation="sdpa")
