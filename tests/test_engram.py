import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from microllama.training.engram import (
    EngramMemory,
    EngramQwen3ForCausalLM,
    EngramState,
    build_engram_model,
    engram_decode_step,
    engram_prefill,
    load_engram_model,
    save_engram_model,
    LOOKUP_V2,
    canonical_token_map,
    prime_table_sizes,
)
from microllama.training.objective import chunked_loss
from transformers import AutoModelForCausalLM, Qwen3ForCausalLM, Qwen3Config


TINY = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000,
    "max_position_embeddings": 64,
    "tie_word_embeddings": True,
    "attention_dropout": 0.0,
    "pad_token_id": 0,
    "eos_token_id": 7,
    "bos_token_id": 1,
}


def tiny_engram(dense=None, revised=False):
    return build_engram_model(
        TINY,
        dense_model=dense,
        table_size=97,
        embedding_dim=8,
        layers=(0, 1),
        **(dict(lookup_version=LOOKUP_V2, hash_heads=4, token_id_map=list(range(64))) if revised else {}),
    )


def test_dense_warm_start_is_near_neutral_and_keeps_tied_embeddings():
    torch.manual_seed(11)
    config = Qwen3Config(**TINY)
    dense = Qwen3ForCausalLM(config).eval()
    engram = tiny_engram(dense).eval()
    ids = torch.randint(1, 64, (2, 13))
    with torch.no_grad():
        expected = dense(input_ids=ids, use_cache=False).logits
        actual = engram(input_ids=ids, use_cache=False).logits
    assert engram.lm_head.weight is engram.model.embed_tokens.weight
    assert torch.max(torch.abs(expected - actual)).item() < 1e-3


def test_table_gate_and_value_receive_gradients_and_update():
    torch.manual_seed(12)
    model = tiny_engram().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([[2, 3, 4, 5, 6, 8, 9, 10]])
    target = torch.tensor([[3, 4, 5, 6, 8, 9, 10, 11]])
    table_before = model.model.engram.bigram_table.weight.detach().clone()
    key_before = model.model.engram.key_proj.weight.detach().clone()

    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, use_cache=False).logits
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten())
        loss.backward()
        for name in ("bigram_table.weight", "trigram_table.weight", "key_proj.weight", "value_proj.weight", "residual_scale"):
            grad = dict(model.model.engram.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
        optimizer.step()

    assert not torch.equal(table_before, model.model.engram.bigram_table.weight)
    assert not torch.equal(key_before, model.model.engram.key_proj.weight)
    assert model.model.engram.residual_scale.item() != pytest.approx(1e-3)
    decay = [parameter for parameter in model.parameters() if parameter.ndim >= 2]
    no_decay = [parameter for parameter in model.parameters() if parameter.ndim < 2]
    assert all(parameter is not model.model.engram.residual_scale for parameter in decay)
    assert any(parameter is model.model.engram.residual_scale for parameter in no_decay)


def test_objective_chunked_loss_direct_model_path_updates_memory():
    torch.manual_seed(121)
    model = tiny_engram().train()
    ids = torch.tensor([[2, 3, 4, 5, 6, 8, 9]])
    targets = torch.tensor([[3, 4, 5, 6, 8, 9, 10]])
    loss = chunked_loss(model, ids, targets, chunk_size=3)
    loss.backward()
    table_grad = model.model.engram.bigram_table.weight.grad
    scale_grad = model.model.engram.residual_scale.grad
    assert table_grad is not None and table_grad.abs().sum() > 0
    assert scale_grad is not None and torch.isfinite(scale_grad) and scale_grad.abs() > 0


@pytest.mark.parametrize("revised", [False, True])
def test_future_tokens_do_not_change_prefix_logits(revised):
    torch.manual_seed(13)
    model = tiny_engram(revised=revised).eval()
    first = torch.randint(1, 64, (1, 12))
    second = first.clone()
    second[:, 7:] = torch.randint(1, 64, (1, 5))
    with torch.no_grad():
        a = model(input_ids=first, use_cache=False).logits
        b = model(input_ids=second, use_cache=False).logits
    torch.testing.assert_close(a[:, :7], b[:, :7], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("revised", [False, True])
def test_history_carries_between_chunks_and_resets_at_eos_and_pad(revised):
    memory = EngramMemory(
        tiny_engram(revised=revised).config
    ).eval()
    state0 = memory.empty_state(1, torch.device("cpu"))
    hidden = torch.zeros(1, 2, 32)
    _, state1 = memory(hidden, torch.tensor([[12, 13]]), state0)
    _, state2 = memory(hidden[:, :1], torch.tensor([[14]]), state1)
    full_hidden = torch.zeros(1, 3, 32)
    _, full_state = memory(full_hidden, torch.tensor([[12, 13, 14]]), state0)
    torch.testing.assert_close(state2.history_ids, full_state.history_ids)

    a = torch.tensor([[20, 7, 21, 22]])
    b = torch.tensor([[30, 7, 21, 22]])
    _, out_a = memory(torch.zeros(1, 4, 32), a, state0)
    _, out_b = memory(torch.zeros(1, 4, 32), b, state0)
    assert torch.equal(out_a.history_ids, out_b.history_ids)

    # The memory output after EOS is independent of the preceding document.
    hidden = torch.zeros(1, 4, 32)
    out_a, _ = memory(hidden, a, state0)
    out_b, _ = memory(hidden, b, state0)
    torch.testing.assert_close(out_a[:, 2:], out_b[:, 2:], atol=0, rtol=0)

    # PAD also breaks the n-gram suffix and leaves an empty final state.
    _, pad_state = memory(torch.zeros(1, 3, 32), torch.tensor([[21, 0, 22]]), state0)
    assert pad_state.history_ids.tolist() == [[-1, 22]]


@pytest.mark.parametrize("revised", [False, True])
def test_cached_prefill_and_incremental_decode_match_full_forward(revised):
    torch.manual_seed(14)
    model = tiny_engram(revised=revised).eval()
    ids = torch.tensor([[2, 3, 4, 5, 6, 8, 9]])
    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits
        prefix = engram_prefill(model, ids[:, :4])
        cache, state = prefix.past_key_values, prefix.engram_state
        pieces = [prefix.logits]
        for index in range(4, ids.shape[1]):
            step = engram_decode_step(
                model,
                ids[:, index : index + 1],
                past_key_values=cache,
                engram_state=state,
            )
            cache, state = step.past_key_values, step.engram_state
            pieces.append(step.logits)
    incremental = torch.cat(pieces, dim=1)
    torch.testing.assert_close(incremental, full, atol=3e-5, rtol=3e-5)
    assert state.history_ids.tolist() == [[8, 9]]


@pytest.mark.parametrize("revised", [False, True])
def test_save_reload_is_strict_and_optimizer_resume_matches(tmp_path, revised):
    torch.manual_seed(15)
    model = tiny_engram(revised=revised).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ids = torch.tensor([[2, 3, 4, 5, 6, 8]])
    target = torch.tensor([[3, 4, 5, 6, 8, 9]])

    def update(current, opt):
        opt.zero_grad(set_to_none=True)
        logits = current(input_ids=ids, use_cache=False).logits
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), target.flatten()).backward()
        opt.step()

    update(model, optimizer)
    directory = save_engram_model(model, tmp_path / "engram")
    torch.save({"optimizer": optimizer.state_dict()}, directory / "training_state.pt")
    update(model, optimizer)

    restored = load_engram_model(directory).train()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = torch.load(directory / "training_state.pt", weights_only=True)
    restored_optimizer.load_state_dict(state["optimizer"])
    update(restored, restored_optimizer)
    for left, right in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    metadata = json.loads((directory / "engram_config.json").read_text())
    assert metadata["added_trainable_parameters"] == model.engram_parameter_count()
    assert metadata["implementation"].startswith("normalized multi-head" if revised else "single hash head")
    auto_restored = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True)
    assert isinstance(auto_restored, EngramQwen3ForCausalLM)
    assert auto_restored.engram_parameter_count() == model.engram_parameter_count()


@pytest.mark.parametrize("revised", [False, True])
def test_fresh_process_rng_optimizer_and_next_update_resume(tmp_path, revised):
    torch.manual_seed(151)
    model = tiny_engram(revised=revised).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def random_update(current, opt):
        ids = torch.randint(1, 64, (1, 8))
        targets = torch.randint(1, 64, (1, 8))
        opt.zero_grad(set_to_none=True)
        chunked_loss(current, ids, targets, chunk_size=4).backward()
        opt.step()

    random_update(model, optimizer)
    directory = save_engram_model(model, tmp_path / "fresh-process")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
        "step": 1,
    }, directory / "training_state.pt")
    random_update(model, optimizer)
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    expected_next_rng = torch.rand(5)

    child = r'''import sys, torch
from pathlib import Path
from microllama.training.engram import load_engram_model
from microllama.training.objective import chunked_loss
p = Path(sys.argv[1])
m = load_engram_model(p).train()
o = torch.optim.AdamW(m.parameters(), lr=1e-3)
s = torch.load(p / "training_state.pt", map_location="cpu", weights_only=True)
o.load_state_dict(s["optimizer"])
torch.set_rng_state(s["rng"])
x = torch.randint(1, 64, (1, 8))
y = torch.randint(1, 64, (1, 8))
o.zero_grad(set_to_none=True)
chunked_loss(m, x, y, chunk_size=4).backward()
o.step()
torch.save({"model": m.state_dict(), "next_rng": torch.rand(5)}, p / "fresh-process-result.pt")
'''
    completed = subprocess.run(
        [sys.executable, "-c", child, str(directory)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0
    actual = torch.load(directory / "fresh-process-result.pt", map_location="cpu", weights_only=True)
    for key, expected_value in expected.items():
        torch.testing.assert_close(actual["model"][key], expected_value, atol=0, rtol=0)
    torch.testing.assert_close(actual["next_rng"], expected_next_rng, atol=0, rtol=0)


def test_default_added_parameter_count_and_architecture_marker():
    with torch.device("meta"):
        model = build_engram_model(
            {**TINY, "hidden_size": 1024, "num_hidden_layers": 24,
             "intermediate_size": 2816, "num_attention_heads": 16,
             "num_key_value_heads": 4, "head_dim": 64},
            table_size=194_000,
            embedding_dim=128,
            layers=(3, 11),
        )
    assert model.engram_parameter_count() == 50_190_337
    assert model.config.model_type == "engram_qwen3"
    assert model.config.architectures == ["EngramQwen3ForCausalLM"]


def test_generate_fails_instead_of_resetting_explicit_memory_history():
    model = tiny_engram()
    with pytest.raises(RuntimeError, match="EngramState"):
        model.generate(torch.tensor([[2, 3]]), max_new_tokens=1)


def test_memory_normalization_preserves_controls_partial_bytes_whitespace_and_padding():
    values = ["<pad>", "<bos>", "Hello", " hello", "ＨＥＬＬＯ", "\ufffd", "\ufffd", "<eos>", " ", "  "]
    mapping = canonical_token_map(values, {0, 1, 7}, 12)
    assert mapping[2:5] == [2, 2, 2]
    assert all(mapping[i] == i for i in (0, 1, 5, 6, 7, 8, 9, 10, 11))
    assert values[3] == " hello"


def test_revised_hashes_normalize_lookup_and_keep_independent_prime_heads():
    mapping = list(range(64))
    mapping[13] = 12
    model = build_engram_model(TINY, table_size=97, embedding_dim=8, layers=(0, 1),
                              lookup_version=LOOKUP_V2, hash_heads=4, token_id_map=mapping)
    memory = model.model.engram
    state = memory.empty_state(1, torch.device("cpu"))
    a = memory._indices(torch.tensor([[2, 12, 14, 15]]), state)
    b = memory._indices(torch.tensor([[2, 13, 14, 15]]), state)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)
    assert a[0].shape == (1, 4, 4)
    sizes = [t.num_embeddings for t in memory.tables()]
    assert sizes == prime_table_sizes(97, 8) and len(set(sizes)) == 8
    assert len({tuple(a[0][0, :, h].tolist()) for h in range(4)}) == 4
    with pytest.raises(ValueError, match="controls"):
        build_engram_model(TINY, lookup_version=LOOKUP_V2, hash_heads=4,
                           token_id_map=[0] * 64, layers=(0, 1))


def test_revised_default_stays_near_original_parameter_budget():
    with torch.device("meta"):
        model = build_engram_model({**TINY, "hidden_size": 1024, "num_hidden_layers": 24,
            "intermediate_size": 2816, "num_attention_heads": 16,
            "num_key_value_heads": 4, "head_dim": 64}, lookup_version=LOOKUP_V2,
            hash_heads=4, token_id_map=list(range(64)))
    assert 50_190_337 <= model.engram_parameter_count() < 50_250_000
