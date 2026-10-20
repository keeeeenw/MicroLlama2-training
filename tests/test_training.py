import copy
import json
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from microllama.training.cuda import build_model, chunked_loss, save_checkpoint
from microllama.data_io.blocks import TokenBlocks
from test_data import write_shard


TINY = {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 8, "rms_norm_eps": 1e-6, "rope_theta": 10000,
        "max_position_embeddings": 64, "tie_word_embeddings": True,
        "attention_dropout": 0.0, "pad_token_id": 0}


def test_chunked_loss_matches_full_logits_and_gradients():
    torch.manual_seed(42)
    a = build_model(TINY)
    b = copy.deepcopy(a)
    x = torch.randint(1, 64, (2, 13))
    y = torch.randint(1, 64, (2, 13))
    chunk = chunked_loss(a, x, y, 7)
    logits = b(x).logits
    full = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), y.reshape(-1))
    torch.testing.assert_close(chunk, full)
    chunk.backward()
    full.backward()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-7, rtol=1e-4)


def test_future_tokens_do_not_change_prefix():
    model = build_model(TINY).eval()
    x = torch.randint(1, 64, (1, 16))
    y = x.clone()
    y[:, 8:] = torch.randint(1, 64, (1, 8))
    with torch.no_grad():
        torch.testing.assert_close(model(x).logits[:, :8], model(y).logits[:, :8])


def test_checkpoint_restores_optimizer_and_next_step(tmp_path):
    from safetensors.torch import load_model
    torch.manual_seed(1)
    root = tmp_path
    (root / "tokens").mkdir()
    write_shard(root / "tokens" / "a", range(1, 63))
    data = TokenBlocks(root / "tokens", 8)
    model = build_model(TINY)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def update(m, o, d):
        batch = torch.from_numpy(d.next_batch(1))
        o.zero_grad()
        chunked_loss(m, batch[:, :-1], batch[:, 1:], 8).backward()
        o.step()

    update(model, opt, data)
    checkpoint = save_checkpoint(root, "test", model, opt, data, {}, 1, 8)
    update(model, opt, data)
    other = build_model(TINY)
    load_model(other, str(checkpoint / "model.safetensors"))
    other_opt = torch.optim.AdamW(other.parameters(), lr=1e-3)
    state = torch.load(checkpoint / "training_state.pt", weights_only=True)
    other_opt.load_state_dict(state["optimizer"])
    other_data = TokenBlocks(root / "tokens", 8)
    other_data.load_state_dict(state["data"])
    update(other, other_opt, other_data)
    for p, q in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_full_parameter_count_and_tied_embeddings():
    config = json.loads(Path("configs/model-338m.json").read_text())
    with torch.device("meta"):
        model = build_model(config)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert sum(p.numel() for p in model.parameters()) == 337_694_720
