import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from microllama.training.cuda import build_model, chunked_loss
from microllama.training.optimizers import build_optimizer, set_schedule_fraction
from test_training import TINY


def test_muon_groups_exclude_tied_embedding_and_norms():
    model = build_model(TINY, gradient_checkpointing=False)
    optimizer = build_optimizer(model, "muon")
    muon = [p for g in optimizer.param_groups if g["use_muon"] for p in g["params"]]
    assert model.model.embed_tokens.weight is model.lm_head.weight
    assert all(p is not model.model.embed_tokens.weight and p.ndim == 2 for p in muon)
    assert len({id(p) for g in optimizer.param_groups for p in g["params"]}) == len(list(model.parameters()))
    set_schedule_fraction(optimizer, 0.5)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert all(g["lr"] == 0.00015 for g in optimizer.param_groups if not g["use_muon"])


@pytest.mark.parametrize("kind", ["adamw", "muon"])
def test_finite_update_and_exact_optimizer_resume(kind):
    torch.manual_seed(3)
    a = build_model(TINY, gradient_checkpointing=False)
    oa = build_optimizer(a, kind)
    x = torch.randint(1, 64, (1, 7))
    y = torch.randint(1, 64, (1, 7))

    def update(m, o):
        o.zero_grad(set_to_none=True)
        chunked_loss(m, x, y, 3).backward()
        o.step()
        assert all(torch.isfinite(p).all() for p in m.parameters())

    update(a, oa)
    b = copy.deepcopy(a)
    ob = build_optimizer(b, kind)
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    update(a, oa)
    update(b, ob)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)


def test_memory_table_rate_no_decay_and_exact_resume():
    from test_engram import tiny_engram
    a = tiny_engram(revised=True)
    oa = build_optimizer(a, memory_lr_multiplier=5)
    tables = {id(t.weight) for t in a.model.engram.tables()}
    memory_groups = [g for g in oa.param_groups if g.get("group_name") == "memory_tables"]
    assert len(memory_groups) == 1
    assert {id(p) for p in memory_groups[0]["params"]} == tables
    assert memory_groups[0]["lr"] == pytest.approx(.0015)
    assert memory_groups[0]["weight_decay"] == 0
    set_schedule_fraction(oa, .4)
    assert memory_groups[0]["lr"] == pytest.approx(.0006)
    grouped = [p for g in oa.param_groups for p in g["params"]]
    assert len(grouped) == len({id(p) for p in a.parameters()}) == len({id(p) for p in grouped})
    x = torch.tensor([[2, 3, 4, 5, 6, 8, 9]])
    y = torch.tensor([[3, 4, 5, 6, 8, 9, 10]])
    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        chunked_loss(model, x, y, 3).backward()
        for t in model.model.engram.tables():
            assert t.weight.grad.abs().sum() > 0
        optimizer.step()
    update(a, oa)
    b = copy.deepcopy(a)
    ob = build_optimizer(b, memory_lr_multiplier=5)
    ob.load_state_dict(copy.deepcopy(oa.state_dict()))
    update(a, oa)
    update(b, ob)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
