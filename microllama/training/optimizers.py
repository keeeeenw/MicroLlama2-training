"""One pinned single-device Muon candidate plus the unchanged AdamW control."""

import torch

from .vendor.muon import SingleDeviceMuonWithAuxAdam


def build_optimizer(model, kind="adamw", adam_lr=3e-4, muon_lr=0.02,
                    weight_decay=0.1, fused=False, memory_lr_multiplier=1.0):
    if kind not in ("adamw", "muon"):
        raise ValueError("Unknown optimizer")
    embedding = model.get_input_embeddings().weight
    if memory_lr_multiplier <= 0 or (kind != "adamw" and memory_lr_multiplier != 1.0):
        raise ValueError("Special memory rates require positive AdamW configuration")
    matrices, decay, no_decay, memory = [], [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Only hidden projections: embeddings/tied head and memory stay on AdamW.
        hidden = parameter.ndim == 2 and parameter is not embedding and "engram" not in name
        table = ".engram." in name and ("bigram_table" in name or "trigram_table" in name)
        if table and memory_lr_multiplier != 1.0:
            memory.append(parameter)
        elif kind == "muon" and hidden:
            matrices.append(parameter)
        elif parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    all_parameters = [p for p in model.parameters() if p.requires_grad]
    grouped = matrices + decay + no_decay + memory
    if len(grouped) != len(all_parameters) or len({id(p) for p in grouped}) != len(grouped):
        raise ValueError("Optimizer groups must cover each tied parameter exactly once")
    if kind == "adamw":
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        if memory:
            groups.append({"params": memory, "weight_decay": 0.0,
                           "lr": adam_lr * memory_lr_multiplier, "group_name": "memory_tables"})
        optimizer = torch.optim.AdamW(groups, lr=adam_lr, betas=(0.9, 0.95),
                                      eps=1e-8, fused=fused)
    else:
        if not matrices:
            raise ValueError("Muon requires hidden projection matrices")
        groups = [dict(params=matrices, use_muon=True, lr=muon_lr,
                       momentum=0.95, weight_decay=weight_decay)]
        for parameters, wd in [(decay, weight_decay), (no_decay, 0.0)]:
            if parameters:
                groups.append(dict(params=parameters, use_muon=False, lr=adam_lr,
                                   betas=(0.9, 0.95), eps=1e-8, weight_decay=wd))
        optimizer = SingleDeviceMuonWithAuxAdam(groups)
    for group in optimizer.param_groups:
        group["peak_lr"] = group["lr"]
    return optimizer


def set_schedule_fraction(optimizer, fraction):
    if not 0 < fraction <= 1:
        raise ValueError("Schedule fraction must be in (0, 1]")
    for group in optimizer.param_groups:
        group["lr"] = group["peak_lr"] * fraction
