"""Cheap uncached greedy continuations for the active trainer's own W&B run."""

from contextlib import nullcontext
import json
from pathlib import Path

import torch

from ..data_io.encoding import encode_ordinary_text, load_ordinary_text_tokenizer


def fixed_samples(model, tokenizer_path, settings_path):
    """Use only the primary head, ordinary text IDs, and fixed greedy decoding."""
    tokenizer = load_ordinary_text_tokenizer(tokenizer_path)
    vocabulary = json.loads(Path(tokenizer_path).read_text())
    settings = json.loads(Path(settings_path).read_text())
    eos = model.config.eos_token_id
    suppressed = {t['id'] for t in vocabulary.get('added_tokens',[]) if t['id'] != eos}
    suppressed.update(range(tokenizer.get_vocab_size(with_added_tokens=True), model.config.vocab_size))
    device = next(model.parameters()).device
    original = model.training
    model.eval()
    rows = []
    try:
        with torch.inference_mode(), (torch.autocast('cuda',dtype=torch.bfloat16)
                                     if device.type=='cuda' else nullcontext()):
            for prompt in settings['generation_prompts']:
                prefix = [model.config.bos_token_id]+encode_ordinary_text(tokenizer,prompt).ids
                ids = torch.tensor([prefix],device=device)
                generated = []
                for _ in range(settings['max_new_tokens']):
                    hidden = model.model(input_ids=ids,use_cache=False).last_hidden_state[:,-1]
                    logits = model.lm_head(hidden).float()
                    logits[:,sorted(suppressed)] = -torch.inf
                    next_id = int(logits.argmax(-1))
                    generated.append(next_id)
                    if next_id == eos:
                        break
                    ids = torch.cat((ids,torch.tensor([[next_id]],device=device)),dim=1)
                decoded = generated[:-1] if generated[-1]==eos else generated
                rows.append({'prompt':prompt,'continuation':tokenizer.decode(decoded,skip_special_tokens=True),
                             'generated_token_ids':generated,'ended_with_eos':generated[-1]==eos})
    finally:
        model.train(original)
    return rows
