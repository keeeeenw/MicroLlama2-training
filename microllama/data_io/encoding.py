"""Ordinary-document encoding with tokenizer control spellings kept as text."""
from __future__ import annotations

import json
from pathlib import Path
import re


ENCODING_POLICY_VERSION = "liquid-ordinary-text-v1"
_CONTROL_SPELLING = re.compile(r"<\|[^<>]*\|>")


def encoding_policy():
    """Return the stable policy identity for manifests and evaluation fixtures."""
    return {
        "version": ENCODING_POLICY_VERSION,
        "literal_control_spellings": "ordinary-model-pieces",
        "pinned_vocabulary": "unchanged",
    }


def require_encoding_policy(manifest, path):
    version = manifest.get("encoding_policy_version")
    if version != ENCODING_POLICY_VERSION:
        found = version if version is not None else "legacy/unspecified"
        raise RuntimeError(
            f"Incompatible corpus encoding at {path}: found {found}, "
            f"expected {ENCODING_POLICY_VERSION}"
        )


class OrdinaryTextTokenizer:
    """A runtime-only tokenizer clone that cannot recognize control spellings."""

    def __init__(self, tokenizer, control_ids):
        self._tokenizer = tokenizer
        self.control_ids = frozenset(int(token_id) for token_id in control_ids)

    def _check(self, encoding):
        found = self.control_ids.intersection(encoding.ids)
        if found:
            raise RuntimeError(f"Ordinary text encoding emitted control token IDs: {sorted(found)}")
        return encoding

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("Ordinary document encoding never inserts special tokens")
        return self._check(self._tokenizer.encode(text, add_special_tokens=False))

    def encode_batch(self, inputs, add_special_tokens=False, **kwargs):
        if add_special_tokens:
            raise ValueError("Ordinary document encoding never inserts special tokens")
        encodings = self._tokenizer.encode_batch(inputs, add_special_tokens=False, **kwargs)
        return [self._check(encoding) for encoding in encodings]

    def decode(self, ids, skip_special_tokens=False):
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def __getattr__(self, name):
        return getattr(self._tokenizer, name)


def load_ordinary_text_tokenizer(path: str | Path) -> OrdinaryTextTokenizer:
    """Load the pinned tokenizer without matching literal control spellings.

    The on-disk tokenizer is only read. Control AddedTokens are removed from an
    in-memory JSON clone, leaving model vocabulary and ordinary additions intact.
    """
    from tokenizers import Tokenizer

    source = Path(path)
    tokenizer_json = json.loads(source.read_text())
    added_tokens = tokenizer_json.get("added_tokens", [])
    retained = []
    control_ids = set()
    for token in added_tokens:
        content = token.get("content", "")
        # Some reserved controls (notably tool-call markers) are marked
        # special=false upstream but still must not become single control IDs
        # when seen in source prose.
        if token.get("special", False) or _CONTROL_SPELLING.fullmatch(content):
            control_ids.add(token["id"])
        else:
            retained.append(token)
    tokenizer_json["added_tokens"] = retained

    original = Tokenizer.from_file(str(source))
    ordinary = Tokenizer.from_str(json.dumps(tokenizer_json, ensure_ascii=False))
    if ordinary.get_vocab(with_added_tokens=True) != original.get_vocab(with_added_tokens=True):
        raise RuntimeError("Removing control-token matching changed the pinned tokenizer vocabulary")
    return OrdinaryTextTokenizer(ordinary, control_ids)


def encode_ordinary_text(tokenizer: OrdinaryTextTokenizer, text: str):
    """Encode one document as ordinary text, without BOS/EOS insertion."""
    return tokenizer.encode(text, add_special_tokens=False)


def encode_ordinary_text_batch(tokenizer: OrdinaryTextTokenizer, texts: list[str]):
    """Batch counterpart to :func:`encode_ordinary_text`."""
    return tokenizer.encode_batch(texts, add_special_tokens=False)
