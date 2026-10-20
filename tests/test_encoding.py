import json

import pytest

from microllama.common.paths import data_root


def test_pinned_liquid_tokenizer_encodes_control_spellings_as_text():
    from tokenizers import Tokenizer
    from microllama.data_io.encoding import (
        ENCODING_POLICY_VERSION,
        encode_ordinary_text,
        encode_ordinary_text_batch,
        encoding_policy,
        load_ordinary_text_tokenizer,
    )

    path = data_root() / "tokenizer" / "tokenizer.json"
    if not path.exists():
        pytest.skip("Pinned tokenizer assets are not downloaded")
    original_bytes = path.read_bytes()
    original = Tokenizer.from_file(str(path))
    tokenizer_json = json.loads(original_bytes)
    ordinary = load_ordinary_text_tokenizer(path)

    assert ordinary.get_vocab(with_added_tokens=True) == original.get_vocab(with_added_tokens=True)
    assert encoding_policy()["version"] == ENCODING_POLICY_VERSION
    controls = [item for item in tokenizer_json["added_tokens"]
                if item.get("special", False) or
                (item["content"].startswith("<|") and item["content"].endswith("|>"))]
    assert controls
    assert len(ordinary.control_ids) == len({item["id"] for item in controls})

    # Every pinned special/control spelling is broken into ordinary model pieces.
    for item in controls:
        text = item["content"]
        encoding = encode_ordinary_text(ordinary, text)
        assert not ordinary.control_ids.intersection(encoding.ids), text
        assert ordinary.decode(encoding.ids) == text

    # The pinned non-special <think> tokens and unrelated prose retain their IDs.
    probes = [
        "The quick brown fox jumps over the lazy dog.",
        "Integers 1234567890; decimal 3.1415926535; ratio 22/7.",
        "Unicode: café, Ελληνικά, 東京, العربية; math: ∑(x²) ≤ 10 → ∞.",
        "A literal <think> tag remains an ordinary tokenizer addition.",
    ]
    for text, encoding in zip(probes, encode_ordinary_text_batch(ordinary, probes)):
        assert encoding.ids == original.encode(text, add_special_tokens=False).ids
        assert ordinary.decode(encoding.ids) == text

    marker = "<|im_start|>"
    assert original.encode(marker, add_special_tokens=False).ids == [original.token_to_id(marker)]
    assert encode_ordinary_text(ordinary, marker).ids != [original.token_to_id(marker)]
    assert path.read_bytes() == original_bytes


def test_encoding_policy_is_explicit_and_legacy_chunks_are_rejected():
    from microllama.data_io.encoding import (
        ENCODING_POLICY_VERSION,
        encoding_policy,
        require_encoding_policy,
    )

    assert encoding_policy() == {
        "version": ENCODING_POLICY_VERSION,
        "literal_control_spellings": "ordinary-model-pieces",
        "pinned_vocabulary": "unchanged",
    }
    require_encoding_policy({"encoding_policy_version": ENCODING_POLICY_VERSION}, "new-chunk")
    with pytest.raises(RuntimeError, match="legacy/unspecified"):
        require_encoding_policy({}, "published-old-chunk")
