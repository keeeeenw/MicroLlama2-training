import json

import pytest

from microllama.common.paths import data_root


def test_downloaded_tokenizer_preserves_text_and_tool_markers():
    from tokenizers import Tokenizer
    directory = data_root() / "tokenizer"
    if not (directory / "tokenizer.json").exists():
        pytest.skip("Tokenizer assets not downloaded")
    tok = Tokenizer.from_file(str(directory / "tokenizer.json"))
    source = json.loads((directory / "source_manifest.json").read_text())
    assert source["max_token_id"] < 65536
    for text in ("Hello  world\n\tÅ中🙂", "1234567890", "  spaces  ", "a\r\nb"):
        assert tok.decode(tok.encode(text, add_special_tokens=False).ids) == text
    for marker in ("<|im_start|>", "<|im_end|>", "<|tool_call_start|>", "<|tool_call_end|>"):
        ids = tok.encode(marker, add_special_tokens=False).ids
        assert ids == [tok.token_to_id(marker)]


def test_chat_template_renders_complete_tool_exchange():
    transformers = pytest.importorskip("transformers")
    directory = data_root() / "tokenizer"
    if not (directory / "tokenizer.json").exists():
        pytest.skip("Tokenizer assets not downloaded")
    tok = transformers.AutoTokenizer.from_pretrained(directory, local_files_only=True)
    messages = [
        {"role": "user", "content": "Look up Paris weather."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": '{"temperature": 18}'},
        {"role": "assistant", "content": "It is 18 degrees."}]
    result = tok.apply_chat_template(messages, tokenize=False)
    assert "<|tool_call_start|>[weather(city='Paris')]<|tool_call_end|>" in result
    assert '<|im_start|>tool\n{"temperature": 18}<|im_end|>' in result
