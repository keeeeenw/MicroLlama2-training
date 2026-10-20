"""Fetch an existing modern tokenizer; never train or alter its vocabulary."""
import argparse
import hashlib
import json
from pathlib import Path

from ..common.paths import configure_storage

DEFAULT_REPO = "LiquidAI/LFM2.5-230M-Base"
DEFAULT_REVISION = "9d2be5519834990d30996f878b6771cccbd24f2c"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default=DEFAULT_REPO)
    args = p.parse_args()
    root = configure_storage()
    from huggingface_hub import HfApi, hf_hub_download
    from tokenizers import Tokenizer
    out = root / "tokenizer"
    manifest_path = out / "source_manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["repo"] != args.repo:
            raise RuntimeError("Different tokenizer already exists; choose a new data root")
        revision = saved["revision"]
    else:
        revision = DEFAULT_REVISION if args.repo == DEFAULT_REPO else HfApi().model_info(args.repo).sha
    info = HfApi().model_info(args.repo, revision=revision)
    names = {s.rfilename for s in info.siblings}
    downloaded = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if name not in names:
            continue
        path = Path(hf_hub_download(args.repo, name, revision=revision, local_dir=out))
        downloaded[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    config = json.loads((out / "tokenizer_config.json").read_text())
    def token_id(name):
        value = config.get(name)
        if isinstance(value, dict):
            value = value["content"]
        return tok.token_to_id(value) if value else None
    max_id = max(tok.get_vocab().values())
    if max_id > 65535:
        raise RuntimeError("This tokenizer exceeds uint16 storage; choose a compact tokenizer")
    eos = token_id("eos_token")
    bos = token_id("bos_token")
    pad = token_id("pad_token")
    if eos is None:
        raise RuntimeError("Tokenizer has no EOS token")
    manifest = {"repo": args.repo, "revision": revision, "files": downloaded,
                "vocab_size": max_id + 1, "max_token_id": max_id,
                "bos_token_id": bos, "eos_token_id": eos, "pad_token_id": pad,
                "tokenizer_sha256": downloaded["tokenizer.json"]}
    # Keep the vocabulary intact; generic fast loading supports portable exports.
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config["model_max_length"] = 8192
    (out / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
