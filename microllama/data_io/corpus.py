"""Stable document splits and bounded Parquet readers."""
import hashlib

SPLIT_SEED = 42
VALIDATION_PERMILLE = 5


def content_hash(text):
    # Normalizing whitespace groups trivial duplicate variants in the same split.
    return hashlib.blake2b(" ".join(text.split()).encode("utf-8"), digest_size=16).hexdigest()


def split_for(text):
    digest = content_hash(text)
    value = int(hashlib.blake2b(f"{SPLIT_SEED}:{digest}".encode(), digest_size=8).hexdigest(), 16)
    return "validation" if value % 1000 < VALIDATION_PERMILLE else "train"


def documents(path, batch_size=512):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["text", "id"]):
        yield from batch.to_pylist()
