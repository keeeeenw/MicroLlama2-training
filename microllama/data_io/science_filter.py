"""Deterministic, conservative selection of explanatory Cosmopedia v2 science.

This is a lexical admission rule, not a factual-accuracy or contamination guarantee.
The caller supplies benchmark documents; this module never fetches data or models.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
import re
import unicodedata


RULE_VERSION = "cosmopedia-science-v2"
_MIN_WORDS = 32
_TOPIC_WORDS = 120
_FORMATS = ("tutorial", "explanation", "explanatory_article", "article", "blogpost", "blog_post", "wikihow")
# Two distinct anchors in generated text and at least one strong anchor are required.
# Singular and plural forms are separate alternatives, not separate evidence.
_DOMAINS = {
    "physical_science": (
        ("atom", "atoms"), ("molecule", "molecules"), ("electron", "electrons"),
        ("proton", "protons"), ("neutron", "neutrons"), ("chemical reaction", "chemical reactions"),
        ("electric current",), ("electric charge",), ("magnetic field", "magnetic fields"),
        ("kinetic energy",), ("potential energy",), ("thermal energy",), ("heat transfer",),
        ("sound wave", "sound waves"), ("wavelength", "wavelengths"), ("gravity",),
        ("friction",), ("acceleration",), ("density",), ("buoyancy",),
        ("force", "forces"), ("mass",), ("energy",), ("temperature",),
        ("melting",), ("freezing",), ("solid", "solids"), ("liquid", "liquids"), ("gas", "gases"),
    ),
    "life_science": (
        ("photosynthesis",), ("chlorophyll",), ("chloroplast", "chloroplasts"),
        ("cell membrane", "cell membranes"), ("cellular respiration",), ("mitochondria", "mitochondrion"),
        ("ecosystem", "ecosystems"), ("food chain", "food chains"), ("natural selection",),
        ("habitat", "habitats"), ("organism", "organisms"), ("reproduction",),
        ("bacteria", "bacterium"), ("enzyme", "enzymes"), ("dna",), ("gene", "genes"),
        ("adaptation", "adaptations"), ("nutrient", "nutrients"), ("oxygen",), ("carbon dioxide",),
        ("plant", "plants"), ("animal", "animals"), ("cell", "cells"),
    ),
    "earth_space_science": (
        ("water cycle",), ("evaporation",), ("condensation",), ("transpiration",),
        ("plate tectonics",), ("tectonic plate", "tectonic plates"), ("rock cycle",),
        ("sedimentary",), ("igneous",), ("metamorphic",), ("erosion",),
        ("solar system",), ("earth rotation", "earth s rotation"), ("orbit", "orbits"),
        ("lunar",), ("solar eclipse",), ("precipitation",), ("weathering",),
        ("atmosphere",), ("earthquake", "earthquakes"), ("volcano", "volcanoes"),
        ("planet", "planets"), ("climate",), ("fossil", "fossils"), ("season", "seasons"),
    ),
    "scientific_method": (
        ("scientific method",), ("controlled experiment", "controlled experiments"),
        ("control group",), ("independent variable",), ("dependent variable",),
        ("testable hypothesis",), ("experimental design",), ("laboratory experiment",),
        ("hypothesis", "hypotheses"), ("measurement", "measurements"),
    ),
}
_WEAK = {"force", "mass", "energy", "temperature", "melting", "freezing", "solid", "liquid", "gas",
         "ecosystem", "habitat", "reproduction", "adaptation", "nutrient", "oxygen", "carbon dioxide",
         "plant", "animal", "cell", "orbit", "lunar", "precipitation", "weathering", "atmosphere",
         "planet", "climate", "fossil", "season", "hypothesis", "measurement"}
_EXCLUSIONS = {
    "unsupported_healing": (("crystal healing", "healing crystals", "human energy field",
                             "human energy fields", "chakra balancing", "quantum healing", "reiki"), 1),
    "coding": (("python programming", "python code", "python tutorial", "import numpy", "import pandas",
                "javascript", "programming", "software development", "computer program", "computer code",
                "code snippet", "coding tutorial", "for loop", "data structure", "machine learning"), 1),
    "business": (("marketing", "branding", "business strategy", "financial management", "corporate ecosystem"), 1),
    "humanities": (("literary analysis", "literary criticism", "social science", "political science",
                    "religious beliefs", "sociology", "theology"), 1),
    "business_context": (("customer", "customers", "sales", "profit", "profits", "investment", "investments",
                          "entrepreneur", "entrepreneurs", "business", "businesses"), 2),
    "humanities_context": (("poetry", "literature", "religion", "democracy", "politics", "novel", "novels",
                            "theater", "theatre", "philosophy"), 2),
    "fiction": (("once upon a time", "magical kingdom", "fairy tale", "fairy tales"), 1),
}
_TOPIC_EXCLUSIONS = {
    "relationships": (("marriage and divorce", "blended families", "extended family",
                       "personal relationships"), 1),
    "fitness_practice": (("pilates practice", "pilates exercises", "golfing", "golfers"), 1),
    "policing": (("undercover operations", "police procedurals"), 1),
    "logistics": (("cargo handling", "cargo stowage"), 1),
    "architecture": (("african architecture", "modern architecture", "architectural styles"), 1),
}
_BENCHMARK_RULES = {"shingle_words": 10, "coverage": 0.5, "long_match_words": 20,
                    "near_shingle_words": 5, "near_max_words": 64, "near_coverage": 0.8,
                    "near_longest_words": 8, "minimum_benchmark_words": 5,
                    "shingle_hash": "blake2b-128-null-separated-utf8"}
_RULES = {"version": RULE_VERSION, "normalization": "NFKC-casefold-unicode-alphanumeric-v1",
          "minimum_words": _MIN_WORDS, "topic_words": _TOPIC_WORDS,
          "generated_topic_exclusions": _TOPIC_EXCLUSIONS,
          "formats": _FORMATS, "format_prefix": "textbook",
          "domains": _DOMAINS, "weak_anchors": sorted(_WEAK), "exclusions": _EXCLUSIONS,
          "minimum_domain_anchors": 2, "benchmark": _BENCHMARK_RULES}


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


RULE_DIGEST = _digest(_RULES)


def normalized_words(text: str) -> tuple[str, ...]:
    """Normalize case, Unicode compatibility characters, punctuation and spacing."""
    return tuple(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def _shingles(words: tuple[str, ...], size: int):
    for start in range(len(words) - size + 1):
        yield hashlib.blake2b("\0".join(words[start:start + size]).encode("utf-8"),
                              digest_size=16).digest()


def _contains(normalized: str, phrase: str) -> bool:
    return f" {phrase} " in normalized


@dataclass(frozen=True)
class BenchmarkMatch:
    benchmark_id: str
    field: str
    method: str
    coverage: float


class BenchmarkOverlapBlocker:
    """Block local exact/near question copies using caller-supplied benchmark text.

    Documents may be strings or mappings with ``id`` and ``text``. Supply questions
    separately from question-plus-options documents to cover short questions too.
    This lexical blocker cannot establish that paraphrased contamination is absent.
    """

    def __init__(self, documents: Iterable[str | Mapping[str, object]]):
        self._documents: dict[str, tuple[str, ...]] = {}
        self._index: dict[bytes, set[str]] = defaultdict(set)
        self._near_index: dict[bytes, set[str]] = defaultdict(set)
        for index, document in enumerate(documents):
            if isinstance(document, str):
                identity, text = str(index), document
            elif isinstance(document, Mapping):
                identity, text = str(document.get("id", index)), document.get("text")
            else:
                raise TypeError("Benchmark documents must be strings or mappings")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Benchmark documents require nonempty text")
            if identity in self._documents:
                raise ValueError(f"Duplicate benchmark id: {identity}")
            words = normalized_words(text)
            self._documents[identity] = words
            if len(words) < _BENCHMARK_RULES["minimum_benchmark_words"]:
                continue
            for shingle in _shingles(words, 10):
                self._index[shingle].add(identity)
            if len(words) <= _BENCHMARK_RULES["near_max_words"]:
                for shingle in _shingles(words, 5):
                    self._near_index[shingle].add(identity)
        if not self._index and not self._near_index:
            raise ValueError("Benchmark corpus requires at least one document with five normalized words")
        self.digest = _digest({"rules": _BENCHMARK_RULES,
                               "documents": sorted(self._documents.items())})

    def match(self, *, prompt: str = "", text: str = "") -> BenchmarkMatch | None:
        for field, content in (("prompt", prompt), ("text", text)):
            words = normalized_words(content)
            exact, near = set(), set()
            for shingle in _shingles(words, 10):
                exact.update(self._index.get(shingle, ()))
            for shingle in _shingles(words, 5):
                near.update(self._near_index.get(shingle, ()))
            for identity in sorted(exact | near):
                benchmark = self._documents[identity]
                blocks = SequenceMatcher(None, benchmark, words, autojunk=False).get_matching_blocks()
                coverage = sum(block.size for block in blocks) / len(benchmark)
                longest = max(block.size for block in blocks)
                if identity in exact and longest >= 10 and (coverage >= _BENCHMARK_RULES["coverage"]
                                          or longest >= _BENCHMARK_RULES["long_match_words"]):
                    return BenchmarkMatch(identity, field, "ten_word_overlap", round(coverage, 6))
                if identity in near and coverage >= _BENCHMARK_RULES["near_coverage"] and longest >= min(
                        _BENCHMARK_RULES["near_longest_words"], len(benchmark)):
                    return BenchmarkMatch(identity, field, "near_question", round(coverage, 6))
        return None


@dataclass(frozen=True)
class ScienceDecision:
    accepted: bool
    reason: str
    domains: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    rule_version: str = RULE_VERSION
    rule_digest: str = RULE_DIGEST
    benchmark_digest: str | None = None
    benchmark_match: BenchmarkMatch | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classify_cosmopedia_science(record: Mapping[str, object],
                               benchmark_blocker: BenchmarkOverlapBlocker | None = None) -> ScienceDecision:
    """Return stable admission/rejection metadata without modifying the source row."""
    benchmark_digest = benchmark_blocker.digest if benchmark_blocker is not None else None

    def decision(accepted: bool, reason: str, **metadata) -> ScienceDecision:
        return ScienceDecision(accepted, reason, benchmark_digest=benchmark_digest, **metadata)

    text, prompt = record.get("text"), record.get("prompt", "")
    if not isinstance(text, str) or not text.strip():
        return decision(False, "empty_text")
    if not isinstance(prompt, str):
        return decision(False, "invalid_prompt")
    if benchmark_blocker is not None:
        match = benchmark_blocker.match(prompt=prompt, text=text)
        if match is not None:
            return decision(False, "benchmark_overlap", benchmark_match=match)
    format_value = record.get("format", "")
    if not isinstance(format_value, str):
        return decision(False, "unsupported_format")
    format_name = format_value.casefold().strip().replace("-", "_").replace(" ", "_")
    if not (format_name.startswith("textbook") or format_name in _FORMATS):
        return decision(False, "unsupported_format")
    words = normalized_words(text)
    if len(words) < _MIN_WORDS:
        return decision(False, "too_short")
    combined = " " + " ".join(normalized_words(prompt + "\n" + text)) + " "
    # Do not reject the biological animal "python" or mathematical "functions".
    # Use subject phrases rather than those ambiguous single words.
    if "```" in text:
        return decision(False, "excluded_coding")
    for category, (phrases, threshold) in _EXCLUSIONS.items():
        # Count singular/plural alternatives together for context exclusions.
        hits = {"business" if phrase in ("business", "businesses") else phrase.removesuffix("s")
                for phrase in phrases if _contains(combined, phrase)}
        if len(hits) >= threshold:
            return decision(False, "excluded_" + category)
    # Natural-science terms in later analogies do not turn a chapter about
    # relationships, exercise instruction or another excluded topic into science.
    # Keep this check on the generated opening so a legitimate science chapter
    # can still discuss these applications later without an automatic rejection.
    generated_topic = " " + " ".join(words[:_TOPIC_WORDS]) + " "
    for category, (phrases, threshold) in _TOPIC_EXCLUSIONS.items():
        if sum(_contains(generated_topic, phrase) for phrase in phrases) >= threshold:
            return decision(False, "excluded_topic_" + category)
    generated = " " + " ".join(words) + " "
    domains, evidence = [], set()
    for domain, anchors in _DOMAINS.items():
        hits = {alternatives[0] for alternatives in anchors
                if any(_contains(generated, phrase) for phrase in alternatives)}
        if len(hits) >= 2 and hits - _WEAK:
            domains.append(domain)
            evidence.update(hits)
    if not domains:
        return decision(False, "insufficient_science_evidence")
    return decision(True, "accepted_science", domains=tuple(domains), evidence=tuple(sorted(evidence)))
