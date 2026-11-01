#!/usr/bin/env python3
"""Build a representative tokenizer corpus and train a 64K ByteLevel BPE tokenizer.

Default paths are tuned for the user's machine, but all environment-specific paths
and resource knobs are CLI-overridable. The algorithmic contract (mixture logic,
ByteLevel BPE, special-token order, code/support classification) is intentionally
frozen in code for reproducibility.

The stage format is a simple length-prefixed UTF-8 record stream (*.rec). This keeps
newlines and whitespace byte-exact while allowing fast sequential NVMe reads during
BPE training.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import struct
import subprocess
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# Frozen training contract
# -----------------------------------------------------------------------------

SPECIAL_TOKENS: List[str] = [
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|repo_start|>",
    "<|repo_end|>",
    "<|file_start|>",
    "<|file_end|>",
    "<|document_start|>",
    "<|document_end|>",
]

SCRIPT_REVISION = "2026-08-30-local-repair-v3"
TOTAL_VOCAB_DEFAULT = 65_536
DEFAULT_STACK_GENERATION = "990b4288-3824-41ac-94a0-b6fd6fa23ffe"
DEFAULT_SEED = 20260829

# Tokenizer corpus mixture. This is intentionally NOT the final model-pretraining
# mixture: tokenizer training needs symbol and syntax coverage, so code is upweighted.
TOP_MIXTURE = OrderedDict([
    ("general", 0.38),
    ("math", 0.23),
    ("code", 0.24),
    ("support", 0.06),
    ("academic", 0.06),
    ("wiki", 0.03),
])

CODE_LANG_SHARES = OrderedDict([
    ("Python", 0.22),
    ("C++", 0.20),
    ("JavaScript", 0.18),
    ("Java", 0.16),
    ("C", 0.14),
    ("TypeScript", 0.10),
])

# Explicit names are part of the stage-format contract. Never derive these with
# slug(): slug("C++") == slug("C") == "c" and silently corrupts the mixture.
CODE_BUCKET_NAMES = OrderedDict([
    ("Python", "code_python"),
    ("C++", "code_cpp"),
    ("JavaScript", "code_javascript"),
    ("Java", "code_java"),
    ("C", "code_c"),
    ("TypeScript", "code_typescript"),
])
if len(set(CODE_BUCKET_NAMES.values())) != len(CODE_BUCKET_NAMES):
    raise RuntimeError("CODE_BUCKET_NAMES must be one-to-one")

SUPPORT_SHARES = OrderedDict([
    ("markdown", 0.25),
    ("html", 0.17),
    ("json", 0.14),
    ("shell", 0.12),
    ("yaml_toml", 0.09),
    ("frontend", 0.09),
    ("build", 0.09),
    ("other_config", 0.05),
])

MATH_SOURCE_SHARES_DEFAULT = OrderedDict([
    ("nemotron", 0.50),
    ("finemath", 0.50),
])

DEFAULT_RAW_ROOT = Path("/home/zjr1236/data/raw")
DEFAULT_STACK_ROOT = Path("/mnt/D/data/stack-v3")
DEFAULT_NEMOTRON_ROOT = Path("/mnt/D/data/Nemotron-CC-Math-v1/4plus_MIND")
DEFAULT_STAGE_DIR = Path("/home/zjr1236/data/tokenizer_stage_24g")
DEFAULT_OUTPUT_DIR = Path("/home/zjr1236/data/tokenizer_64k")

TEXT_COLUMN_CANDIDATES = (
    "text", "content", "raw_content", "document", "body", "markdown", "article"
)
MATH_FIELDS = (
    "text", "problem", "question", "prompt", "solution", "answer", "response",
    "reasoning", "explanation", "proof"
)

ANCHOR_LANGUAGES = set(CODE_LANG_SHARES.keys())

SUPPORT_LANGUAGE_BUCKETS = {
    "markdown": {
        "Markdown", "MDX", "reStructuredText", "AsciiDoc", "RMarkdown", "Org",
    },
    "html": {
        "HTML", "HTML+ERB", "HTML+PHP", "HTML+Razor", "HTML+EEX", "HTML+ECR",
        "Java Server Pages", "ASP.NET", "Classic ASP",
    },
    "json": {"JSON", "JSON with Comments", "JSON5", "Jsonnet"},
    "shell": {"Shell", "fish", "Nushell", "PowerShell", "Batchfile"},
    "yaml_toml": {"YAML", "TOML", "INI", "Dotenv", "HOCON"},
    "frontend": {
        "TSX", "CSS", "SCSS", "Sass", "Less", "Vue", "Svelte", "Astro",
        "PostCSS", "Stylus",
    },
    "build": {
        "Makefile", "CMake", "Maven POM", "Gradle", "Gradle Kotlin DSL", "Dockerfile",
        "Ninja", "Meson", "Ant Build System", "BitBake", "Cabal Config", "Go Module",
        "Pip Requirements", "RPM Spec", "Starlark",
    },
    "other_config": {
        "EditorConfig", "Ignore List", "Git Config", "Git Attributes", "NPM Config",
        "Browserslist", "Option List", "ApacheConf", "Nginx", "SSH Config",
        "Readline Config", "ShellCheck Config", "Terraform Template", "Caddyfile",
    },
}

BAD_PATH_PARTS = (
    "/node_modules/", "/vendor/", "/.venv/", "/venv/", "/dist/", "/build/",
    "/target/", "/__pycache__/", "/.git/", "/coverage/",
)
BAD_BASENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "cargo.lock", "poetry.lock", "pipfile.lock",
}
BAD_SUFFIXES = (".min.js", ".min.css", ".map")

# -----------------------------------------------------------------------------
# Pure helpers (unit-tested)
# -----------------------------------------------------------------------------

def allocate_bytes(total: int, shares: Mapping[str, float]) -> Dict[str, int]:
    """Allocate exactly *total* integer bytes using largest-remainder rounding."""
    if total < 0:
        raise ValueError("total must be >= 0")
    s = float(sum(shares.values()))
    if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"shares must sum to 1.0, got {s}")
    raw = {k: total * float(v) for k, v in shares.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    left = total - sum(out.values())
    order = sorted(shares.keys(), key=lambda k: (-(raw[k] - out[k]), list(shares.keys()).index(k)))
    for k in order[:left]:
        out[k] += 1
    return out


def chunk_text_utf8(text: str, max_bytes: int) -> Iterator[str]:
    """Split text into UTF-8-safe chunks <= max_bytes, preserving exact concatenation."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if not text:
        return
    b = text.encode("utf-8")
    start = 0
    n = len(b)
    while start < n:
        end = min(n, start + max_bytes)
        if end < n:
            # Back up to a UTF-8 boundary.
            while end > start and (b[end] & 0xC0) == 0x80:
                end -= 1
            if end == start:  # Defensive; max_bytes might be smaller than one codepoint.
                end = min(n, start + max_bytes)
                while end < n and (b[end] & 0xC0) == 0x80:
                    end += 1
            # Prefer a nearby newline without losing bytes.
            window_start = max(start, end - min(4096, max_bytes // 2))
            nl = b.rfind(b"\n", window_start, end)
            if nl >= start:
                end = nl + 1
        piece = b[start:end].decode("utf-8")
        yield piece
        start = end


def stack_generation_matches(name: str, generation: str) -> bool:
    return f"-{generation}-c000.snappy.parquet" in Path(name).name


def classify_support(language: Optional[str], file_path: Optional[str]) -> Optional[str]:
    language = language or ""
    p = (file_path or "").replace("\\", "/")
    base = p.rsplit("/", 1)[-1].lower()

    # File-name semantics override generic language labels for build/config files.
    if base.startswith("readme"):
        return "markdown"
    if base in {"dockerfile", "makefile", "cmakelists.txt", "meson.build", "build.gradle",
                "build.gradle.kts", "pom.xml", "go.mod", "go.sum", "pyproject.toml",
                "requirements.txt", "setup.cfg", "package.json", "tsconfig.json"}:
        if base in {"package.json", "tsconfig.json"}:
            return "json"
        if base == "pyproject.toml":
            return "yaml_toml"
        return "build"

    for bucket, langs in SUPPORT_LANGUAGE_BUCKETS.items():
        if language in langs:
            return bucket
    return None


class RecordWriter:
    """Length-prefixed UTF-8 records: uint32 little-endian length + payload."""
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("wb")
        self.records = 0
        self.text_bytes = 0

    def write(self, text: str) -> int:
        b = text.encode("utf-8")
        if len(b) > 0xFFFFFFFF:
            raise ValueError("record exceeds uint32 length")
        self.f.write(struct.pack("<I", len(b)))
        self.f.write(b)
        self.records += 1
        self.text_bytes += len(b)
        return len(b)

    def close(self) -> None:
        if not self.f.closed:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def iter_record_batches(paths: Sequence[os.PathLike | str], batch_size: int = 2048,
                        max_text_bytes_by_path: Optional[Mapping[str, int]] = None) -> Iterator[List[str]]:
    batch: List[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        limit = None
        if max_text_bytes_by_path is not None:
            limit = max_text_bytes_by_path.get(str(path))
        seen = 0
        with path.open("rb") as f:
            while True:
                h = f.read(4)
                if not h:
                    break
                if len(h) != 4:
                    raise IOError(f"truncated record header: {path}")
                (n,) = struct.unpack("<I", h)
                payload = f.read(n)
                if len(payload) != n:
                    raise IOError(f"truncated record payload: {path}")
                if limit is not None and seen + n > limit:
                    break
                seen += n
                batch.append(payload.decode("utf-8"))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch

# -----------------------------------------------------------------------------
# Source discovery and readers
# -----------------------------------------------------------------------------

def _import_pyarrow():
    try:
        import pyarrow as pa  # noqa
        import pyarrow.parquet as pq
        return pa, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from e


def _import_zstandard():
    try:
        import zstandard as zstd
        return zstd
    except Exception as e:
        raise RuntimeError("zstandard is required for .zst sources: pip install zstandard") from e


def discover_dir(explicit: Optional[Path], raw_root: Path, candidates: Sequence[str], label: str) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{label} root does not exist: {p}")
        return p
    for name in candidates:
        p = raw_root / name
        if p.exists():
            return p.resolve()
    # Fallback to a case-insensitive shallow search.
    lowered = [x.lower().replace("_", "-") for x in candidates]
    for p in raw_root.iterdir() if raw_root.exists() else []:
        norm = p.name.lower().replace("_", "-")
        if any(c in norm or norm in c for c in lowered):
            return p.resolve()
    available = ", ".join(sorted(p.name for p in raw_root.iterdir())) if raw_root.exists() else "<missing root>"
    raise FileNotFoundError(f"could not auto-discover {label} under {raw_root}; children: {available}")


_WIKI_SNAPSHOT_RE = re.compile(r"^(?P<date>\d{8})\.(?P<lang>[A-Za-z0-9_-]+)$")


def resolve_english_wiki_root(root: Path) -> Path:
    """Resolve a Wikipedia parent/root to one English YYYYMMDD.en snapshot.

    A direct non-English snapshot is rejected rather than silently walking to a
    sibling language. For a multilingual parent, the newest English snapshot is
    selected deterministically.
    """
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Wikipedia root does not exist: {root}")

    m = _WIKI_SNAPSHOT_RE.match(root.name)
    if m:
        if m.group("lang").lower() != "en":
            raise ValueError(
                f"Wikipedia root is a non-English snapshot ({root.name}); "
                "English Wikipedia is required. Point --wiki-root at YYYYMMDD.en."
            )
        return root

    candidates = []
    preferred = root / "snapshots" / "master"
    search_roots = [preferred] if preferred.is_dir() else [root]
    for base in search_roots:
        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            mm = _WIKI_SNAPSHOT_RE.match(p.name)
            if mm and mm.group("lang").lower() == "en":
                candidates.append((mm.group("date"), str(p), p.resolve()))
    if not candidates:
        raise FileNotFoundError(
            f"no English Wikipedia YYYYMMDD.en snapshot found under {root}. "
            "Use --wiki-root /path/to/20231101.en."
        )
    candidates.sort(reverse=True)
    return candidates[0][2]


def list_source_files(root: Path) -> List[Path]:
    exts = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst", ".zst", ".txt", ".txt.gz")
    out = []
    for p in root.rglob("*"):
        if p.is_file() and any(p.name.endswith(x) for x in exts):
            out.append(p)
    return sorted(out)


def _choose_parquet_fields(pf, source_kind: str) -> Tuple[str, ...]:
    schema = pf.schema_arrow
    names = set(schema.names)
    if source_kind == "math":
        fields = tuple(x for x in MATH_FIELDS if x in names)
        if fields:
            # Prefer a canonical text field if present, else concatenate math fields.
            if "text" in fields:
                return ("text",)
            return fields
    for x in TEXT_COLUMN_CANDIDATES:
        if x in names:
            return (x,)
    # Last resort: first scalar string-like top-level field.
    for field in schema:
        if str(field.type) in {"string", "large_string"}:
            return (field.name,)
    raise ValueError(f"no usable text column found; schema={schema}")


def _combine_row_fields(row: Mapping[str, object], fields: Sequence[str]) -> Optional[str]:
    parts = []
    for k in fields:
        v = row.get(k)
        if isinstance(v, str) and v:
            parts.append(v)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return "\n\n".join(parts)


def iter_parquet_texts(path: Path, source_kind: str, batch_size: int = 512) -> Iterator[str]:
    _, pq = _import_pyarrow()
    pf = pq.ParquetFile(path)
    fields = _choose_parquet_fields(pf, source_kind)
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(fields), use_threads=True):
        data = batch.to_pydict()
        n = batch.num_rows
        for i in range(n):
            row = {k: data[k][i] for k in fields}
            text = _combine_row_fields(row, fields)
            if text:
                yield text


def _json_text(obj: object, source_kind: str) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    fields = MATH_FIELDS if source_kind == "math" else TEXT_COLUMN_CANDIDATES
    vals = []
    for k in fields:
        v = obj.get(k)
        if isinstance(v, str) and v:
            vals.append(v)
            if k == "text":
                return v
    return "\n\n".join(vals) if vals else None


def iter_json_lines(stream: Iterable[str], source_kind: str) -> Iterator[str]:
    for line in stream:
        line = line.strip("\r\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            # Some .zst corpora are plain text line streams.
            if source_kind != "math":
                yield line
            continue
        text = _json_text(obj, source_kind)
        if text:
            yield text


def iter_texts_from_file(path: Path, source_kind: str) -> Iterator[str]:
    name = path.name
    if name.endswith(".parquet"):
        yield from iter_parquet_texts(path, source_kind)
        return
    if name.endswith(".jsonl"):
        with path.open("rt", encoding="utf-8", errors="strict") as f:
            yield from iter_json_lines(f, source_kind)
        return
    if name.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
            yield from iter_json_lines(f, source_kind)
        return
    if name.endswith(".txt"):
        with path.open("rt", encoding="utf-8", errors="strict") as f:
            # Whole file preserves line structure.
            yield f.read()
        return
    if name.endswith(".txt.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
            yield f.read()
        return
    if name.endswith(".zst"):
        zstd = _import_zstandard()
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8", errors="strict") as txt:
                    yield from iter_json_lines(txt, source_kind)
        return
    raise ValueError(f"unsupported source file: {path}")


def is_bad_code_path(path: str) -> bool:
    p = "/" + (path or "").replace("\\", "/").lower().lstrip("/")
    base = p.rsplit("/", 1)[-1]
    if any(x in p for x in BAD_PATH_PARTS):
        return True
    if base in BAD_BASENAMES:
        return True
    if any(base.endswith(x) for x in BAD_SUFFIXES):
        return True
    return False


def looks_minified(text: str) -> bool:
    if len(text) < 8192:
        return False
    nl = text.count("\n")
    if nl == 0:
        return len(text) > 16_384
    avg_line = len(text) / (nl + 1)
    return avg_line > 1200


def acceptable_text(text: str, min_bytes: int = 64) -> bool:
    if not isinstance(text, str) or not text:
        return False
    b = text.encode("utf-8", errors="strict")
    if len(b) < min_bytes:
        return False
    # Binary-ish text guard. Preserve Unicode and whitespace; only reject pathological NUL/control data.
    if b.count(b"\x00") > max(2, len(b) // 10000):
        return False
    return True

# -----------------------------------------------------------------------------
# Stage builder
# -----------------------------------------------------------------------------

@dataclass
class Bucket:
    name: str
    target_bytes: int
    path: Path
    writer: RecordWriter
    text_bytes: int = 0
    records: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.target_bytes - self.text_bytes)

    @property
    def done(self) -> bool:
        return self.text_bytes >= self.target_bytes

    def add_text(self, text: str, max_record_bytes: int) -> int:
        added = 0
        if self.done:
            return 0
        for chunk in chunk_text_utf8(text, max_record_bytes):
            if self.done:
                break
            b = chunk.encode("utf-8")
            if len(b) > self.remaining:
                # Fit the final piece exactly-ish without breaking UTF-8.
                fitted = next(chunk_text_utf8(chunk, self.remaining), "") if self.remaining > 0 else ""
                if not fitted:
                    break
                chunk = fitted
                b = chunk.encode("utf-8")
            if not b:
                continue
            self.writer.write(chunk)
            self.text_bytes += len(b)
            self.records += 1
            added += len(b)
        return added


def _make_buckets(stage_dir: Path, quotas: Mapping[str, int]) -> Dict[str, Bucket]:
    record_dir = stage_dir / "records"
    record_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, target in quotas.items():
        p = record_dir / f"{name}.rec"
        out[name] = Bucket(name, int(target), p, RecordWriter(p))
    return out


def _close_buckets(buckets: Mapping[str, Bucket]) -> None:
    for b in buckets.values():
        b.writer.close()


def _log_bucket_progress(label: str, buckets: Mapping[str, Bucket]) -> None:
    print(f"\n[{label}]")
    for name, b in buckets.items():
        pct = 100.0 * b.text_bytes / b.target_bytes if b.target_bytes else 100.0
        print(f"  {name:24s} {b.text_bytes/2**30:7.3f}/{b.target_bytes/2**30:7.3f} GiB  {pct:6.2f}%  {b.records:,} records")


def sample_generic_source(root: Path, bucket: Bucket, source_kind: str, rng: random.Random,
                          max_record_bytes: int, max_files: Optional[int] = None) -> List[str]:
    files = list_source_files(root)
    if not files:
        raise FileNotFoundError(f"no supported files found under {root}")
    rng.shuffle(files)
    selected = []
    for idx, path in enumerate(files):
        if bucket.done:
            break
        if max_files is not None and idx >= max_files:
            break
        before = bucket.text_bytes
        try:
            for text in iter_texts_from_file(path, source_kind):
                if bucket.done:
                    break
                if acceptable_text(text):
                    bucket.add_text(text, max_record_bytes)
        except Exception as e:
            print(f"WARN: skip source file {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if bucket.text_bytes > before:
            selected.append(str(path))
            print(f"  {bucket.name}: {bucket.text_bytes/2**30:.3f}/{bucket.target_bytes/2**30:.3f} GiB  <- {path.name}")
    return selected


def _find_stack_parquets(root: Path, generation: str) -> List[Path]:
    out = [p for p in root.rglob("*.parquet") if stack_generation_matches(p.name, generation)]
    if not out:
        # If old generations were already removed and names differ, fail rather than silently mix snapshots.
        gens = sorted({p.name for p in root.rglob("part-*.parquet")})[:5]
        raise FileNotFoundError(
            f"no Stack-v3 parquet matched generation {generation} under {root}. "
            f"Use --stack-generation to override. Example files: {gens}"
        )
    return sorted(out)


def sample_stack(root: Path, generation: str, code_buckets: Mapping[str, Bucket],
                 support_buckets: Mapping[str, Bucket], rng: random.Random,
                 max_code_file_bytes: int, max_support_per_repo_bytes: int,
                 max_record_bytes: int, max_accept_per_shard_bytes: int) -> List[str]:
    _, pq = _import_pyarrow()
    paths = _find_stack_parquets(root, generation)
    rng.shuffle(paths)
    selected = []

    def all_done() -> bool:
        return all(b.done for b in code_buckets.values()) and all(b.done for b in support_buckets.values())

    columns = [
        "repo_path",
        "files.list.element.content",
        "files.list.element.size_bytes",
        "files.list.element.file_path",
        "files.list.element.language",
        "files.list.element.is_vendor",
    ]

    for shard_no, path in enumerate(paths, 1):
        if all_done():
            break
        pf = pq.ParquetFile(path)
        accepted_shard = 0
        touched = False
        print(f"  Stack [{shard_no}/{len(paths)}] {path.name}")
        for batch in pf.iter_batches(batch_size=64, columns=columns, use_threads=True):
            if all_done() or accepted_shard >= max_accept_per_shard_bytes:
                break
            rows = batch.to_pylist()
            rng.shuffle(rows)
            for row in rows:
                if all_done() or accepted_shard >= max_accept_per_shard_bytes:
                    break
                support_used = 0
                files = row.get("files") or []
                rng.shuffle(files)
                for f in files:
                    if all_done() or accepted_shard >= max_accept_per_shard_bytes:
                        break
                    if not isinstance(f, dict) or f.get("is_vendor"):
                        continue
                    lang = f.get("language")
                    file_path = f.get("file_path") or ""
                    text = f.get("content")
                    size = f.get("size_bytes")
                    if not isinstance(text, str) or not acceptable_text(text):
                        continue
                    if is_bad_code_path(file_path):
                        continue
                    actual_size = len(text.encode("utf-8"))
                    if isinstance(size, int) and size > 0:
                        actual_size = min(actual_size, size)

                    bucket = None
                    if lang in code_buckets:
                        if actual_size > max_code_file_bytes or looks_minified(text):
                            continue
                        if not code_buckets[lang].done:
                            bucket = code_buckets[lang]
                    else:
                        cat = classify_support(lang, file_path)
                        if cat and cat in support_buckets and not support_buckets[cat].done:
                            if support_used >= max_support_per_repo_bytes:
                                continue
                            if actual_size > max_code_file_bytes or looks_minified(text):
                                continue
                            bucket = support_buckets[cat]

                    if bucket is None:
                        continue
                    before = bucket.text_bytes
                    added = bucket.add_text(text, max_record_bytes)
                    delta = bucket.text_bytes - before
                    if delta:
                        accepted_shard += delta
                        touched = True
                        if bucket.name.startswith("support_"):
                            support_used += delta
        if touched:
            selected.append(str(path))
        _log_bucket_progress("Stack progress", {**code_buckets, **support_buckets})
    if not all_done():
        missing = {k: b.remaining for k, b in {**code_buckets, **support_buckets}.items() if not b.done}
        raise RuntimeError(f"Stack-v3 sample exhausted before quotas were filled: {missing}")
    return selected


def _source_roots(args) -> Dict[str, Path]:
    raw = args.raw_root.expanduser().resolve()
    return {
        "fineweb": discover_dir(args.fineweb_root, raw,
                                ["fineweb-edu", "fineweb_edu", "FineWeb-Edu", "fineweb"], "FineWeb-Edu"),
        "finemath": discover_dir(args.finemath_root, raw,
                                 ["finemath", "FineMath", "fine-math"], "FineMath"),
        "pes2o": discover_dir(args.pes2o_root, raw,
                              ["pes2o", "peS2o", "pes2o-v3"], "peS2o"),
        "wiki": resolve_english_wiki_root(discover_dir(
            args.wiki_root, raw, ["wikipedia", "wiki", "wikipedia-en"], "Wikipedia"
        )),
        "nemotron": args.nemotron_root.expanduser().resolve(),
        "stack": args.stack_root.expanduser().resolve(),
    }


def build_stage(args) -> Path:
    total_bytes = int(args.corpus_gib * (2**30))
    top = allocate_bytes(total_bytes, TOP_MIXTURE)
    code_q = allocate_bytes(top["code"], CODE_LANG_SHARES)
    support_q = allocate_bytes(top["support"], SUPPORT_SHARES)
    math_shares = OrderedDict([
        ("nemotron", args.math_nemotron_share),
        ("finemath", 1.0 - args.math_nemotron_share),
    ])
    math_q = allocate_bytes(top["math"], math_shares)

    stage = args.stage_dir.expanduser().resolve()
    stage.mkdir(parents=True, exist_ok=True)
    manifest_path = stage / "manifest.json"
    if manifest_path.exists() and not args.rebuild_stage:
        existing = json.loads(manifest_path.read_text())
        expected = {
            "mode": args.mode,
            "corpus_gib": args.corpus_gib,
            "seed": args.seed,
            "stack_generation": args.stack_generation,
        }
        actual = {k: existing.get(k) for k in expected}
        if actual != expected:
            raise RuntimeError(
                f"existing stage manifest does not match requested build. actual={actual}, expected={expected}. "
                "Use --rebuild-stage to replace it."
            )
        print(f"Stage already exists and matches: {manifest_path}")
        return manifest_path

    if args.rebuild_stage and (stage / "records").exists():
        import shutil
        shutil.rmtree(stage / "records")

    roots = _source_roots(args)
    for k, p in roots.items():
        if not p.exists():
            raise FileNotFoundError(f"{k} root missing: {p}")

    quotas: Dict[str, int] = {
        "general": top["general"],
        "academic": top["academic"],
        "wiki": top["wiki"],
        "math_nemotron": math_q["nemotron"],
        "math_finemath": math_q["finemath"],
    }
    for lang, q in code_q.items():
        quotas[CODE_BUCKET_NAMES[lang]] = q
    if sum(quotas.values()) + sum(support_q.values()) != total_bytes:
        raise RuntimeError("internal quota allocation error before support buckets")
    for name, q in support_q.items():
        quotas[f"support_{name}"] = q

    buckets = _make_buckets(stage, quotas)
    max_record_bytes = int(args.max_record_mib * (2**20))
    rng = random.Random(args.seed)
    selected_inputs: Dict[str, List[str]] = {}

    try:
        print("\n=== Building general / academic / wiki / math stage ===")
        selected_inputs["general"] = sample_generic_source(
            roots["fineweb"], buckets["general"], "general", random.Random(rng.randrange(2**63)), max_record_bytes)
        selected_inputs["academic"] = sample_generic_source(
            roots["pes2o"], buckets["academic"], "general", random.Random(rng.randrange(2**63)), max_record_bytes)
        selected_inputs["wiki"] = sample_generic_source(
            roots["wiki"], buckets["wiki"], "general", random.Random(rng.randrange(2**63)), max_record_bytes)
        selected_inputs["math_nemotron"] = sample_generic_source(
            roots["nemotron"], buckets["math_nemotron"], "math", random.Random(rng.randrange(2**63)), max_record_bytes)
        selected_inputs["math_finemath"] = sample_generic_source(
            roots["finemath"], buckets["math_finemath"], "math", random.Random(rng.randrange(2**63)), max_record_bytes)

        print("\n=== Building Stack-v3 code + repo support stage ===")
        code_buckets = {lang: buckets[CODE_BUCKET_NAMES[lang]] for lang in CODE_LANG_SHARES}
        support_buckets = {name: buckets[f"support_{name}"] for name in SUPPORT_SHARES}
        selected_inputs["stack"] = sample_stack(
            roots["stack"], args.stack_generation, code_buckets, support_buckets,
            random.Random(rng.randrange(2**63)),
            max_code_file_bytes=int(args.max_code_file_mib * (2**20)),
            max_support_per_repo_bytes=int(args.max_support_per_repo_mib * (2**20)),
            max_record_bytes=max_record_bytes,
            max_accept_per_shard_bytes=int(args.stack_max_accept_per_shard_mib * (2**20)),
        )
    finally:
        _close_buckets(buckets)

    missing = {k: b.remaining for k, b in buckets.items() if not b.done}
    if missing and not args.allow_short:
        raise RuntimeError(f"stage build ended short of quotas: {missing}")

    manifest = {
        "format_version": 2,
        "mode": args.mode,
        "corpus_gib": args.corpus_gib,
        "total_target_bytes": total_bytes,
        "seed": args.seed,
        "stack_generation": args.stack_generation,
        "vocab_size": args.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "top_mixture": TOP_MIXTURE,
        "code_lang_shares": CODE_LANG_SHARES,
        "support_shares": SUPPORT_SHARES,
        "math_nemotron_share": args.math_nemotron_share,
        "roots": {k: str(v) for k, v in roots.items()},
        "buckets": {
            k: {
                "path": str(b.path),
                "target_bytes": b.target_bytes,
                "text_bytes": b.text_bytes,
                "records": b.records,
            } for k, b in buckets.items()
        },
        "selected_inputs": selected_inputs,
        "created_at_unix": time.time(),
    }
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    os.replace(tmp, manifest_path)
    _log_bucket_progress("Final stage", buckets)
    print(f"\nStage manifest: {manifest_path}")
    return manifest_path


def _atomic_write_json(path: Path, obj: Mapping) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _manifest_text_total(manifest: Mapping) -> int:
    return sum(int(x.get("text_bytes", 0)) for x in manifest.get("buckets", {}).values())


def _replace_files_and_manifest(
    replacements: Sequence[Tuple[Path, Path]], manifest_path: Path, new_manifest: Mapping
) -> None:
    """Replace staged record files and manifest with rollback on ordinary exceptions."""
    backups: List[Tuple[Path, Path]] = []
    finals_without_backup: List[Path] = []
    token = f"repair-bak-{os.getpid()}-{time.time_ns()}"
    try:
        for _tmp, final in replacements:
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                backup = final.with_name(final.name + "." + token)
                os.replace(final, backup)
                backups.append((final, backup))
            else:
                finals_without_backup.append(final)
        for tmp, final in replacements:
            os.replace(tmp, final)
        _atomic_write_json(manifest_path, new_manifest)
    except Exception:
        for _tmp, final in replacements:
            if final.exists():
                final.unlink()
        for final, backup in backups:
            if backup.exists():
                os.replace(backup, final)
        raise
    else:
        for _final, backup in backups:
            with contextlib.suppress(FileNotFoundError):
                backup.unlink()


def _load_stage_manifest(stage_dir: Path) -> Tuple[Path, dict]:
    stage = stage_dir.expanduser().resolve()
    manifest_path = stage / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"stage manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if "buckets" not in manifest or "total_target_bytes" not in manifest:
        raise RuntimeError(f"invalid stage manifest: {manifest_path}")
    return manifest_path, manifest


def repair_wiki_stage(args) -> Path:
    """Rebuild only wiki.rec from a strictly English Wikipedia snapshot."""
    manifest_path, manifest = _load_stage_manifest(args.stage_dir)
    wiki_meta = manifest.get("buckets", {}).get("wiki")
    if not wiki_meta:
        raise RuntimeError("stage manifest has no wiki bucket")

    if args.wiki_root is None:
        raise ValueError("repair-wiki requires --wiki-root pointing at English Wikipedia")
    wiki_root = resolve_english_wiki_root(args.wiki_root)
    target = int(wiki_meta["target_bytes"])
    if target <= 0:
        raise RuntimeError(f"invalid wiki target_bytes={target}")

    final_path = Path(wiki_meta.get("path") or (args.stage_dir / "records" / "wiki.rec")).expanduser().resolve()
    temp_path = final_path.with_name(final_path.name + f".repair-wiki.tmp.{os.getpid()}")
    with contextlib.suppress(FileNotFoundError):
        temp_path.unlink()

    bucket = Bucket("wiki", target, temp_path, RecordWriter(temp_path))
    print("\n=== Repairing only English Wikipedia tokenizer bucket ===")
    print(f"  wiki root : {wiki_root}")
    print(f"  target    : {target/2**30:.6f} GiB")
    try:
        selected = sample_generic_source(
            wiki_root,
            bucket,
            "general",
            random.Random(int(getattr(args, "seed", manifest.get("seed", DEFAULT_SEED)))),
            int(args.max_record_mib * (2**20)),
        )
    finally:
        bucket.writer.close()

    if not bucket.done:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError(
            f"English Wikipedia exhausted before wiki quota: "
            f"{bucket.text_bytes}/{target} bytes"
        )
    if bucket.text_bytes != target:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError(f"wiki repair must be exact: {bucket.text_bytes} != {target}")
    if not selected:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError("wiki repair selected no English source files")

    before_total = _manifest_text_total(manifest)
    new_manifest = json.loads(json.dumps(manifest))
    new_manifest["format_version"] = max(2, int(new_manifest.get("format_version", 1)))
    new_manifest.setdefault("roots", {})["wiki"] = str(wiki_root)
    new_manifest.setdefault("selected_inputs", {})["wiki"] = [str(Path(x).resolve()) for x in selected]
    new_manifest["buckets"]["wiki"] = {
        "path": str(final_path),
        "target_bytes": target,
        "text_bytes": bucket.text_bytes,
        "records": bucket.records,
    }
    new_manifest.setdefault("repairs", []).append({
        "kind": "wiki",
        "script_revision": SCRIPT_REVISION,
        "source_root": str(wiki_root),
        "created_at_unix": time.time(),
    })
    new_manifest["updated_at_unix"] = time.time()
    after_total = _manifest_text_total(new_manifest)
    if after_total != before_total:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError(f"wiki repair changed total staged bytes: {before_total} -> {after_total}")

    _replace_files_and_manifest([(temp_path, final_path)], manifest_path, new_manifest)
    expected_total = int(new_manifest["total_target_bytes"])
    print("\nLocal Wikipedia repair complete:")
    print(f"  source   : {wiki_root}")
    print(f"  wiki     : {target/2**30:.6f} GiB")
    if after_total == expected_total:
        print(f"  stage    : {after_total/2**30:.6f} GiB exact")
    else:
        print(
            f"  stage    : {after_total/2**30:.6f} GiB; manifest target "
            f"{expected_total/2**30:.6f} GiB (run repair-code if code_cpp is still missing)"
        )
    print(f"  files    : {len(selected)} English Wiki source file(s)")
    return manifest_path


def _code_targets_from_manifest(manifest: Mapping) -> Dict[str, int]:
    total = int(manifest["total_target_bytes"])
    top_q = allocate_bytes(total, TOP_MIXTURE)
    lang_q = allocate_bytes(top_q["code"], CODE_LANG_SHARES)
    return {CODE_BUCKET_NAMES[lang]: int(q) for lang, q in lang_q.items()}


def repair_code_stage(args) -> Path:
    """Rebuild C and C++ buckets in one Stack pass, fixing the historical slug collision."""
    manifest_path, manifest = _load_stage_manifest(args.stage_dir)
    targets = _code_targets_from_manifest(manifest)
    target_c = targets["code_c"]
    target_cpp = targets["code_cpp"]

    stage = args.stage_dir.expanduser().resolve()
    record_dir = stage / "records"
    final_c = Path(manifest.get("buckets", {}).get("code_c", {}).get("path", record_dir / "code_c.rec")).expanduser().resolve()
    final_cpp = Path(manifest.get("buckets", {}).get("code_cpp", {}).get("path", record_dir / "code_cpp.rec")).expanduser().resolve()
    tmp_c = final_c.with_name(final_c.name + f".repair-code.tmp.{os.getpid()}")
    tmp_cpp = final_cpp.with_name(final_cpp.name + f".repair-code.tmp.{os.getpid()}")
    for p in (tmp_c, tmp_cpp):
        with contextlib.suppress(FileNotFoundError):
            p.unlink()

    buckets = {
        "C": Bucket("code_c", target_c, tmp_c, RecordWriter(tmp_c)),
        "C++": Bucket("code_cpp", target_cpp, tmp_cpp, RecordWriter(tmp_cpp)),
    }
    print("\n=== Repairing only C + C++ tokenizer buckets ===")
    stack_root = args.stack_root.expanduser().resolve()
    generation = getattr(args, "stack_generation", None) or manifest.get("stack_generation") or DEFAULT_STACK_GENERATION
    try:
        selected = sample_stack(
            stack_root,
            generation,
            buckets,
            {},
            random.Random(int(getattr(args, "seed", manifest.get("seed", DEFAULT_SEED)))),
            max_code_file_bytes=int(args.max_code_file_mib * (2**20)),
            max_support_per_repo_bytes=int(args.max_support_per_repo_mib * (2**20)),
            max_record_bytes=int(args.max_record_mib * (2**20)),
            max_accept_per_shard_bytes=int(args.stack_max_accept_per_shard_mib * (2**20)),
        )
    finally:
        _close_buckets(buckets)

    if any(not b.done or b.text_bytes != b.target_bytes for b in buckets.values()):
        for p in (tmp_c, tmp_cpp):
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
        raise RuntimeError("C/C++ repair ended short of exact quotas")

    new_manifest = json.loads(json.dumps(manifest))
    new_manifest["format_version"] = max(2, int(new_manifest.get("format_version", 1)))
    new_manifest.setdefault("roots", {})["stack"] = str(stack_root)
    new_manifest.setdefault("selected_inputs", {})["stack_repair_code"] = [str(Path(x).resolve()) for x in selected]
    for name, b, final in (
        ("code_c", buckets["C"], final_c),
        ("code_cpp", buckets["C++"], final_cpp),
    ):
        new_manifest["buckets"][name] = {
            "path": str(final), "target_bytes": b.target_bytes,
            "text_bytes": b.text_bytes, "records": b.records,
        }
    new_manifest.setdefault("repairs", []).append({
        "kind": "code_c_cpp",
        "script_revision": SCRIPT_REVISION,
        "created_at_unix": time.time(),
    })
    new_manifest["updated_at_unix"] = time.time()
    total = _manifest_text_total(new_manifest)
    expected = int(new_manifest["total_target_bytes"])
    if total != expected:
        for p in (tmp_c, tmp_cpp):
            with contextlib.suppress(FileNotFoundError):
                p.unlink()
        raise RuntimeError(f"C/C++ repair did not restore exact stage total: {total} != {expected}")

    _replace_files_and_manifest(
        [(tmp_c, final_c), (tmp_cpp, final_cpp)], manifest_path, new_manifest
    )
    print("\nLocal code repair complete:")
    print(f"  code_c   : {target_c/2**30:.6f} GiB")
    print(f"  code_cpp : {target_cpp/2**30:.6f} GiB")
    print(f"  stage    : {total/2**30:.6f} GiB exact")
    return manifest_path


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")

# -----------------------------------------------------------------------------
# BPE trainer + memory monitor
# -----------------------------------------------------------------------------

def _training_limits(manifest: dict, train_gib: Optional[float]) -> Tuple[List[str], Dict[str, int], int]:
    buckets = manifest["buckets"]
    total = sum(int(x["text_bytes"]) for x in buckets.values())
    if train_gib is None:
        frac = 1.0
    else:
        want = int(train_gib * (2**30))
        frac = min(1.0, want / total) if total else 1.0
    paths = []
    limits = {}
    records_est = 0
    for meta in buckets.values():
        p = str(Path(meta["path"]).resolve())
        paths.append(p)
        lim = int(meta["text_bytes"] * frac)
        limits[p] = lim
        records_est += int(meta["records"] * frac)
    return paths, limits, max(1, records_est)


def train_child(args) -> None:
    try:
        from tokenizers import Tokenizer, decoders
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.trainers import BpeTrainer
    except Exception as e:
        raise RuntimeError("tokenizers is required: pip install tokenizers") from e

    os.environ["RAYON_NUM_THREADS"] = str(args.threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    stage_manifest = args.stage_dir.expanduser().resolve() / "manifest.json"
    manifest = json.loads(stage_manifest.read_text())
    paths, limits, records_est = _training_limits(manifest, args.train_gib)

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token=SPECIAL_TOKENS[3]))
    tokenizer.normalizer = None
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        max_token_length=args.max_token_length,
    )

    print("=== BPE training ===")
    print(f"vocab_size      : {args.vocab_size}")
    print(f"special tokens  : {len(SPECIAL_TOKENS)}")
    print(f"threads         : {args.threads}")
    print(f"records est.    : {records_est:,}")
    print(f"train text GiB  : {sum(limits.values())/2**30:.3f}")
    t0 = time.time()
    tokenizer.train_from_iterator(
        iter_record_batches(paths, batch_size=args.trainer_batch_size, max_text_bytes_by_path=limits),
        trainer=trainer,
        length=records_est,
    )
    elapsed = time.time() - t0

    # Verify the reserved ID contract before saving.
    bad = []
    for expected_id, tok in enumerate(SPECIAL_TOKENS):
        got = tokenizer.token_to_id(tok)
        if got != expected_id:
            bad.append((tok, expected_id, got))
    if bad:
        raise RuntimeError(f"special token ID contract violated: {bad}")
    if tokenizer.get_vocab_size() != args.vocab_size:
        raise RuntimeError(f"vocab size mismatch: got {tokenizer.get_vocab_size()}, expected {args.vocab_size}")

    tok_path = output / "tokenizer.json"
    tokenizer.save(str(tok_path), pretty=True)
    tokenizer.model.save(str(output))

    special_map = {
        "pad_token": SPECIAL_TOKENS[0],
        "bos_token": SPECIAL_TOKENS[1],
        "eos_token": SPECIAL_TOKENS[2],
        "unk_token": SPECIAL_TOKENS[3],
        "additional_special_tokens": SPECIAL_TOKENS[4:],
    }
    (output / "special_tokens_map.json").write_text(json.dumps(special_map, indent=2))
    config = {
        **special_map,
        "model_max_length": 16384,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "clean_up_tokenization_spaces": False,
        "add_prefix_space": False,
        "vocab_size": args.vocab_size,
        "special_token_ids": {tok: i for i, tok in enumerate(SPECIAL_TOKENS)},
    }
    (output / "tokenizer_config.json").write_text(json.dumps(config, indent=2))
    report = {
        "elapsed_seconds": elapsed,
        "train_text_bytes": sum(limits.values()),
        "threads": args.threads,
        "vocab_size": tokenizer.get_vocab_size(),
        "special_token_ids": {tok: tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS},
    }
    (output / "train_report.json").write_text(json.dumps(report, indent=2))
    print(f"Training complete in {elapsed/60:.1f} min -> {tok_path}")


def _rss_tree_bytes(pid: int) -> int:
    try:
        import psutil
        p = psutil.Process(pid)
        total = p.memory_info().rss
        for c in p.children(recursive=True):
            with contextlib.suppress(Exception):
                total += c.memory_info().rss
        return total
    except Exception:
        # Linux fallback: direct process only.
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) * 1024
        except Exception:
            pass
        return 0


def train_monitored(args) -> None:
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--phase", "train-child",
        "--mode", args.mode,
        "--stage-dir", str(args.stage_dir),
        "--output-dir", str(args.output_dir),
        "--vocab-size", str(args.vocab_size),
        "--threads", str(args.threads),
        "--trainer-batch-size", str(args.trainer_batch_size),
        "--min-frequency", str(args.min_frequency),
        "--max-token-length", str(args.max_token_length),
        "--ram-soft-gib", str(args.ram_soft_gib),
        "--ram-hard-gib", str(args.ram_hard_gib),
    ]
    if args.train_gib is not None:
        cmd += ["--train-gib", str(args.train_gib)]
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(args.threads)
    env["TOKENIZERS_PARALLELISM"] = "true"
    p = subprocess.Popen(cmd, env=env)
    soft = int(args.ram_soft_gib * 2**30)
    hard = int(args.ram_hard_gib * 2**30)
    warned = False
    peak = 0
    try:
        while p.poll() is None:
            rss = _rss_tree_bytes(p.pid)
            peak = max(peak, rss)
            if rss >= soft and not warned:
                print(f"\nWARNING: tokenizer RSS reached {rss/2**30:.2f} GiB (soft limit {args.ram_soft_gib:.1f} GiB)", file=sys.stderr)
                warned = True
            if rss >= hard:
                print(f"\nERROR: tokenizer RSS reached {rss/2**30:.2f} GiB; hard limit is {args.ram_hard_gib:.1f} GiB. Terminating child.", file=sys.stderr)
                p.terminate()
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    p.kill()
                raise MemoryError(
                    "BPE training exceeded RAM hard limit. The 24 GiB staged corpus is still intact; "
                    "retry with --phase train --train-gib 16 (or 12) without rebuilding."
                )
            time.sleep(1.0)
    finally:
        with contextlib.suppress(Exception):
            if p.poll() is None:
                p.terminate()
    rc = p.wait()
    print(f"Peak tokenizer RSS: {peak/2**30:.2f} GiB")
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def validate_tokenizer(args) -> None:
    try:
        from tokenizers import Tokenizer
    except Exception as e:
        raise RuntimeError("tokenizers is required: pip install tokenizers") from e
    output = args.output_dir.expanduser().resolve()
    tok = Tokenizer.from_file(str(output / "tokenizer.json"))

    for expected_id, special in enumerate(SPECIAL_TOKENS):
        got = tok.token_to_id(special)
        if got != expected_id:
            raise AssertionError(f"special token {special}: expected {expected_id}, got {got}")
        ids = tok.encode(special, add_special_tokens=False).ids
        if ids != [expected_id]:
            raise AssertionError(f"special token not atomic: {special} -> {ids}")

    probes = [
        "hello world\n",
        "def f(x):\n    return x + 1\n",
        "#include <vector>\nint main(){return 0;}\n",
        "#!/usr/bin/env bash\nset -euo pipefail\necho \"${HOME}\"\n",
        "<div class=\"container\"><script type=\"module\"></script></div>\n",
        "\\frac{a}{b} + \\sqrt{x^2 + y^2}\n",
        "世界你好 αβγ ∑ ∫ →\n",
        '{"name":"example","enabled":true}\n',
    ]
    for text in probes:
        ids = tok.encode(text, add_special_tokens=False).ids
        dec = tok.decode(ids, skip_special_tokens=False)
        if dec != text:
            raise AssertionError(f"round-trip mismatch: {text!r} -> {dec!r}")

    report = {
        "vocab_size": tok.get_vocab_size(),
        "special_token_ids": {s: tok.token_to_id(s) for s in SPECIAL_TOKENS},
        "probes": [
            {"text": t, "bytes": len(t.encode("utf-8")), "tokens": len(tok.encode(t, add_special_tokens=False).ids)}
            for t in probes
        ],
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("Tokenizer validation: PASS")
    for x in report["probes"]:
        print(f"  {x['bytes']:4d} bytes / {x['tokens']:3d} tokens = {x['bytes']/max(1,x['tokens']):.2f} bytes/token")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def positive_float(s: str) -> float:
    v = float(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mode", choices=["final", "pilot"], default="final")
    p.add_argument(
        "--phase",
        choices=["all", "build", "repair-code", "repair-wiki", "train", "train-child", "validate"],
        default="all",
    )

    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    p.add_argument("--nemotron-root", type=Path, default=DEFAULT_NEMOTRON_ROOT)
    p.add_argument("--fineweb-root", type=Path, default=None)
    p.add_argument("--finemath-root", type=Path, default=None)
    p.add_argument("--pes2o-root", type=Path, default=None)
    p.add_argument("--wiki-root", type=Path, default=None)
    p.add_argument("--stack-generation", default=DEFAULT_STACK_GENERATION)

    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--corpus-gib", type=positive_float, default=None,
                   help="UTF-8 text GiB to stage; final defaults to 24, pilot to 2")
    p.add_argument("--train-gib", type=positive_float, default=None,
                   help="optionally train on a proportional subset of an existing larger stage")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--vocab-size", type=int, default=TOTAL_VOCAB_DEFAULT)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--trainer-batch-size", type=int, default=2048)
    p.add_argument("--min-frequency", type=int, default=2)
    p.add_argument("--max-token-length", type=int, default=64)

    p.add_argument("--ram-soft-gib", type=positive_float, default=52.0)
    p.add_argument("--ram-hard-gib", type=positive_float, default=60.0)
    p.add_argument("--max-record-mib", type=positive_float, default=1.0)
    p.add_argument("--max-code-file-mib", type=positive_float, default=1.0)
    p.add_argument("--max-support-per-repo-mib", type=positive_float, default=0.5)
    p.add_argument("--stack-max-accept-per-shard-mib", type=positive_float, default=64.0,
                   help="cap accepted tokenizer text per Stack shard so sampling spans many shards")
    p.add_argument("--math-nemotron-share", type=float, default=0.50)
    p.add_argument("--allow-short", action="store_true")
    p.add_argument("--rebuild-stage", action="store_true")
    return p


def _physical_cores() -> int:
    try:
        import psutil
        return int(psutil.cpu_count(logical=False) or os.cpu_count() or 1)
    except Exception:
        n = os.cpu_count() or 1
        return max(1, n // 2)


def normalize_args(args):
    if args.corpus_gib is None:
        args.corpus_gib = 24.0 if args.mode == "final" else 2.0
    if args.threads is None:
        args.threads = _physical_cores()
    if not (0.0 <= args.math_nemotron_share <= 1.0):
        raise ValueError("--math-nemotron-share must be between 0 and 1")
    if args.ram_soft_gib >= args.ram_hard_gib:
        raise ValueError("--ram-soft-gib must be < --ram-hard-gib")
    if args.vocab_size <= len(SPECIAL_TOKENS) + 256:
        raise ValueError("vocab size too small for special tokens + byte alphabet")
    return args


def print_contract(args) -> None:
    print("=== 64K ByteLevel BPE tokenizer pipeline ===")
    print(f"script revision: {SCRIPT_REVISION}")
    print(f"mode           : {args.mode}")
    print(f"phase          : {args.phase}")
    print(f"corpus         : {args.corpus_gib:.2f} GiB UTF-8 text")
    print(f"vocab          : {args.vocab_size:,} total, including {len(SPECIAL_TOKENS)} specials")
    print(f"threads        : {args.threads}")
    print(f"RAM soft/hard  : {args.ram_soft_gib:.1f}/{args.ram_hard_gib:.1f} GiB")
    print(f"raw root       : {args.raw_root}")
    print(f"stack root     : {args.stack_root}")
    print(f"nemotron root  : {args.nemotron_root}")
    print(f"stage dir      : {args.stage_dir}")
    print(f"output dir     : {args.output_dir}")
    print("special IDs:")
    for i, s in enumerate(SPECIAL_TOKENS):
        print(f"  {i:2d} {s}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    print_contract(args)
    if args.phase == "build":
        build_stage(args)
    elif args.phase == "repair-code":
        repair_code_stage(args)
    elif args.phase == "repair-wiki":
        repair_wiki_stage(args)
    elif args.phase == "train":
        train_monitored(args)
    elif args.phase == "train-child":
        train_child(args)
    elif args.phase == "validate":
        validate_tokenizer(args)
    elif args.phase == "all":
        build_stage(args)
        train_monitored(args)
        validate_tokenizer(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
