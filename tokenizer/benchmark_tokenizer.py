#!/usr/bin/env python3
"""
Strict held-out benchmark builder + evaluator for the 64K ByteLevel BPE tokenizer.

Design:
- Reads the tokenizer training stage manifest and excludes every raw source file/shard
  that contributed text to tokenizer training.
- Builds ~1 GiB (default) of held-out text, stratified across:
  FineWeb-Edu, Nemotron Math, FineMath, peS2o, Wikipedia,
  Python/C/C++/Java/JavaScript/TypeScript, and repo support types.
- Stack-v3 holdout is sampled only from unused Parquet shards of the frozen generation.
- Writes length-prefixed UTF-8 .rec files so benchmark evaluation is fast and repeatable.
- Evaluates bytes/token, tokens/KiB, unknown-token usage, sampled exact round-trip,
  throughput, and optional token inflation versus a baseline tokenizer.

No network access is required.
"""

from __future__ import annotations

import argparse
import csv
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
import struct
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple


SCRIPT_REVISION = "2026-08-30-heldout-benchmark-wiki-repair-v3"
DEFAULT_SEED = 20260830

DEFAULT_TRAIN_MANIFEST = Path("/home/zjr1236/data/tokenizer_stage_24g/manifest.json")
DEFAULT_TOKENIZER = Path("/home/zjr1236/data/tokenizer_64k/tokenizer.json")
DEFAULT_HOLDOUT_DIR = Path("/home/zjr1236/data/tokenizer_benchmark_1g")
DEFAULT_REPORT_DIR = Path("/home/zjr1236/data/tokenizer_64k/benchmark")

DEFAULT_RAW_ROOT = Path("/home/zjr1236/data/raw")
DEFAULT_STACK_ROOT = Path("/mnt/D/data/stack-v3")
DEFAULT_NEMOTRON_ROOT = Path("/mnt/D/data/Nemotron-CC-Math-v1/4plus_MIND")
DEFAULT_STACK_GENERATION = "990b4288-3824-41ac-94a0-b6fd6fa23ffe"

# Exactly 1024 MiB.  These are raw UTF-8 text bytes, not file sizes.
DEFAULT_BENCH_MIB: "OrderedDict[str, int]" = OrderedDict([
    ("general", 128),
    ("math_nemotron", 64),
    ("math_finemath", 64),
    ("academic", 128),
    ("wiki", 64),

    ("python", 64),
    ("c", 64),
    ("cpp", 64),
    ("java", 64),
    ("javascript", 64),
    ("typescript", 64),

    ("markdown", 32),
    ("html", 32),
    ("shell", 32),
    ("json_yaml", 32),
    ("build", 32),
    ("frontend", 32),
])

ANCHOR_LANGUAGE_TO_CATEGORY = {
    "Python": "python",
    "C": "c",
    "C++": "cpp",
    "Java": "java",
    "JavaScript": "javascript",
    "TypeScript": "typescript",
}

SUPPORT_LANGUAGE_TO_CATEGORY = {
    # Documentation.
    "Markdown": "markdown",
    "MDX": "markdown",
    "reStructuredText": "markdown",
    "AsciiDoc": "markdown",
    "RMarkdown": "markdown",
    "Org": "markdown",

    # Web markup.
    "HTML": "html",
    "HTML+ERB": "html",
    "HTML+PHP": "html",
    "HTML+Razor": "html",
    "HTML+EEX": "html",
    "HTML+ECR": "html",
    "Java Server Pages": "html",
    "ASP.NET": "html",
    "Classic ASP": "html",

    # Shell / command environments.
    "Shell": "shell",
    "fish": "shell",
    "Nushell": "shell",
    "PowerShell": "shell",
    "Batchfile": "shell",

    # Structured configuration.
    "JSON": "json_yaml",
    "JSON with Comments": "json_yaml",
    "JSON5": "json_yaml",
    "Jsonnet": "json_yaml",
    "YAML": "json_yaml",
    "TOML": "json_yaml",
    "INI": "json_yaml",
    "Dotenv": "json_yaml",
    "HOCON": "json_yaml",

    # Build/project files.
    "Makefile": "build",
    "CMake": "build",
    "Maven POM": "build",
    "Gradle": "build",
    "Gradle Kotlin DSL": "build",
    "Dockerfile": "build",
    "Ninja": "build",
    "Meson": "build",
    "Ant Build System": "build",
    "BitBake": "build",
    "Cabal Config": "build",
    "Go Module": "build",
    "Pip Requirements": "build",
    "RPM Spec": "build",
    "Starlark": "build",

    # Frontend companions.
    "TSX": "frontend",
    "CSS": "frontend",
    "SCSS": "frontend",
    "Sass": "frontend",
    "Less": "frontend",
    "Vue": "frontend",
    "Svelte": "frontend",
    "Astro": "frontend",
    "PostCSS": "frontend",
    "Stylus": "frontend",
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

TEXT_COLUMN_CANDIDATES = (
    "text", "content", "raw_content", "document", "body", "markdown", "article"
)
MATH_FIELDS = (
    "text", "problem", "question", "prompt", "solution", "answer", "response",
    "reasoning", "explanation", "proof"
)

SOURCE_TO_MANIFEST_KEYS = {
    "general": ("general",),
    "academic": ("academic",),
    "wiki": ("wiki",),
    "math_nemotron": ("math_nemotron",),
    "math_finemath": ("math_finemath",),
    # repair-code versions may add a second Stack key.
    "stack": ("stack", "stack_repair_code", "repair_code", "code_repair"),
}

GROUPS = OrderedDict([
    ("general", ("general",)),
    ("math", ("math_nemotron", "math_finemath")),
    ("academic", ("academic",)),
    ("wiki", ("wiki",)),
    ("code_anchor", ("python", "c", "cpp", "java", "javascript", "typescript")),
    ("code_support", ("markdown", "html", "shell", "json_yaml", "build", "frontend")),
])


def allocate_bytes(total: int, shares: Mapping[str, float]) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    s = float(sum(shares.values()))
    if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"shares must sum to 1.0, got {s}")
    raw = {k: total * float(v) for k, v in shares.items()}
    out = {k: int(math.floor(v)) for k, v in raw.items()}
    left = total - sum(out.values())
    keys = list(shares)
    order = sorted(keys, key=lambda k: (-(raw[k] - out[k]), keys.index(k)))
    for k in order[:left]:
        out[k] += 1
    return out


def scaled_budgets(total_mib: int) -> "OrderedDict[str, int]":
    if total_mib <= 0:
        raise ValueError("--total-mib must be > 0")
    base_total = sum(DEFAULT_BENCH_MIB.values())
    shares = OrderedDict((k, v / base_total) for k, v in DEFAULT_BENCH_MIB.items())
    bytes_alloc = allocate_bytes(total_mib * (2**20), shares)
    return OrderedDict((k, bytes_alloc[k]) for k in DEFAULT_BENCH_MIB)


def effective_budgets(total_mib: int, overrides: Sequence[str]) -> "OrderedDict[str, int]":
    """Scale the default benchmark mixture, then apply explicit category=MiB overrides."""
    out = scaled_budgets(total_mib)
    parsed = parse_budget_overrides(overrides)
    for category, mib in parsed.items():
        out[category] = mib * (2**20)
    return out


def norm_path(p: os.PathLike | str) -> str:
    return str(Path(p).expanduser().resolve(strict=False))


def _stable_u64(seed: int, salt: str, value: str) -> int:
    h = hashlib.blake2b(digest_size=8, person=b"tokbench")
    h.update(str(seed).encode())
    h.update(b"\0")
    h.update(salt.encode())
    h.update(b"\0")
    h.update(value.encode("utf-8", errors="surrogatepass"))
    return int.from_bytes(h.digest(), "little")


def deterministic_order(items: Sequence[os.PathLike | str], seed: int, salt: str) -> List:
    return sorted(items, key=lambda x: _stable_u64(seed, salt, str(x)))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def used_paths_for(manifest: Mapping, source: str) -> Set[str]:
    selected = manifest.get("selected_inputs") or {}
    keys = SOURCE_TO_MANIFEST_KEYS.get(source, (source,))
    out: Set[str] = set()
    for key, vals in selected.items():
        if key in keys or (source == "stack" and str(key).startswith("stack")):
            if isinstance(vals, (list, tuple)):
                out.update(norm_path(x) for x in vals)
    return out


def path_is_used(path: Path, used: Set[str]) -> bool:
    return norm_path(path) in used


def chunk_text_utf8(text: str, max_bytes: int) -> Iterator[str]:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if not text:
        return
    data = text.encode("utf-8")
    start = 0
    while start < len(data):
        end = min(len(data), start + max_bytes)
        if end < len(data):
            while end > start and (data[end] & 0xC0) == 0x80:
                end -= 1
            if end == start:
                end = min(len(data), start + max_bytes)
                while end < len(data) and (data[end] & 0xC0) == 0x80:
                    end += 1
            window = max(start, end - min(4096, max_bytes // 2))
            nl = data.rfind(b"\n", window, end)
            if nl >= start:
                end = nl + 1
        yield data[start:end].decode("utf-8")
        start = end


class RecordWriter:
    def __init__(self, path: os.PathLike | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("wb")
        self.records = 0
        self.text_bytes = 0

    def write(self, text: str) -> int:
        payload = text.encode("utf-8")
        if len(payload) > 0xFFFFFFFF:
            raise ValueError("record exceeds uint32 length")
        self.f.write(struct.pack("<I", len(payload)))
        self.f.write(payload)
        self.records += 1
        self.text_bytes += len(payload)
        return len(payload)

    def close(self) -> None:
        if not self.f.closed:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()


def iter_records(path: os.PathLike | str) -> Iterator[str]:
    path = Path(path)
    with path.open("rb") as f:
        while True:
            h = f.read(4)
            if not h:
                return
            if len(h) != 4:
                raise IOError(f"truncated record header: {path}")
            (n,) = struct.unpack("<I", h)
            payload = f.read(n)
            if len(payload) != n:
                raise IOError(f"truncated record payload: {path}")
            yield payload.decode("utf-8")


def iter_record_batches(path: os.PathLike | str, batch_size: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for text in iter_records(path):
        batch.append(text)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@dataclass
class Sink:
    category: str
    target_bytes: int
    writer: RecordWriter
    max_record_bytes: int
    text_bytes: int = 0
    records: int = 0

    @property
    def done(self) -> bool:
        return self.text_bytes >= self.target_bytes

    @property
    def remaining(self) -> int:
        return max(0, self.target_bytes - self.text_bytes)

    def add(self, text: str) -> int:
        if self.done:
            return 0
        added = 0
        for piece in chunk_text_utf8(text, self.max_record_bytes):
            if self.done:
                break
            raw = piece.encode("utf-8")
            if len(raw) > self.remaining:
                fitted = next(chunk_text_utf8(piece, self.remaining), "") if self.remaining else ""
                if not fitted:
                    break
                piece = fitted
                raw = piece.encode("utf-8")
            if not raw:
                continue
            self.writer.write(piece)
            n = len(raw)
            self.text_bytes += n
            self.records += 1
            added += n
        return added


def acceptable_text(text: object, min_bytes: int = 64) -> bool:
    if not isinstance(text, str) or not text:
        return False
    try:
        b = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    if len(b) < min_bytes:
        return False
    if b.count(b"\x00") > max(2, len(b) // 10000):
        return False
    return True


def is_bad_code_path(path: str) -> bool:
    p = "/" + (path or "").replace("\\", "/").lower().lstrip("/")
    base = p.rsplit("/", 1)[-1]
    if any(x in p for x in BAD_PATH_PARTS):
        return True
    if base in BAD_BASENAMES:
        return True
    return any(base.endswith(x) for x in BAD_SUFFIXES)


def looks_minified(text: str) -> bool:
    if len(text) < 8192:
        return False
    nl = text.count("\n")
    if nl == 0:
        return len(text) > 16384
    return (len(text) / (nl + 1)) > 1200


def stack_category(language: Optional[str], file_path: Optional[str]) -> Optional[str]:
    language = language or ""
    path = (file_path or "").replace("\\", "/")
    base = path.rsplit("/", 1)[-1].lower()

    if language in ANCHOR_LANGUAGE_TO_CATEGORY:
        return ANCHOR_LANGUAGE_TO_CATEGORY[language]

    # Strong filename semantics first.
    if base.startswith("readme"):
        return "markdown"
    if base in {"package.json", "tsconfig.json", "deno.json", "deno.jsonc"}:
        return "json_yaml"
    if base in {
        "dockerfile", "makefile", "cmakelists.txt", "meson.build", "build.gradle",
        "build.gradle.kts", "pom.xml", "go.mod", "go.sum", "pyproject.toml",
        "requirements.txt", "setup.cfg",
    }:
        return "build"

    return SUPPORT_LANGUAGE_TO_CATEGORY.get(language)


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq
        return pq
    except Exception as e:
        raise RuntimeError("pyarrow is required: pip install pyarrow") from e


def _import_zstandard():
    try:
        import zstandard as zstd
        return zstd
    except Exception as e:
        raise RuntimeError("zstandard is required for .zst sources: pip install zstandard") from e


def list_source_files(root: Path) -> List[Path]:
    exts = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst", ".zst", ".txt", ".txt.gz")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and any(p.name.endswith(x) for x in exts)
    )


def _choose_parquet_fields(pf, source_kind: str) -> Tuple[str, ...]:
    schema = pf.schema_arrow
    names = set(schema.names)
    if source_kind == "math":
        fields = tuple(x for x in MATH_FIELDS if x in names)
        if "text" in fields:
            return ("text",)
        if fields:
            return fields
    for x in TEXT_COLUMN_CANDIDATES:
        if x in names:
            return (x,)
    for field in schema:
        if str(field.type) in {"string", "large_string"}:
            return (field.name,)
    raise ValueError(f"no usable text column found; schema={schema}")


def _combine_fields(data: Mapping[str, list], fields: Sequence[str], i: int) -> Optional[str]:
    vals = []
    for k in fields:
        v = data[k][i]
        if isinstance(v, str) and v:
            vals.append(v)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else "\n\n".join(vals)


def iter_parquet_texts(path: Path, source_kind: str, batch_size: int = 512) -> Iterator[str]:
    pq = _import_pyarrow()
    pf = pq.ParquetFile(path)
    fields = _choose_parquet_fields(pf, source_kind)
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(fields), use_threads=True):
        data = batch.to_pydict()
        for i in range(batch.num_rows):
            text = _combine_fields(data, fields, i)
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
        line = line.rstrip("\r\n")
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if source_kind != "math":
                yield line
            continue
        text = _json_text(obj, source_kind)
        if text:
            yield text


def iter_texts_from_file(path: Path, source_kind: str) -> Iterator[str]:
    n = path.name
    if n.endswith(".parquet"):
        yield from iter_parquet_texts(path, source_kind)
    elif n.endswith(".jsonl"):
        with path.open("rt", encoding="utf-8", errors="strict") as f:
            yield from iter_json_lines(f, source_kind)
    elif n.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
            yield from iter_json_lines(f, source_kind)
    elif n.endswith(".txt"):
        with path.open("rt", encoding="utf-8", errors="strict") as f:
            yield f.read()
    elif n.endswith(".txt.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
            yield f.read()
    elif n.endswith(".zst"):
        zstd = _import_zstandard()
        with path.open("rb") as raw:
            with zstd.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8", errors="strict") as txt:
                    yield from iter_json_lines(txt, source_kind)
    else:
        raise ValueError(f"unsupported source file: {path}")


def discover_root(
    manifest: Mapping,
    manifest_key: str,
    override: Optional[Path],
    raw_root: Path,
    candidates: Sequence[str],
) -> Path:
    if override is not None:
        p = override.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    roots = manifest.get("roots") or {}
    raw = roots.get(manifest_key)
    if raw:
        p = Path(raw).expanduser().resolve()
        if p.exists():
            return p

    for name in candidates:
        p = raw_root / name
        if p.exists():
            return p.resolve()

    raise FileNotFoundError(
        f"cannot locate {manifest_key}; use explicit --{manifest_key.replace('_','-')}-root"
    )


_WIKI_SNAPSHOT_RE = re.compile(r"^(?P<date>\d{8})\.(?P<lang>[A-Za-z0-9_-]+)$")


def resolve_english_wiki_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Wikipedia root does not exist: {root}")
    m = _WIKI_SNAPSHOT_RE.match(root.name)
    if m:
        if m.group("lang").lower() != "en":
            raise ValueError(
                f"Wikipedia root is a non-English snapshot ({root.name}); "
                "English Wikipedia is required."
            )
        return root

    preferred = root / "snapshots" / "master"
    search_roots = [preferred] if preferred.is_dir() else [root]
    candidates = []
    for base in search_roots:
        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            mm = _WIKI_SNAPSHOT_RE.match(p.name)
            if mm and mm.group("lang").lower() == "en":
                candidates.append((mm.group("date"), str(p), p.resolve()))
    if not candidates:
        raise FileNotFoundError(
            f"no English Wikipedia YYYYMMDD.en snapshot found under {root}"
        )
    candidates.sort(reverse=True)
    return candidates[0][2]


def _path_under(path: str, root: Path) -> bool:
    p = Path(path).expanduser().resolve(strict=False)
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def fill_generic(
    sink: Sink,
    root: Path,
    used_paths: Set[str],
    source_kind: str,
    seed: int,
    selected: List[str],
) -> None:
    files = [p for p in list_source_files(root) if not path_is_used(p, used_paths)]
    files = deterministic_order(files, seed, sink.category)
    if not files:
        raise RuntimeError(
            f"{sink.category}: no strictly held-out source files remain under {root}. "
            "The benchmark refuses to reuse tokenizer-training files."
        )

    print(f"\n[{sink.category}] held-out candidates={len(files):,}, used excluded={len(used_paths):,}")
    for i, path in enumerate(files, 1):
        if sink.done:
            break
        before = sink.text_bytes
        try:
            for text in iter_texts_from_file(path, source_kind):
                if sink.done:
                    break
                if acceptable_text(text):
                    sink.add(text)
        except Exception as e:
            print(f"WARN: skip {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if sink.text_bytes > before:
            selected.append(str(path))
            print(
                f"  {sink.text_bytes/2**20:8.1f}/{sink.target_bytes/2**20:8.1f} MiB "
                f"<- {path.name}"
            )
    if not sink.done:
        raise RuntimeError(
            f"{sink.category}: strict held-out data exhausted; "
            f"got {sink.text_bytes/2**20:.1f} MiB / {sink.target_bytes/2**20:.1f} MiB"
        )


def find_stack_parquets(root: Path, generation: str) -> List[Path]:
    suffix = f"-{generation}-c000.snappy.parquet"
    out = sorted(p for p in root.rglob("*.parquet") if p.name.endswith(suffix))
    if not out:
        raise FileNotFoundError(
            f"no Stack-v3 shard generation {generation} under {root}"
        )
    return out


def fill_stack(
    sinks: Mapping[str, Sink],
    root: Path,
    generation: str,
    used_paths: Set[str],
    seed: int,
    max_code_file_bytes: int,
    max_accept_per_shard_bytes: int,
    selected: List[str],
) -> None:
    pq = _import_pyarrow()
    paths = [p for p in find_stack_parquets(root, generation) if not path_is_used(p, used_paths)]
    paths = deterministic_order(paths, seed, "stack-heldout")
    if not paths:
        raise RuntimeError(
            "no strictly held-out Stack-v3 shards remain after excluding tokenizer-training shards"
        )

    columns = [
        "repo_path",
        "files.list.element.content",
        "files.list.element.size_bytes",
        "files.list.element.file_path",
        "files.list.element.language",
        "files.list.element.is_vendor",
    ]

    def all_done() -> bool:
        return all(x.done for x in sinks.values())

    print(
        f"\n[stack] generation={generation} "
        f"held-out candidates={len(paths):,}, tokenizer-used excluded={len(used_paths):,}"
    )

    for shard_no, path in enumerate(paths, 1):
        if all_done():
            break
        pf = pq.ParquetFile(path)
        rng = random.Random(_stable_u64(seed, "stack-shard", path.name))
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
                files = row.get("files") or []
                rng.shuffle(files)
                for f in files:
                    if all_done() or accepted_shard >= max_accept_per_shard_bytes:
                        break
                    if not isinstance(f, dict) or f.get("is_vendor"):
                        continue
                    path_s = f.get("file_path") or ""
                    if is_bad_code_path(path_s):
                        continue
                    text = f.get("content")
                    if not acceptable_text(text):
                        continue
                    if len(text.encode("utf-8")) > max_code_file_bytes:
                        continue
                    if looks_minified(text):
                        continue

                    cat = stack_category(f.get("language"), path_s)
                    if cat is None or cat not in sinks or sinks[cat].done:
                        continue
                    n = sinks[cat].add(text)
                    if n:
                        accepted_shard += n
                        touched = True

        if touched:
            selected.append(str(path))

        progress = " ".join(
            f"{k}={v.text_bytes/2**20:.0f}/{v.target_bytes/2**20:.0f}MiB"
            for k, v in sinks.items()
        )
        print("    " + progress)

    missing = {
        k: v.target_bytes - v.text_bytes
        for k, v in sinks.items() if not v.done
    }
    if missing:
        pretty = {k: f"{v/2**20:.1f} MiB" for k, v in missing.items()}
        raise RuntimeError(f"Stack strict held-out data exhausted before quotas filled: {pretty}")


def sha256_file(path: Path, buf_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(buf_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_holdout(args) -> Path:
    train_manifest = load_json(args.train_manifest.expanduser().resolve())
    stage = args.holdout_dir.expanduser().resolve()
    manifest_path = stage / "heldout_manifest.json"

    if manifest_path.exists() and not args.rebuild_holdout:
        existing = load_json(manifest_path)
        expected = {
            "script_revision": SCRIPT_REVISION,
            "seed": args.seed,
            "total_mib": args.total_mib,
            "budgets_bytes": dict(effective_budgets(args.total_mib, args.budget)),
            "stack_generation": args.stack_generation or train_manifest.get(
                "stack_generation", DEFAULT_STACK_GENERATION
            ),
            "train_manifest_sha256": sha256_file(args.train_manifest.expanduser().resolve()),
        }
        actual = {k: existing.get(k) for k in expected}
        if actual != expected:
            raise RuntimeError(
                f"existing held-out manifest mismatch.\nactual={actual}\nexpected={expected}\n"
                "Use --rebuild-holdout."
            )
        print(f"Held-out corpus already exists: {manifest_path}")
        return manifest_path

    if args.rebuild_holdout and stage.exists():
        import shutil
        shutil.rmtree(stage)
    records_dir = stage / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    budgets = effective_budgets(args.total_mib, args.budget)
    sinks: Dict[str, Sink] = {}
    max_record_bytes = int(args.max_record_mib * 2**20)
    for cat, n in budgets.items():
        sinks[cat] = Sink(
            cat, n, RecordWriter(records_dir / f"{cat}.rec"), max_record_bytes
        )

    raw_root = args.raw_root.expanduser().resolve()
    roots = {
        "fineweb": discover_root(
            train_manifest, "fineweb", args.fineweb_root, raw_root,
            ["fineweb-edu", "fineweb_edu", "FineWeb-Edu", "fineweb"],
        ),
        "finemath": discover_root(
            train_manifest, "finemath", args.finemath_root, raw_root,
            ["finemath", "FineMath", "fine-math"],
        ),
        "pes2o": discover_root(
            train_manifest, "pes2o", args.pes2o_root, raw_root,
            ["pes2o", "peS2o", "pes2o-v3"],
        ),
        "wiki": resolve_english_wiki_root(discover_root(
            train_manifest, "wiki", args.wiki_root, raw_root,
            ["wikipedia", "wiki", "wikipedia-en"],
        )),
        "nemotron": (
            args.nemotron_root.expanduser().resolve()
            if args.nemotron_root is not None
            else Path((train_manifest.get("roots") or {}).get("nemotron", DEFAULT_NEMOTRON_ROOT))
                .expanduser().resolve()
        ),
        "stack": (
            args.stack_root.expanduser().resolve()
            if args.stack_root is not None
            else Path((train_manifest.get("roots") or {}).get("stack", DEFAULT_STACK_ROOT))
                .expanduser().resolve()
        ),
    }

    for name, p in roots.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} root missing: {p}")

    generation = (
        args.stack_generation
        or train_manifest.get("stack_generation")
        or DEFAULT_STACK_GENERATION
    )

    selected: Dict[str, List[str]] = {k: [] for k in budgets}

    try:
        fill_generic(
            sinks["general"], roots["fineweb"],
            used_paths_for(train_manifest, "general"),
            "general", args.seed, selected["general"],
        )
        fill_generic(
            sinks["math_nemotron"], roots["nemotron"],
            used_paths_for(train_manifest, "math_nemotron"),
            "math", args.seed, selected["math_nemotron"],
        )
        fill_generic(
            sinks["math_finemath"], roots["finemath"],
            used_paths_for(train_manifest, "math_finemath"),
            "math", args.seed, selected["math_finemath"],
        )
        fill_generic(
            sinks["academic"], roots["pes2o"],
            used_paths_for(train_manifest, "academic"),
            "general", args.seed, selected["academic"],
        )
        fill_generic(
            sinks["wiki"], roots["wiki"],
            used_paths_for(train_manifest, "wiki"),
            "general", args.seed, selected["wiki"],
        )

        stack_categories = [
            "python", "c", "cpp", "java", "javascript", "typescript",
            "markdown", "html", "shell", "json_yaml", "build", "frontend",
        ]
        stack_selected: List[str] = []
        fill_stack(
            {k: sinks[k] for k in stack_categories},
            roots["stack"],
            generation,
            used_paths_for(train_manifest, "stack"),
            args.seed,
            max_code_file_bytes=int(args.max_code_file_mib * 2**20),
            max_accept_per_shard_bytes=int(args.stack_max_accept_per_shard_mib * 2**20),
            selected=stack_selected,
        )
        # Store the same selected shard list for all Stack categories, plus a compact top-level key.
        for k in stack_categories:
            selected[k] = stack_selected
    finally:
        for s in sinks.values():
            s.writer.close()

    short = {k: s.remaining for k, s in sinks.items() if not s.done}
    if short:
        raise RuntimeError(f"held-out build short: {short}")

    source_sha = sha256_file(args.train_manifest.expanduser().resolve())
    manifest = {
        "script_revision": SCRIPT_REVISION,
        "created_at_unix": time.time(),
        "seed": args.seed,
        "total_mib": args.total_mib,
        "budgets_bytes": dict(budgets),
        "total_target_bytes": sum(budgets.values()),
        "train_manifest": norm_path(args.train_manifest),
        "train_manifest_sha256": source_sha,
        "stack_generation": generation,
        "roots": {k: str(v) for k, v in roots.items()},
        "strict_heldout": True,
        "categories": {},
        "selected_inputs": selected,
    }

    for cat, sink in sinks.items():
        rec_path = sink.writer.path
        manifest["categories"][cat] = {
            "path": str(rec_path),
            "target_bytes": sink.target_bytes,
            "text_bytes": sink.text_bytes,
            "records": sink.records,
            "sha256": None if args.no_sha256 else sha256_file(rec_path),
        }

    if sum(x["text_bytes"] for x in manifest["categories"].values()) != sum(budgets.values()):
        raise RuntimeError("internal error: held-out text byte total does not match exact target")

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    os.replace(tmp, manifest_path)

    print("\n=== HELD-OUT CORPUS COMPLETE ===")
    for cat, meta in manifest["categories"].items():
        print(
            f"{cat:18s} {meta['text_bytes']/2**20:8.1f} MiB "
            f"{meta['records']:10,d} records"
        )
    print(f"total: {sum(budgets.values())/2**30:.3f} GiB")
    print(f"manifest: {manifest_path}")
    return manifest_path


def _atomic_write_json(path: Path, obj: Mapping) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _replace_file_and_manifest(temp: Path, final: Path, manifest_path: Path, new_manifest: Mapping) -> None:
    backup = final.with_name(final.name + f".repair-bak-{os.getpid()}-{time.time_ns()}")
    had_old = final.exists()
    try:
        if had_old:
            os.replace(final, backup)
        os.replace(temp, final)
        _atomic_write_json(manifest_path, new_manifest)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            final.unlink()
        if had_old and backup.exists():
            os.replace(backup, final)
        raise
    else:
        with contextlib.suppress(FileNotFoundError):
            backup.unlink()


def repair_wiki_holdout(args) -> Path:
    """Replace only the Wiki held-out record using unused English-Wiki files."""
    train_manifest_path = args.train_manifest.expanduser().resolve()
    train_manifest = load_json(train_manifest_path)
    stage = args.holdout_dir.expanduser().resolve()
    manifest_path = stage / "heldout_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"held-out manifest missing: {manifest_path}; build the benchmark corpus first"
        )
    manifest = load_json(manifest_path)
    wiki_meta = (manifest.get("categories") or {}).get("wiki")
    if not wiki_meta:
        raise RuntimeError("held-out manifest has no wiki category")

    raw_root = args.raw_root.expanduser().resolve()
    discovered = discover_root(
        train_manifest, "wiki", args.wiki_root, raw_root,
        ["wikipedia", "wiki", "wikipedia-en"],
    )
    wiki_root = resolve_english_wiki_root(discovered)
    train_used = used_paths_for(train_manifest, "wiki")
    outside = sorted(p for p in train_used if not _path_under(p, wiki_root))
    if outside:
        preview = "\n".join(outside[:5])
        raise RuntimeError(
            "tokenizer training manifest still contains non-English/out-of-root Wiki inputs. "
            "Run tokenizer_pipeline.py --phase repair-wiki first. Examples:\n" + preview
        )

    target = int(wiki_meta["target_bytes"])
    final_path = Path(wiki_meta["path"]).expanduser().resolve()
    temp_path = final_path.with_name(final_path.name + f".repair-wiki.tmp.{os.getpid()}")
    with contextlib.suppress(FileNotFoundError):
        temp_path.unlink()
    sink = Sink("wiki", target, RecordWriter(temp_path), int(args.max_record_mib * 2**20))
    selected: List[str] = []

    print("\n=== Repairing only English-Wikipedia held-out bucket ===")
    print(f"  wiki root       : {wiki_root}")
    print(f"  train exclusions: {len(train_used):,} file(s)")
    print(f"  target          : {target/2**20:.1f} MiB")
    try:
        fill_generic(sink, wiki_root, train_used, "general", args.seed, selected)
    finally:
        sink.writer.close()

    if not sink.done or sink.text_bytes != target:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError(f"Wiki benchmark repair short: {sink.text_bytes} != {target}")
    if any(path_is_used(Path(x), train_used) for x in selected):
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError("strict holdout violation: selected Wiki file was used in tokenizer training")

    before_total = sum(int(x["text_bytes"]) for x in manifest["categories"].values())
    new_manifest = json.loads(json.dumps(manifest))
    new_manifest["script_revision"] = SCRIPT_REVISION
    new_manifest["train_manifest"] = str(train_manifest_path)
    new_manifest["train_manifest_sha256"] = sha256_file(train_manifest_path)
    new_manifest.setdefault("roots", {})["wiki"] = str(wiki_root)
    new_manifest.setdefault("selected_inputs", {})["wiki"] = [str(Path(x).resolve()) for x in selected]
    new_manifest["categories"]["wiki"] = {
        "path": str(final_path),
        "target_bytes": target,
        "text_bytes": sink.text_bytes,
        "records": sink.records,
        "sha256": None if args.no_sha256 else sha256_file(temp_path),
    }
    new_manifest.setdefault("repairs", []).append({
        "kind": "wiki",
        "script_revision": SCRIPT_REVISION,
        "source_root": str(wiki_root),
        "created_at_unix": time.time(),
    })
    new_manifest["updated_at_unix"] = time.time()
    after_total = sum(int(x["text_bytes"]) for x in new_manifest["categories"].values())
    if before_total != after_total:
        with contextlib.suppress(FileNotFoundError):
            temp_path.unlink()
        raise RuntimeError(f"Wiki repair changed benchmark byte total: {before_total} -> {after_total}")

    _replace_file_and_manifest(temp_path, final_path, manifest_path, new_manifest)
    print("\nEnglish-Wiki held-out repair complete:")
    print(f"  wiki   : {target/2**20:.1f} MiB exact")
    print(f"  files  : {len(selected)} strictly held-out English Wiki file(s)")
    return manifest_path


def resolve_tokenizer_json(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {path}")
    return path


def load_tokenizer(path: Path):
    try:
        from tokenizers import Tokenizer
    except Exception as e:
        raise RuntimeError("tokenizers is required: pip install tokenizers") from e
    return Tokenizer.from_file(str(resolve_tokenizer_json(path)))


def evaluate_category(
    path: Path,
    tokenizer,
    baseline,
    batch_size: int,
    roundtrip_samples: int,
) -> dict:
    total_bytes = 0
    total_tokens = 0
    baseline_tokens = 0
    unk_count = 0
    roundtrip_checked = 0
    roundtrip_failures = 0
    unk_id = tokenizer.token_to_id("<|unk|>")
    t0 = time.time()

    for texts in iter_record_batches(path, batch_size):
        enc = tokenizer.encode_batch(texts, add_special_tokens=False)
        base_enc = baseline.encode_batch(texts, add_special_tokens=False) if baseline else None

        for i, (text, e) in enumerate(zip(texts, enc)):
            raw_n = len(text.encode("utf-8"))
            tok_n = len(e.ids)
            total_bytes += raw_n
            total_tokens += tok_n
            if unk_id is not None:
                unk_count += e.ids.count(unk_id)

            if roundtrip_checked < roundtrip_samples:
                decoded = tokenizer.decode(e.ids, skip_special_tokens=False)
                roundtrip_checked += 1
                if decoded != text:
                    roundtrip_failures += 1

            if base_enc is not None:
                baseline_tokens += len(base_enc[i].ids)

    elapsed = time.time() - t0
    if total_tokens == 0:
        raise RuntimeError(f"zero tokens from {path}")

    out = {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
        "tokens_per_kib": total_tokens / (total_bytes / 1024.0),
        "unk_tokens": unk_count,
        "unk_rate": unk_count / total_tokens,
        "roundtrip_checked": roundtrip_checked,
        "roundtrip_failures": roundtrip_failures,
        "elapsed_seconds": elapsed,
        "throughput_mib_s": (total_bytes / 2**20) / elapsed if elapsed > 0 else None,
    }
    if baseline:
        out["baseline_tokens"] = baseline_tokens
        out["token_inflation"] = total_tokens / baseline_tokens - 1.0
        out["baseline_bytes_per_token"] = total_bytes / baseline_tokens
    return out


def aggregate_metrics(report: Mapping[str, dict], cats: Sequence[str]) -> dict:
    rows = [report[c] for c in cats if c in report]
    b = sum(x["bytes"] for x in rows)
    t = sum(x["tokens"] for x in rows)
    out = {
        "bytes": b,
        "tokens": t,
        "bytes_per_token": b / t if t else None,
        "tokens_per_kib": t / (b / 1024.0) if b else None,
        "unk_tokens": sum(x["unk_tokens"] for x in rows),
        "roundtrip_failures": sum(x["roundtrip_failures"] for x in rows),
    }
    if rows and all("baseline_tokens" in x for x in rows):
        bt = sum(x["baseline_tokens"] for x in rows)
        out["baseline_tokens"] = bt
        out["token_inflation"] = t / bt - 1.0 if bt else None
        out["baseline_bytes_per_token"] = b / bt if bt else None
    return out


def benchmark(args) -> Path:
    heldout_manifest = args.holdout_dir.expanduser().resolve() / "heldout_manifest.json"
    manifest = load_json(heldout_manifest)
    if not manifest.get("strict_heldout"):
        raise RuntimeError("held-out manifest is not marked strict_heldout=true")

    tokenizer_path = resolve_tokenizer_json(args.tokenizer)
    tokenizer = load_tokenizer(tokenizer_path)
    baseline = load_tokenizer(args.baseline_tokenizer) if args.baseline_tokenizer else None

    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    categories = manifest["categories"]
    results: Dict[str, dict] = {}

    print("\n=== TOKENIZER BENCHMARK ===")
    print(f"tokenizer : {tokenizer_path}")
    if args.baseline_tokenizer:
        print(f"baseline  : {resolve_tokenizer_json(args.baseline_tokenizer)}")

    for cat in DEFAULT_BENCH_MIB:
        if cat not in categories:
            continue
        path = Path(categories[cat]["path"])
        print(f"\n[{cat}] {categories[cat]['text_bytes']/2**20:.1f} MiB")
        m = evaluate_category(
            path, tokenizer, baseline,
            batch_size=args.batch_size,
            roundtrip_samples=args.roundtrip_samples,
        )
        results[cat] = m
        extra = (
            f" inflation={100*m['token_inflation']:+.2f}%"
            if "token_inflation" in m else ""
        )
        print(
            f"  {m['bytes_per_token']:.4f} bytes/token  "
            f"{m['tokens_per_kib']:.2f} tokens/KiB  "
            f"UNK={m['unk_tokens']}  RTfail={m['roundtrip_failures']}"
            f"{extra}"
        )

    groups = {
        name: aggregate_metrics(results, cats)
        for name, cats in GROUPS.items()
    }
    total = aggregate_metrics(results, tuple(results.keys()))

    full_report = {
        "script_revision": SCRIPT_REVISION,
        "created_at_unix": time.time(),
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "baseline_tokenizer": (
            str(resolve_tokenizer_json(args.baseline_tokenizer))
            if args.baseline_tokenizer else None
        ),
        "baseline_sha256": (
            sha256_file(resolve_tokenizer_json(args.baseline_tokenizer))
            if args.baseline_tokenizer else None
        ),
        "heldout_manifest": str(heldout_manifest),
        "heldout_manifest_sha256": sha256_file(heldout_manifest),
        "categories": results,
        "groups": groups,
        "total": total,
    }

    json_path = report_dir / "benchmark_report.json"
    json_path.write_text(json.dumps(full_report, indent=2, ensure_ascii=False))

    csv_path = report_dir / "benchmark_report.csv"
    fields = [
        "category", "bytes", "tokens", "bytes_per_token", "tokens_per_kib",
        "unk_tokens", "unk_rate", "roundtrip_checked", "roundtrip_failures",
        "elapsed_seconds", "throughput_mib_s", "baseline_tokens",
        "baseline_bytes_per_token", "token_inflation",
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cat, m in results.items():
            row = {"category": cat, **m}
            w.writerow({k: row.get(k) for k in fields})

    print("\n=== GROUP SUMMARY ===")
    for name, m in groups.items():
        if not m["bytes"]:
            continue
        extra = (
            f"  inflation={100*m['token_inflation']:+.2f}%"
            if "token_inflation" in m else ""
        )
        print(f"{name:14s} {m['bytes_per_token']:.4f} bytes/token{extra}")

    extra = (
        f" inflation={100*total['token_inflation']:+.2f}%"
        if "token_inflation" in total else ""
    )
    print(f"\nTOTAL {total['bytes_per_token']:.4f} bytes/token{extra}")
    print(f"JSON: {json_path}")
    print(f"CSV : {csv_path}")

    if total["roundtrip_failures"] != 0:
        raise RuntimeError(
            f"round-trip validation failed for {total['roundtrip_failures']} held-out samples"
        )
    if total["unk_tokens"] != 0:
        raise RuntimeError(
            f"unexpected <|unk|> usage on ByteLevel held-out corpus: {total['unk_tokens']}"
        )
    return json_path


def parse_budget_overrides(values: Sequence[str]) -> Dict[str, int]:
    out = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"--budget must be category=MiB, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in DEFAULT_BENCH_MIB:
            raise ValueError(f"unknown benchmark category {k!r}")
        n = int(v)
        if n <= 0:
            raise ValueError("benchmark MiB must be > 0")
        out[k] = n
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Build a strict held-out tokenizer corpus and benchmark tokenizer compression.",
    )
    p.add_argument("--phase", choices=["all", "build", "repair-wiki", "bench"], default="all")
    p.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--baseline-tokenizer", type=Path, default=None,
        help="optional old/baseline tokenizer.json or directory containing tokenizer.json",
    )
    p.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)

    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--stack-root", type=Path, default=None)
    p.add_argument("--nemotron-root", type=Path, default=None)
    p.add_argument("--fineweb-root", type=Path, default=None)
    p.add_argument("--finemath-root", type=Path, default=None)
    p.add_argument("--pes2o-root", type=Path, default=None)
    p.add_argument("--wiki-root", type=Path, default=None)
    p.add_argument("--stack-generation", default=None)

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--total-mib", type=int, default=1024,
        help="scale the default category mixture to this many held-out MiB",
    )
    p.add_argument(
        "--budget", action="append", default=[],
        help="optional category=MiB override; when used, all unspecified categories keep scaled defaults",
    )
    p.add_argument("--max-record-mib", type=float, default=1.0)
    p.add_argument("--max-code-file-mib", type=float, default=1.0)
    p.add_argument(
        "--stack-max-accept-per-shard-mib", type=float, default=16.0,
        help="forces Stack benchmark to span many unused shards",
    )
    p.add_argument("--rebuild-holdout", action="store_true")
    p.add_argument("--no-sha256", action="store_true")

    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument(
        "--roundtrip-samples", type=int, default=2000,
        help="exact encode/decode checks per category",
    )
    return p


def normalize_args(args):
    if args.total_mib <= 0:
        raise ValueError("--total-mib must be > 0")
    if args.max_record_mib <= 0 or args.max_code_file_mib <= 0:
        raise ValueError("MiB limits must be > 0")
    if args.stack_max_accept_per_shard_mib <= 0:
        raise ValueError("--stack-max-accept-per-shard-mib must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.roundtrip_samples < 0:
        raise ValueError("--roundtrip-samples must be >= 0")
    return args


def print_contract(args):
    print("=== Strict held-out tokenizer benchmark ===")
    print(f"script revision : {SCRIPT_REVISION}")
    print(f"phase           : {args.phase}")
    print(f"train manifest  : {args.train_manifest}")
    print(f"tokenizer       : {args.tokenizer}")
    print(f"baseline        : {args.baseline_tokenizer or '<none>'}")
    print(f"heldout dir     : {args.holdout_dir}")
    print(f"report dir      : {args.report_dir}")
    print(f"total heldout   : {args.total_mib} MiB")
    print(f"seed            : {args.seed}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    print_contract(args)

    if args.phase in {"all", "build"}:
        build_holdout(args)
    if args.phase == "repair-wiki":
        repair_wiki_holdout(args)
    if args.phase in {"all", "bench"}:
        benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
