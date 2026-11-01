#!/usr/bin/env python3
"""Fast 1B-token read/count pipeline with 28-language repository-preserving code packs.

This version intentionally implements only two operations:
  1) ingest raw source records into large source-isolated .pack shards on SSD;
  2) stream each .pack shard in bounded batches and count exact tokenizer payload tokens.

No FIM, AST, final sequence packing, or training integration is performed here.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import gzip
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import threading
import queue
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

SCRIPT_REVISION = "2026-09-11-read-count-v7-resume-capacity-safe"
CHECKPOINT_VERSION = 2
DEFAULT_STALL_TIMEOUT_S = 600.0
STRICT_PACK_CONTRACT = "PTSTAGE1"
DEFAULT_SEED = 20260830
DEFAULT_TARGET_TOKENS = 1_250_000_000
DEFAULT_STACK_GENERATION = "990b4288-3824-41ac-94a0-b6fd6fa23ffe"

DEFAULT_TOKENIZER = Path("/home/zjr1236/data/tokenizer_64k/tokenizer.json")
DEFAULT_RAW_ROOT = Path("/home/zjr1236/data/raw")
DEFAULT_FINEWEB_ROOT = Path("/mnt/D/data/fineweb-edu-ms-200B/sample/350BT")
DEFAULT_STACK_ROOT = Path("/mnt/D/data/stack-v3")
DEFAULT_NEMOTRON_ROOT = Path("/home/zjr1236/data/raw/Nemotron-CC-Math-v1/4plus_MIND")
DEFAULT_STAGE_DIR = Path("/home/zjr1236/data/pretrain_read_count_1p25b_strict")

SOURCE_SHARES: "OrderedDict[str, float]" = OrderedDict([
    ("general", 0.45),
    ("math_nemotron", 0.125),
    ("math_finemath", 0.125),
    ("code", 0.20),
    ("academic", 0.07),
    ("wiki", 0.03),
])
SOURCE_ORDER = tuple(SOURCE_SHARES)

# Empirical bytes/token priors from the successful 1B V2 probe.  They are used
# only to decide how early to flush/count and how much work to prefetch; exact
# quotas still come exclusively from tokenizer counts.
SOURCE_BYTES_PER_TOKEN_PRIOR = {
    "general": 4.8,
    "math_nemotron": 3.9,
    "math_finemath": 3.7,
    "code": 3.6,
    "academic": 4.8,
    "wiki": 4.4,
}

SPECIAL_TOKENS = [
    "<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|repo_start|>", "<|repo_end|>", "<|file_start|>",
    "<|file_end|>", "<|document_start|>", "<|document_end|>",
]

# Natural frequency: these are filters/labels, NOT quota buckets.
CODE_LANGUAGES = (
    "Python", "C", "C++", "C#", "Java", "JavaScript", "TypeScript", "TSX",
    "Go", "Rust", "Ruby", "PHP", "Kotlin", "Swift", "Scala", "Lua",
    "Shell", "Dart", "Julia", "R", "Haskell", "Elixir", "Erlang", "OCaml",
    "Clojure", "Objective-C", "Zig", "Solidity",
)
CODE_LANGUAGE_IDS = {name: i + 1 for i, name in enumerate(CODE_LANGUAGES)}
ID_TO_CODE_LANGUAGE = {v: k for k, v in CODE_LANGUAGE_IDS.items()}
CODE_LANGUAGE_ALIASES = {
    "python": "Python", "c": "C", "c++": "C++", "cpp": "C++",
    "c plus plus": "C++", "c#": "C#", "c sharp": "C#", "csharp": "C#",
    "java": "Java", "javascript": "JavaScript", "javascript jsx": "JavaScript",
    "typescript": "TypeScript", "typescript jsx": "TSX", "tsx": "TSX",
    "go": "Go", "golang": "Go", "rust": "Rust", "ruby": "Ruby", "php": "PHP",
    "kotlin": "Kotlin", "swift": "Swift", "scala": "Scala", "lua": "Lua",
    "shell": "Shell", "shell script": "Shell", "bash": "Shell", "zsh": "Shell",
    "dart": "Dart", "julia": "Julia", "r": "R", "haskell": "Haskell",
    "elixir": "Elixir", "erlang": "Erlang", "ocaml": "OCaml", "clojure": "Clojure",
    "objective-c": "Objective-C", "objective c": "Objective-C", "objectivec": "Objective-C",
    "zig": "Zig", "solidity": "Solidity",
}

TEXT_COLUMN_CANDIDATES = (
    "text", "content", "raw_content", "document", "body", "markdown", "article"
)
MATH_FIELDS = (
    "text", "problem", "question", "prompt", "solution", "answer", "response",
    "reasoning", "explanation", "proof"
)

BAD_PATH_PARTS = (
    "/node_modules/", "/vendor/", "/.venv/", "/venv/", "/dist/", "/build/",
    "/target/", "/__pycache__/", "/.git/", "/coverage/", "/.cache/",
)
BAD_BASENAMES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "composer.lock",
    "cargo.lock", "poetry.lock", "pipfile.lock",
}
BAD_SUFFIXES = (".min.js", ".min.css", ".map")

WIKI_SNAPSHOT_RE = re.compile(r"^(?P<date>\d{8})\.(?P<lang>[A-Za-z0-9_-]+)$")

PACK_MAGIC = b"PTPACK1\n"
PACK_RECORD = struct.Struct("<BI")  # subtype uint8, UTF-8 byte length uint32
PACK_HEADER_BYTES = len(PACK_MAGIC)


# -----------------------------------------------------------------------------
# Pure contracts/helpers
# -----------------------------------------------------------------------------

def _largest_remainder(total: int, shares: Mapping[str, float]) -> Dict[str, int]:
    if total < 0:
        raise ValueError("total must be >= 0")
    s = float(sum(shares.values()))
    if not math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"shares must sum to 1, got {s}")
    keys = list(shares)
    raw = {k: total * float(shares[k]) for k in keys}
    out = {k: int(math.floor(raw[k])) for k in keys}
    left = total - sum(out.values())
    order = sorted(keys, key=lambda k: (-(raw[k] - out[k]), keys.index(k)))
    for k in order[:left]:
        out[k] += 1
    return out


def compute_quotas(target_tokens: int) -> Dict[str, int]:
    return _largest_remainder(int(target_tokens), SOURCE_SHARES)


def stable_u64(seed: int, namespace: str, value: str) -> int:
    h = hashlib.blake2b(digest_size=8, person=b"readcount")
    h.update(str(seed).encode())
    h.update(b"\0")
    h.update(namespace.encode())
    h.update(b"\0")
    h.update(value.encode("utf-8", errors="surrogatepass"))
    return int.from_bytes(h.digest(), "big")


def deterministic_order(items: Sequence[Any], seed: int, namespace: str, key=str) -> List[Any]:
    return sorted(items, key=lambda x: (stable_u64(seed, namespace, key(x)), key(x)))


def norm_path(path: os.PathLike | str) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def acceptable_text(text: object, min_bytes: int = 32) -> bool:
    if not isinstance(text, str) or not text:
        return False
    try:
        b = text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if len(b) < min_bytes:
        return False
    if b.count(b"\x00") > max(2, len(b) // 10_000):
        return False
    return True


def is_bad_code_path(path: str) -> bool:
    p = "/" + (path or "").replace("\\", "/").lower().lstrip("/")
    base = p.rsplit("/", 1)[-1]
    return (
        any(part in p for part in BAD_PATH_PARTS)
        or base in BAD_BASENAMES
        or any(base.endswith(s) for s in BAD_SUFFIXES)
    )


def looks_minified(text: str) -> bool:
    if len(text) < 8192:
        return False
    lines = text.count("\n") + 1
    return len(text) / lines > 1200


def canonical_code_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    raw = str(language).strip()
    if raw in CODE_LANGUAGE_IDS:
        return raw
    return CODE_LANGUAGE_ALIASES.get(raw.lower())


def code_language_id(language: Optional[str]) -> Optional[int]:
    canonical = canonical_code_language(language)
    return CODE_LANGUAGE_IDS.get(canonical or "")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def adaptive_pack_target_bytes(
    source: str, remaining_tokens: int, hard_cap_bytes: int, min_pack_bytes: int,
    safety: float = 1.20, observed_bytes_per_token: Optional[float] = None,
) -> int:
    """Choose an early-count pack target without changing exact token quotas."""
    if remaining_tokens <= 0:
        return max(1, min(int(hard_cap_bytes), int(min_pack_bytes)))
    bpt = float(observed_bytes_per_token or SOURCE_BYTES_PER_TOKEN_PRIOR[source])
    estimated = int(math.ceil(remaining_tokens * bpt * float(safety)))
    return max(1, min(int(hard_cap_bytes), max(int(min_pack_bytes), estimated)))


def adaptive_prefetch_count(
    source: str, remaining_tokens: int, max_prefetch: int, spool_block_bytes: int,
    safety: float = 1.50, observed_bytes_per_token: Optional[float] = None,
) -> int:
    """Bound in-flight tasks from an estimated amount of text still required."""
    bpt = float(observed_bytes_per_token or SOURCE_BYTES_PER_TOKEN_PRIOR[source])
    need = max(1, int(math.ceil(max(0, remaining_tokens) * bpt * float(safety))))
    blocks = max(1, int(math.ceil(need / max(1, int(spool_block_bytes)))))
    return max(1 if int(max_prefetch) < 2 else 2, min(int(max_prefetch), blocks))


# -----------------------------------------------------------------------------
# Packed corpus format: source-isolated, RAM-loadable, cheap to parse
# -----------------------------------------------------------------------------

def encode_pack_record(subtype: int, text: str) -> bytes:
    if not 0 <= int(subtype) <= 255:
        raise ValueError(f"subtype out of uint8 range: {subtype}")
    payload = text.encode("utf-8")
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("one record exceeds uint32 payload length")
    return PACK_RECORD.pack(int(subtype), len(payload)) + payload


def write_pack(path: Path, records: Iterable[Tuple[int, str]], fsync: bool = False) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    count = 0
    text_bytes = 0
    subtype_records = Counter()
    with tmp.open("wb") as f:
        f.write(PACK_MAGIC)
        for subtype, text in records:
            raw = text.encode("utf-8")
            f.write(PACK_RECORD.pack(int(subtype), len(raw)))
            f.write(raw)
            count += 1
            text_bytes += len(raw)
            subtype_records[int(subtype)] += 1
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)
    return {
        "path": str(path),
        "records": count,
        "text_bytes": text_bytes,
        "file_bytes": path.stat().st_size,
        "subtype_records": {str(k): int(v) for k, v in sorted(subtype_records.items())},
    }


def _iter_pack_records_from_bytes(data: bytes) -> Iterator[Tuple[int, str]]:
    """Compatibility helper used by tests/small in-memory callers."""
    if len(data) < PACK_HEADER_BYTES or data[:PACK_HEADER_BYTES] != PACK_MAGIC:
        raise RuntimeError("invalid .pack magic/version")
    mv = memoryview(data)
    pos = PACK_HEADER_BYTES
    end = len(data)
    while pos < end:
        if end - pos < PACK_RECORD.size:
            raise RuntimeError(f"truncated .pack record header at byte {pos}")
        subtype, n = PACK_RECORD.unpack_from(mv, pos)
        pos += PACK_RECORD.size
        stop = pos + n
        if stop > end:
            raise RuntimeError(f"truncated .pack payload at byte {pos}: need {n}")
        text = bytes(mv[pos:stop]).decode("utf-8", errors="strict")
        yield int(subtype), text
        pos = stop


def _iter_pack_records_from_file(path: Path) -> Iterator[Tuple[int, str]]:
    """Stream records from disk; never materialize the whole pack in RAM."""
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(PACK_HEADER_BYTES)
        if magic != PACK_MAGIC:
            raise RuntimeError(f"invalid .pack magic/version: {path}")
        while True:
            header = f.read(PACK_RECORD.size)
            if not header:
                return
            if len(header) != PACK_RECORD.size:
                raise RuntimeError(f"truncated .pack record header: {path}")
            subtype, n = PACK_RECORD.unpack(header)
            payload = f.read(n)
            if len(payload) != n:
                raise RuntimeError(f"truncated .pack payload: {path}; need {n}, got {len(payload)}")
            yield int(subtype), payload.decode("utf-8", errors="strict")


def iter_pack_records(path: Path) -> Iterator[Tuple[int, str]]:
    yield from _iter_pack_records_from_file(Path(path))


def _process_rss_bytes() -> int:
    """Current process RSS on Linux; psutil is optional."""
    try:
        import psutil  # type: ignore
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    try:
        with open("/proc/self/status", "rt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


class MemoryGuard:
    """Cooperative RSS + reservation guard shared by all ingest threads."""
    def __init__(self, limit_bytes: int, resume_ratio: float = 0.90):
        self.limit_bytes = max(1, int(limit_bytes))
        self.resume_bytes = max(1, int(self.limit_bytes * float(resume_ratio)))
        self._cond = threading.Condition()
        self._reserved = 0
        self._warned = False

    def rss_bytes(self) -> int:
        return _process_rss_bytes()

    def wait(self, extra_bytes: int = 0) -> None:
        extra = max(0, int(extra_bytes))
        with self.reserve(extra):
            return

    @contextmanager
    def reserve(self, extra_bytes: int):
        extra = min(max(0, int(extra_bytes)), self.limit_bytes)
        while True:
            with self._cond:
                rss = self.rss_bytes()
                projected = (rss if rss > 0 else 0) + self._reserved + extra
                if rss <= 0 or projected < self.limit_bytes:
                    self._reserved += extra
                    if self._warned and rss < self.resume_bytes:
                        self._warned = False
                    break
                if not self._warned:
                    print(
                        f"  RAM guard: RSS={rss/2**30:.2f} GiB, "
                        f"reserved={self._reserved/2**30:.2f} GiB, "
                        f"limit={self.limit_bytes/2**30:.2f} GiB; pausing producers...",
                        flush=True,
                    )
                    self._warned = True
            gc.collect()
            time.sleep(0.10)
        try:
            yield
        finally:
            with self._cond:
                self._reserved = max(0, self._reserved - extra)
                self._cond.notify_all()


def count_pack(
    path: Path,
    tokenizer: Any,
    batch_size: int = 256,
    token_limit: Optional[int] = None,
    max_batch_bytes: int = 64 << 20,
    ram_guard: Optional[MemoryGuard] = None,
) -> Dict[str, Any]:
    """Stream one pack and count exact tokenizer payload tokens with bounded batches.

    `batch_size` bounds record count and `max_batch_bytes` bounds UTF-8 payload bytes.
    One exceptionally large record is still processed alone so data is never dropped.
    """
    t0 = time.perf_counter()
    path = Path(path)
    file_bytes = path.stat().st_size

    total_tokens = 0
    total_records = 0
    total_text_bytes = 0
    subtype_tokens = Counter()
    subtype_records = Counter()
    subtype_text_bytes = Counter()

    batch: List[Tuple[int, str, int]] = []
    batch_bytes = 0
    done = False

    def consume(rows: List[Tuple[int, str, int]]) -> None:
        nonlocal total_tokens, total_records, total_text_bytes, done
        if not rows or done:
            return
        raw_bytes = sum(x[2] for x in rows)
        reserve_bytes = min(raw_bytes * 4, 2 << 30)
        cm = ram_guard.reserve(reserve_bytes) if ram_guard is not None else nullcontext()
        with cm:
            texts = [text for _, text, _ in rows]
            enc = tokenizer.encode_batch(texts, add_special_tokens=False)
            for (subtype, _text, b), e in zip(rows, enc):
                n = len(e.ids)
                total_tokens += n
                total_records += 1
                total_text_bytes += b
                subtype_tokens[subtype] += n
                subtype_records[subtype] += 1
                subtype_text_bytes[subtype] += b
                if token_limit is not None and total_tokens >= token_limit:
                    done = True
                    break
            del enc, texts

    for subtype, text in _iter_pack_records_from_file(path):
        if done:
            break
        nbytes = len(text.encode("utf-8"))
        if batch and (len(batch) >= batch_size or batch_bytes + nbytes > max_batch_bytes):
            consume(batch)
            batch = []
            batch_bytes = 0
            if done:
                break
        batch.append((subtype, text, nbytes))
        batch_bytes += nbytes
        if len(batch) >= batch_size or batch_bytes >= max_batch_bytes:
            consume(batch)
            batch = []
            batch_bytes = 0
    if batch and not done:
        consume(batch)

    elapsed = time.perf_counter() - t0

    def subtype_name(i: int) -> str:
        return ID_TO_CODE_LANGUAGE.get(i, "generic" if i == 0 else f"subtype_{i}")

    return {
        "path": str(path),
        "file_bytes": file_bytes,
        "text_bytes": total_text_bytes,
        "records": total_records,
        "tokens": total_tokens,
        "load_seconds": 0.0,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (total_tokens / elapsed) if elapsed else None,
        "mib_per_second": (total_text_bytes / 2**20 / elapsed) if elapsed else None,
        "subtype_tokens": {subtype_name(k): int(v) for k, v in sorted(subtype_tokens.items())},
        "subtype_records": {subtype_name(k): int(v) for k, v in sorted(subtype_records.items())},
        "subtype_text_bytes": {subtype_name(k): int(v) for k, v in sorted(subtype_text_bytes.items())},
        "limit_reached": token_limit is not None and total_tokens >= token_limit,
        "overshoot_tokens": max(0, total_tokens - int(token_limit)) if token_limit is not None else 0,
    }


# -----------------------------------------------------------------------------
# Optional dependencies and source discovery
# -----------------------------------------------------------------------------

def _import_pyarrow_parquet():
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
        raise RuntimeError("zstandard is required for .zst inputs: pip install zstandard") from e


def _import_tokenizer_class():
    try:
        from tokenizers import Tokenizer
        return Tokenizer
    except Exception as e:
        raise RuntimeError("tokenizers is required: pip install tokenizers") from e


def resolve_tokenizer_json(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_dir():
        path = path / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_frozen_tokenizer(path: Path, threads: int):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(max(1, int(threads)))
    Tokenizer = _import_tokenizer_class()
    p = resolve_tokenizer_json(path)
    tok = Tokenizer.from_file(str(p))
    actual = {s: tok.token_to_id(s) for s in SPECIAL_TOKENS}
    expected = {s: i for i, s in enumerate(SPECIAL_TOKENS)}
    if actual != expected:
        raise RuntimeError(f"special-token ID contract mismatch: actual={actual}, expected={expected}")
    try:
        vocab = tok.get_vocab_size(with_added_tokens=True)
    except TypeError:
        vocab = tok.get_vocab_size()
    if vocab != 65_536:
        raise RuntimeError(f"expected frozen vocab_size=65536, got {vocab}")
    return tok, p


def discover_dir(explicit: Optional[Path], raw_root: Path, candidates: Sequence[str], label: str) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"{label} root missing: {p}")
        return p
    for name in candidates:
        p = raw_root / name
        if p.exists():
            return p.resolve()
    # Shallow fuzzy fallback is useful for names such as fineweb_edu_100bt.
    wanted = [x.lower().replace("-", "_") for x in candidates]
    if raw_root.is_dir():
        for p in sorted(raw_root.iterdir()):
            norm = p.name.lower().replace("-", "_")
            if any(norm.startswith(x) or x.startswith(norm) for x in wanted):
                return p.resolve()
    raise FileNotFoundError(f"cannot discover {label} below {raw_root}; pass an explicit root")


def resolve_english_wiki_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    m = WIKI_SNAPSHOT_RE.match(root.name)
    if m:
        if m.group("lang").lower() != "en":
            raise ValueError(f"Wikipedia root is not English: {root}")
        return root
    preferred = root / "snapshots" / "master"
    bases = [preferred] if preferred.is_dir() else [root]
    candidates: List[Tuple[str, str, Path]] = []
    for base in bases:
        for p in base.rglob("*"):
            if not p.is_dir():
                continue
            mm = WIKI_SNAPSHOT_RE.match(p.name)
            if mm and mm.group("lang").lower() == "en":
                candidates.append((mm.group("date"), str(p), p.resolve()))
    if not candidates:
        raise FileNotFoundError(f"no YYYYMMDD.en Wikipedia snapshot below {root}")
    candidates.sort(reverse=True)
    return candidates[0][2]


def source_roots(args: argparse.Namespace) -> Dict[str, Path]:
    raw = args.raw_root.expanduser().resolve()
    stack = args.stack_root.expanduser().resolve() if args.stack_root else DEFAULT_STACK_ROOT
    if not stack.exists():
        stack = discover_dir(None, raw, ["stack-v3", "stack_v3", "stack"], "Stack-v3")

    if args.nemotron_root is not None:
        nemotron = args.nemotron_root.expanduser().resolve()
    elif DEFAULT_NEMOTRON_ROOT.exists():
        nemotron = DEFAULT_NEMOTRON_ROOT.resolve()
    else:
        nemotron = discover_dir(
            None, raw,
            ["Nemotron-CC-Math-v1/4plus_MIND", "Nemotron-CC-Math-v1", "nemotron"],
            "Nemotron Math",
        )

    roots = {
        "general": discover_dir(
            args.fineweb_root, raw,
            ["fineweb_edu_100bt", "fineweb-edu", "fineweb_edu", "FineWeb-Edu", "fineweb"],
            "FineWeb-Edu",
        ),
        "math_nemotron": nemotron,
        "math_finemath": discover_dir(
            args.finemath_root, raw, ["finemath", "FineMath", "fine-math"], "FineMath"
        ),
        "code": stack,
        "academic": discover_dir(
            args.pes2o_root, raw, ["pes2o", "peS2o", "pes2o-v3"], "peS2o"
        ),
        "wiki": resolve_english_wiki_root(discover_dir(
            args.wiki_root, raw, ["wikipedia", "wiki", "wikipedia-en"], "Wikipedia"
        )),
    }
    for name, p in roots.items():
        if not p.exists():
            raise FileNotFoundError(f"{name} root missing: {p}")
    general = roots["general"]
    if general == DEFAULT_FINEWEB_ROOT or (general.parent.parent / "selection_manifest.json").is_file():
        from fineweb_modelscope import assert_verified_selection
        assert_verified_selection(general)
    return roots


def list_source_files(root: Path) -> List[Path]:
    suffixes = (".parquet", ".jsonl", ".jsonl.gz", ".jsonl.zst", ".zst", ".txt", ".txt.gz")
    return sorted(
        p for p in root.rglob("*")
        if p.is_file()
        and not {".hf-cache", ".cache", ".git"}.intersection(p.relative_to(root).parts)
        and any(p.name.endswith(s) for s in suffixes)
    )


def find_stack_parquets(root: Path, generation: str) -> List[Path]:
    marker = f"-{generation}-c000.snappy.parquet"
    out = sorted(p for p in root.rglob("*.parquet") if marker in p.name)
    if not out:
        raise FileNotFoundError(f"no Stack-v3 generation {generation} below {root}")
    return out


def _choose_parquet_fields(pf: Any, source_kind: str) -> Tuple[str, ...]:
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
    raise ValueError(f"no usable text column in parquet schema {schema}")


def _combine_values(values: Mapping[str, Any], fields: Sequence[str]) -> Optional[str]:
    vals = [values[k] for k in fields if isinstance(values.get(k), str) and values[k]]
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else "\n\n".join(vals)


def _json_text(obj: Any, source_kind: str) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, Mapping):
        return None
    fields = MATH_FIELDS if source_kind == "math" else TEXT_COLUMN_CANDIDATES
    vals: List[str] = []
    for k in fields:
        v = obj.get(k)
        if isinstance(v, str) and v:
            vals.append(v)
            if k == "text":
                return v
    return "\n\n".join(vals) if vals else None


# -----------------------------------------------------------------------------
# Parallel ingest tasks
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ReadTask:
    index: int
    source: str
    path: str
    kind: str  # generic_parquet_rg | generic_file | stack_parquet_rg
    row_group: int = -1
    source_kind: str = "general"


@dataclass
class TaskResult:
    index: int
    task_key: str
    spool_path: str
    packed_bytes: int
    records: int
    text_bytes: int
    subtype_records: Dict[int, int]
    subtype_text_bytes: Dict[int, int]
    chunk_index: int = 0
    final: bool = True
    error: Optional[str] = None


def _task_key(task: ReadTask) -> str:
    return f"{task.path}#rg={task.row_group}#{task.kind}"


def _raw_task_fingerprint(tasks: Sequence[ReadTask]) -> str:
    """Fingerprint ordered raw work including file identity, size and mtime.

    The previous checkpoint only remembered the prior path/row-group key.  That
    detects reordering, but not an in-place replacement of a raw shard.  A
    rolling fingerprint keeps resume fail-closed without hashing multi-TB input.
    """
    h = hashlib.sha256()
    stat_cache: Dict[str, Tuple[int, int]] = {}
    for task in tasks:
        path = str(Path(task.path).resolve())
        if path not in stat_cache:
            st = Path(path).stat()
            stat_cache[path] = (int(st.st_size), int(st.st_mtime_ns))
        size, mtime_ns = stat_cache[path]
        row = f"{int(task.index)}\0{task.source}\0{path}\0{task.kind}\0{int(task.row_group)}\0{size}\0{mtime_ns}\n"
        h.update(row.encode("utf-8", errors="surrogatepass"))
    return h.hexdigest()


def build_generic_tasks(root: Path, source: str, source_kind: str, seed: int) -> List[ReadTask]:
    pq = _import_pyarrow_parquet()
    files = deterministic_order(list_source_files(root), seed, f"files:{source}", key=lambda p: str(p))
    tasks: List[ReadTask] = []
    idx = 0
    for path in files:
        if path.name.endswith(".parquet"):
            pf = pq.ParquetFile(path)
            for rg in range(pf.num_row_groups):
                tasks.append(ReadTask(idx, source, str(path), "generic_parquet_rg", rg, source_kind))
                idx += 1
        else:
            tasks.append(ReadTask(idx, source, str(path), "generic_file", -1, source_kind))
            idx += 1
    return tasks


def build_stack_tasks(root: Path, generation: str, seed: int) -> List[ReadTask]:
    pq = _import_pyarrow_parquet()
    files = deterministic_order(find_stack_parquets(root, generation), seed, "stack-files", key=lambda p: str(p))
    tasks: List[ReadTask] = []
    idx = 0
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            tasks.append(ReadTask(idx, "code", str(path), "stack_parquet_rg", rg, "code"))
            idx += 1
    return tasks


class _SpoolWriter:
    def __init__(self, task: ReadTask, spool_dir: Path):
        spool_dir.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.path = spool_dir / f"task-{task.index:08d}-{os.getpid()}-{threading.get_ident()}.bin"
        self.f = self.path.open("wb")
        self.packed_bytes = 0
        self.records = 0
        self.text_bytes = 0
        self.sub_records = Counter()
        self.sub_bytes = Counter()

    def add(self, subtype: int, text: str, *, min_bytes: int = 32,
            max_payload_bytes: Optional[int] = None, reject_minified: bool = False) -> None:
        if not isinstance(text, str) or not text:
            return
        if reject_minified and looks_minified(text):
            return
        try:
            payload = text.encode("utf-8")
        except UnicodeEncodeError:
            return
        n = len(payload)
        if n < min_bytes or n > 0xFFFFFFFF:
            return
        if max_payload_bytes is not None and n > max_payload_bytes:
            return
        if payload.count(b"\x00") > max(2, n // 10_000):
            return
        header = PACK_RECORD.pack(int(subtype), n)
        self.f.write(header)
        self.f.write(payload)
        self.packed_bytes += len(header) + n
        self.records += 1
        self.text_bytes += n
        self.sub_records[int(subtype)] += 1
        self.sub_bytes[int(subtype)] += n

    def finish(self) -> TaskResult:
        self.f.close()
        if self.records == 0:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            spool_path = ""
        else:
            spool_path = str(self.path)
        return TaskResult(
            self.task.index, _task_key(self.task), spool_path, self.packed_bytes,
            self.records, self.text_bytes, dict(self.sub_records), dict(self.sub_bytes),
        )

    def abort(self) -> None:
        try:
            self.f.close()
        finally:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _iter_text_lines(path: Path, source_kind: str, io_sem: threading.Semaphore) -> Iterator[str]:
    """Stream compressed line corpora without holding I/O permits across ``yield``.

    A worker may block while publishing a full spool chunk.  Therefore no shared
    semaphore or memory reservation may remain held when control is yielded back to
    the caller.  Lines are read in modest batches to amortize semaphore traffic.
    """
    name = path.name
    raw = None
    if name.endswith(".gz"):
        cm = gzip.open(path, "rt", encoding="utf-8", errors="strict")
    elif name.endswith(".zst"):
        zstd = _import_zstandard()
        raw = path.open("rb")
        reader = zstd.ZstdDecompressor().stream_reader(raw)
        cm = io.TextIOWrapper(reader, encoding="utf-8", errors="strict")
    else:
        cm = path.open("rt", encoding="utf-8", errors="strict")
    try:
        eof = False
        while not eof:
            lines: List[str] = []
            chars = 0
            with io_sem:
                for _ in range(256):
                    line = cm.readline()
                    if not line:
                        eof = True
                        break
                    lines.append(line)
                    chars += len(line)
                    if chars >= (4 << 20):
                        break
            for line in lines:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = _json_text(obj, source_kind)
                except json.JSONDecodeError:
                    text = line if source_kind != "math" else None
                if text:
                    yield text
    finally:
        cm.close()
        if raw is not None:
            try:
                raw.close()
            except Exception:
                pass


def process_task(
    task: ReadTask,
    io_sem: threading.Semaphore,
    max_code_file_bytes: int,
    spool_dir: Path,
    parquet_batch_rows: int,
    ram_guard: MemoryGuard,
) -> TaskResult:
    """Read one logical task in bounded batches and spool its packed bytes to SSD."""
    path = Path(task.path)
    out = _SpoolWriter(task, spool_dir)
    try:
        if task.kind == "generic_parquet_rg":
            pq = _import_pyarrow_parquet()
            pf = pq.ParquetFile(path)
            fields = _choose_parquet_fields(pf, task.source_kind)
            batches = pf.iter_batches(
                batch_size=max(1, int(parquet_batch_rows)),
                row_groups=[task.row_group], columns=list(fields), use_threads=True,
            )
            while True:
                try:
                    with ram_guard.reserve(256 << 20):
                        with io_sem:
                            batch = next(batches)
                        data = batch.to_pydict()
                        for i in range(batch.num_rows):
                            text = _combine_values({k: data[k][i] for k in fields}, fields)
                            if text:
                                out.add(0, text)
                        del data, batch
                except StopIteration:
                    break
            return out.finish()

        if task.kind == "generic_file":
            name = path.name
            if name.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst", ".zst")):
                for text in _iter_text_lines(path, task.source_kind, io_sem):
                    ram_guard.wait(extra_bytes=64 << 20)
                    out.add(0, text)
                return out.finish()
            # Plain .txt/.txt.gz are one logical document. These are normally modest;
            # the RAM guard prevents starting the read when process memory is near cap.
            with ram_guard.reserve(min(path.stat().st_size * 4, 2 << 30)):
                with io_sem:
                    if name.endswith(".txt.gz"):
                        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
                            text = f.read()
                    else:
                        text = path.read_text(encoding="utf-8", errors="strict")
            out.add(0, text)
            del text
            return out.finish()

        if task.kind == "stack_parquet_rg":
            pq = _import_pyarrow_parquet()
            pf = pq.ParquetFile(path)
            batches = pf.iter_batches(
                batch_size=max(1, int(parquet_batch_rows)), row_groups=[task.row_group],
                columns=["repo_path", "files"], use_threads=True,
            )
            while True:
                try:
                    with ram_guard.reserve(512 << 20):
                        with io_sem:
                            batch = next(batches)
                        rows = batch.to_pylist()
                        for row in rows:
                            for f in row.get("files") or []:
                                if not isinstance(f, Mapping) or f.get("is_vendor"):
                                    continue
                                file_path = str(f.get("file_path") or "")
                                if not file_path or is_bad_code_path(file_path):
                                    continue
                                subtype = code_language_id(f.get("language"))
                                if subtype is None:
                                    continue
                                text = f.get("content")
                                if isinstance(text, str):
                                    out.add(
                                        subtype, text, min_bytes=8,
                                        max_payload_bytes=max_code_file_bytes, reject_minified=True,
                                    )
                        del rows, batch
                except StopIteration:
                    break
            return out.finish()

        raise ValueError(f"unknown task kind {task.kind}")
    except Exception:
        out.abort()
        raise



class _StreamingSpoolWriter:
    """Roll one worker's output into small SSD blocks and emit them immediately."""
    def __init__(
        self,
        task: ReadTask,
        spool_dir: Path,
        block_bytes: int,
        emit,
        stop_event: threading.Event,
    ):
        self.task = task
        self.spool_dir = spool_dir
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.block_bytes = max(1, int(block_bytes))
        self.emit = emit
        self.stop_event = stop_event
        self.chunk_index = 0
        self.f = None
        self.path: Optional[Path] = None
        self.packed_bytes = 0
        self.records = 0
        self.text_bytes = 0
        self.sub_records = Counter()
        self.sub_bytes = Counter()
        self.emitted_any = False

    def _open(self) -> None:
        if self.f is not None:
            return
        self.path = self.spool_dir / (
            f"task-{self.task.index:08d}-chunk-{self.chunk_index:06d}-"
            f"{os.getpid()}-{threading.get_ident()}.bin"
        )
        self.f = self.path.open("wb")

    def _reset(self) -> None:
        self.f = None
        self.path = None
        self.packed_bytes = 0
        self.records = 0
        self.text_bytes = 0
        self.sub_records.clear()
        self.sub_bytes.clear()

    def _emit_current(self, final: bool) -> bool:
        if self.f is None or self.records == 0:
            if final:
                marker = TaskResult(
                    self.task.index, _task_key(self.task), "", 0, 0, 0, {}, {},
                    chunk_index=self.chunk_index, final=True,
                )
                return bool(self.emit(marker))
            return True
        assert self.path is not None
        self.f.close()
        event = TaskResult(
            self.task.index,
            _task_key(self.task),
            str(self.path),
            self.packed_bytes,
            self.records,
            self.text_bytes,
            dict(self.sub_records),
            dict(self.sub_bytes),
            chunk_index=self.chunk_index,
            final=final,
        )
        ok = bool(self.emit(event))
        if not ok:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        else:
            self.emitted_any = True
        self.chunk_index += 1
        self._reset()
        return ok

    def add(
        self,
        subtype: int,
        text: str,
        *,
        min_bytes: int = 32,
        max_payload_bytes: Optional[int] = None,
        reject_minified: bool = False,
    ) -> bool:
        if self.stop_event.is_set():
            return False
        if not isinstance(text, str) or not text:
            return True
        if reject_minified and looks_minified(text):
            return True
        try:
            payload = text.encode("utf-8")
        except UnicodeEncodeError:
            return True
        n = len(payload)
        if n < min_bytes or n > 0xFFFFFFFF:
            return True
        if max_payload_bytes is not None and n > max_payload_bytes:
            return True
        if payload.count(b"\x00") > max(2, n // 10_000):
            return True
        record_bytes = PACK_RECORD.size + n
        if self.records and self.packed_bytes + record_bytes > self.block_bytes:
            if not self._emit_current(final=False):
                return False
        self._open()
        assert self.f is not None
        self.f.write(PACK_RECORD.pack(int(subtype), n))
        self.f.write(payload)
        self.packed_bytes += record_bytes
        self.records += 1
        self.text_bytes += n
        self.sub_records[int(subtype)] += 1
        self.sub_bytes[int(subtype)] += n
        return True

    def finish(self) -> None:
        if self.stop_event.is_set():
            self.abort_current()
            return
        self._emit_current(final=True)

    def abort_current(self) -> None:
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass
        if self.path is not None:
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
        self._reset()


def _queue_emit(q: "queue.Queue[TaskResult]", stop_event: threading.Event, event: TaskResult) -> bool:
    while not stop_event.is_set():
        try:
            q.put(event, timeout=0.10)
            return True
        except queue.Full:
            continue
    return False


def process_task_stream(
    task: ReadTask,
    out_queue: "queue.Queue[TaskResult]",
    stop_event: threading.Event,
    io_sem: threading.Semaphore,
    max_code_file_bytes: int,
    spool_dir: Path,
    parquet_batch_rows: int,
    ram_guard: MemoryGuard,
    spool_block_bytes: int,
) -> None:
    """Process a task incrementally; emit bounded SSD chunks instead of one huge spool."""
    path = Path(task.path)
    emit = lambda event: _queue_emit(out_queue, stop_event, event)
    out = _StreamingSpoolWriter(task, spool_dir, spool_block_bytes, emit, stop_event)
    try:
        if task.kind == "generic_parquet_rg":
            pq = _import_pyarrow_parquet()
            pf = pq.ParquetFile(path)
            fields = _choose_parquet_fields(pf, task.source_kind)
            batches = iter(pf.iter_batches(
                batch_size=max(1, int(parquet_batch_rows)),
                row_groups=[task.row_group], columns=list(fields), use_threads=True,
            ))
            while not stop_event.is_set():
                try:
                    with ram_guard.reserve(256 << 20):
                        with io_sem:
                            batch = next(batches)
                        data = batch.to_pydict()
                except StopIteration:
                    break
                should_stop = False
                try:
                    for i in range(batch.num_rows):
                        if stop_event.is_set():
                            should_stop = True
                            break
                        text = _combine_values({k: data[k][i] for k in fields}, fields)
                        if text and not out.add(0, text):
                            should_stop = True
                            break
                finally:
                    del data, batch
                if should_stop:
                    break
            out.finish()
            return

        if task.kind == "generic_file":
            name = path.name
            if name.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst", ".zst")):
                for text in _iter_text_lines(path, task.source_kind, io_sem):
                    if stop_event.is_set():
                        break
                    if not out.add(0, text):
                        break
                out.finish()
                return
            with ram_guard.reserve(min(path.stat().st_size * 4, 2 << 30)):
                with io_sem:
                    if name.endswith(".txt.gz"):
                        with gzip.open(path, "rt", encoding="utf-8", errors="strict") as f:
                            text = f.read()
                    else:
                        text = path.read_text(encoding="utf-8", errors="strict")
            # Never hold a cooperative memory reservation while a bounded output
            # queue can block; RSS still accounts for `text` after the reservation.
            if not stop_event.is_set():
                out.add(0, text)
            del text
            out.finish()
            return

        if task.kind == "stack_parquet_rg":
            pq = _import_pyarrow_parquet()
            pf = pq.ParquetFile(path)
            batches = iter(pf.iter_batches(
                batch_size=max(1, int(parquet_batch_rows)), row_groups=[task.row_group],
                columns=["repo_path", "files"], use_threads=True,
            ))
            while not stop_event.is_set():
                try:
                    with ram_guard.reserve(512 << 20):
                        with io_sem:
                            batch = next(batches)
                        rows = batch.to_pylist()
                except StopIteration:
                    break
                should_stop = False
                try:
                    for row in rows:
                        if stop_event.is_set():
                            should_stop = True
                            break
                        for f in row.get("files") or []:
                            if stop_event.is_set():
                                should_stop = True
                                break
                            if not isinstance(f, Mapping) or f.get("is_vendor"):
                                continue
                            file_path = str(f.get("file_path") or "")
                            if not file_path or is_bad_code_path(file_path):
                                continue
                            subtype = code_language_id(f.get("language"))
                            if subtype is None:
                                continue
                            text = f.get("content")
                            if isinstance(text, str) and not out.add(
                                subtype, text, min_bytes=8,
                                max_payload_bytes=max_code_file_bytes, reject_minified=True,
                            ):
                                should_stop = True
                                break
                finally:
                    del rows, batch
                if should_stop:
                    break
            out.finish()
            return

        raise ValueError(f"unknown task kind {task.kind}")
    except Exception as e:
        out.abort_current()
        if not stop_event.is_set():
            err = TaskResult(
                task.index, _task_key(task), "", 0, 0, 0, {}, {},
                chunk_index=out.chunk_index, final=True,
                error=f"{type(e).__name__}: {e}",
            )
            _queue_emit(out_queue, stop_event, err)
        raise


@dataclass
class _TaskStream:
    task: ReadTask
    q: Any
    future: Future


def ordered_stream_results(
    tasks: Sequence[ReadTask],
    fn,
    workers: int,
    prefetch: int,
    stop_event: threading.Event,
    queue_depth: int = 2,
    stall_timeout_s: float = DEFAULT_STALL_TIMEOUT_S,
) -> Iterator[TaskResult]:
    """Concurrent execution with deterministic order, bounded queues and a stall guard."""
    workers = max(1, int(workers))
    prefetch = max(1, min(int(prefetch), workers))
    queue_depth = max(1, int(queue_depth))
    stall_timeout_s = max(0.0, float(stall_timeout_s))
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest-v6")
    pending: deque[_TaskStream] = deque()
    it = iter(tasks)
    last_progress = time.monotonic()

    def submit_one() -> bool:
        try:
            task = next(it)
        except StopIteration:
            return False
        q: "queue.Queue[TaskResult]" = queue.Queue(maxsize=queue_depth)
        fut = ex.submit(fn, task, q, stop_event)
        pending.append(_TaskStream(task, q, fut))
        return True

    try:
        for _ in range(prefetch):
            if not submit_one():
                break
        while pending and not stop_event.is_set():
            stream = pending[0]
            while not stop_event.is_set():
                try:
                    item = stream.q.get(timeout=0.10)
                except queue.Empty:
                    if stream.future.done():
                        stream.future.result()
                    if stall_timeout_s and time.monotonic() - last_progress >= stall_timeout_s:
                        detail = []
                        for x in list(pending)[:8]:
                            detail.append(
                                f"task={x.task.index} q={x.q.qsize()}/{queue_depth} "
                                f"done={x.future.done()} key={_task_key(x.task)}"
                            )
                        stop_event.set()
                        raise RuntimeError(
                            f"ordered ingest stalled for {stall_timeout_s:.1f}s; "
                            f"head task={stream.task.index} key={_task_key(stream.task)}; "
                            f"pending={len(pending)}; " + " | ".join(detail)
                        )
                    continue
                last_progress = time.monotonic()
                if item.error:
                    stop_event.set()
                    try:
                        stream.future.result()
                    finally:
                        raise RuntimeError(f"task {item.index} failed: {item.error}")
                yield item
                if item.final:
                    break
            if stop_event.is_set():
                break
            stream.future.result()
            pending.popleft()
            last_progress = time.monotonic()
            submit_one()
    finally:
        stop_event.set()
        for stream in pending:
            stream.future.cancel()
        ex.shutdown(wait=True, cancel_futures=True)

def bounded_ordered_results(
    tasks: Sequence[ReadTask],
    fn,
    workers: int,
    prefetch: int,
) -> Iterator[TaskResult]:
    """Run tasks concurrently but yield strictly in task order with bounded futures."""
    workers = max(1, int(workers))
    prefetch = max(1, int(prefetch))
    ex = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ingest")
    pending: deque[Future] = deque()
    it = iter(tasks)
    try:
        for _ in range(prefetch):
            try:
                pending.append(ex.submit(fn, next(it)))
            except StopIteration:
                break
        while pending:
            fut = pending.popleft()
            yield fut.result()
            try:
                pending.append(ex.submit(fn, next(it)))
            except StopIteration:
                pass
    finally:
        for fut in pending:
            fut.cancel()
        ex.shutdown(wait=True, cancel_futures=True)


# -----------------------------------------------------------------------------
# Source runner: ingest one big shard -> RAM count -> repeat until exact quota met
# -----------------------------------------------------------------------------

class ShardAccumulator:
    """Record-aware pack writer with a real hard file-size cap.

    Worker chunks contain raw PACK_RECORD records without PACK_MAGIC.  We stream
    those records into final packs one-by-one so a large Parquet row group can no
    longer force a multi-GiB pack.  The adaptive target may be smaller than the
    hard cap; one record may exceed the adaptive target, but never the hard cap.
    """
    def __init__(
        self,
        source: str,
        out_dir: Path,
        target_bytes: int,
        start_index: int = 0,
        fsync: bool = False,
        hard_cap_bytes: Optional[int] = None,
    ):
        self.source = source
        self.out_dir = out_dir
        self.target_bytes = max(PACK_HEADER_BYTES + PACK_RECORD.size + 1, int(target_bytes))
        self.hard_cap_bytes = max(
            self.target_bytes,
            int(hard_cap_bytes) if hard_cap_bytes is not None else self.target_bytes,
        )
        self.index = int(start_index)
        self.fsync = fsync
        self.f = None
        self.tmp_path: Optional[Path] = None
        self.file_bytes = 0
        self.records = 0
        self.text_bytes = 0
        self.subtype_records = Counter()
        self.subtype_text_bytes = Counter()
        self._completed: List[Dict[str, Any]] = []

    def set_target_bytes(self, value: int) -> None:
        self.target_bytes = min(
            self.hard_cap_bytes,
            max(PACK_HEADER_BYTES + PACK_RECORD.size + 1, int(value)),
        )

    def _ensure_open(self) -> None:
        if self.f is not None:
            return
        d = self.out_dir / self.source
        d.mkdir(parents=True, exist_ok=True)
        final = d / f"{self.source}-{self.index:05d}.pack"
        self.tmp_path = final.with_name(final.name + f".tmp.{os.getpid()}")
        self.f = self.tmp_path.open("wb")
        self.f.write(PACK_MAGIC)
        self.file_bytes = PACK_HEADER_BYTES

    @property
    def has_data(self) -> bool:
        return self.records > 0

    @property
    def full(self) -> bool:
        return self.file_bytes >= self.target_bytes

    def _flush_current(self) -> Optional[Dict[str, Any]]:
        if not self.has_data:
            return None
        assert self.f is not None and self.tmp_path is not None
        final = self.out_dir / self.source / f"{self.source}-{self.index:05d}.pack"
        self.f.flush()
        if self.fsync:
            os.fsync(self.f.fileno())
        self.f.close()
        os.replace(self.tmp_path, final)
        meta = {
            "path": str(final),
            "records": self.records,
            "text_bytes": self.text_bytes,
            "file_bytes": final.stat().st_size,
            "subtype_records": {str(k): int(v) for k, v in sorted(self.subtype_records.items())},
            "subtype_text_bytes": {str(k): int(v) for k, v in sorted(self.subtype_text_bytes.items())},
        }
        if meta["file_bytes"] > self.hard_cap_bytes:
            raise RuntimeError(
                f"hard pack cap violated: {meta['file_bytes']} > {self.hard_cap_bytes}: {final}"
            )
        meta.update(_strict_pack_integrity(final, self.source, self.hard_cap_bytes))
        self.index += 1
        self.f = None
        self.tmp_path = None
        self.file_bytes = 0
        self.records = 0
        self.text_bytes = 0
        self.subtype_records.clear()
        self.subtype_text_bytes.clear()
        return meta

    def _append_raw_record(self, subtype: int, payload: bytes) -> None:
        record_size = PACK_RECORD.size + len(payload)
        if PACK_HEADER_BYTES + record_size > self.hard_cap_bytes:
            raise ValueError(
                f"single record ({record_size} bytes + header) exceeds hard pack cap "
                f"{self.hard_cap_bytes}; increase --pack-mib"
            )
        # Flush before the next record if it would cross the current adaptive target.
        # An empty pack may accept one record larger than the adaptive target as long
        # as that record stays below the hard --pack-mib cap.
        if self.has_data and self.file_bytes + record_size > self.target_bytes:
            meta = self._flush_current()
            if meta:
                self._completed.append(meta)
        self._ensure_open()
        assert self.f is not None
        self.f.write(PACK_RECORD.pack(int(subtype), len(payload)))
        self.f.write(payload)
        self.file_bytes += record_size
        self.records += 1
        self.text_bytes += len(payload)
        self.subtype_records[int(subtype)] += 1
        self.subtype_text_bytes[int(subtype)] += len(payload)
        if self.file_bytes >= self.target_bytes:
            meta = self._flush_current()
            if meta:
                self._completed.append(meta)

    def _consume_spool(self, path: Path) -> None:
        with path.open("rb") as src:
            while True:
                header = src.read(PACK_RECORD.size)
                if not header:
                    break
                if len(header) != PACK_RECORD.size:
                    raise RuntimeError(f"truncated worker spool header: {path}")
                subtype, n = PACK_RECORD.unpack(header)
                payload = src.read(n)
                if len(payload) != n:
                    raise RuntimeError(f"truncated worker spool payload: {path}")
                self._append_raw_record(int(subtype), payload)

    def add(self, r: TaskResult) -> None:
        if not r.records:
            if r.spool_path:
                try:
                    Path(r.spool_path).unlink()
                except FileNotFoundError:
                    pass
            return
        if not r.spool_path:
            raise RuntimeError(f"task {r.index} has records but no spool path")
        spool = Path(r.spool_path)
        try:
            self._consume_spool(spool)
        finally:
            try:
                spool.unlink()
            except FileNotFoundError:
                pass

    def pop_completed(self) -> List[Dict[str, Any]]:
        out = self._completed
        self._completed = []
        return out

    def flush(self) -> Optional[Dict[str, Any]]:
        return self._flush_current()

    def abort(self) -> None:
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass
        if self.tmp_path is not None:
            try:
                self.tmp_path.unlink()
            except FileNotFoundError:
                pass
        self.f = None
        self.tmp_path = None
        self.file_bytes = 0
        self.records = 0
        self.text_bytes = 0
        self.subtype_records.clear()
        self.subtype_text_bytes.clear()


def _sha256_file(path: Path | str, chunk_bytes: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            b = f.read(chunk_bytes)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _strict_pack_integrity(path: Path | str, source: str, hard_cap_bytes: int) -> Dict[str, Any]:
    """Validate the exact on-disk stage contract. No legacy/headerless compatibility."""
    path = Path(path)
    size = path.stat().st_size
    if size > int(hard_cap_bytes):
        raise RuntimeError(f"strict pack hard cap violated: {path}: {size} > {int(hard_cap_bytes)}")
    if source == "code":
        with path.open("rb") as f:
            magic = f.read(CODE_PACK_HEADER_BYTES)
            if magic != CODE_PACK_MAGIC:
                raise RuntimeError(f"strict PTCODE5 magic mismatch: {path}: {magic!r}")
            frames = 0
            while True:
                off = f.tell()
                h = f.read(CODE_FRAME.size)
                if not h:
                    break
                if len(h) != CODE_FRAME.size:
                    raise RuntimeError(f"strict PTCODE5 truncated frame header: {path}@{off}")
                kind, subtype, flags, meta_n, text_n = CODE_FRAME.unpack(h)
                if kind not in {CODE_KIND_REPO, CODE_KIND_FILE}:
                    raise RuntimeError(f"strict PTCODE5 invalid frame kind={kind}: {path}@{off}")
                end = f.tell() + int(meta_n) + int(text_n)
                if end > size:
                    raise RuntimeError(f"strict PTCODE5 truncated frame: {path}@{off}")
                f.seek(int(meta_n) + int(text_n), os.SEEK_CUR)
                frames += 1
        fmt = "PTCODE5"
        units = frames
    else:
        with path.open("rb") as f:
            magic = f.read(PACK_HEADER_BYTES)
            if magic != PACK_MAGIC:
                raise RuntimeError(f"strict PTPACK1 magic mismatch: {path}: {magic!r}")
            records = 0
            while True:
                off = f.tell()
                h = f.read(PACK_RECORD.size)
                if not h:
                    break
                if len(h) != PACK_RECORD.size:
                    raise RuntimeError(f"strict PTPACK1 truncated record header: {path}@{off}")
                subtype, n = PACK_RECORD.unpack(h)
                end = f.tell() + int(n)
                if end > size:
                    raise RuntimeError(f"strict PTPACK1 truncated payload: {path}@{off}")
                f.seek(int(n), os.SEEK_CUR)
                records += 1
        fmt = "PTPACK1"
        units = records
    return {
        "verified_format": fmt,
        "verified_units": int(units),
        "sha256": _sha256_file(path),
        "file_bytes": int(size),
    }


def _empty_source_state(quota: int) -> Dict[str, Any]:
    return {
        "target_tokens": int(quota),
        "tokens": 0,
        "records": 0,
        "text_bytes": 0,
        "next_task_index": 0,
        "next_shard_index": 0,
        "shards": [],
        "subtype_tokens": {},
        "subtype_records": {},
        "complete": False,
        "exhausted": False,
        "available_tokens": 0,
        "checkpoint_version": CHECKPOINT_VERSION,
        "checkpoint_mode": "task_atomic",
        "last_completed_task_key": "",
        "raw_task_fingerprint": "",
        "raw_task_count": 0,
    }


def _legacy_checkpoint_has_progress(state: Mapping[str, Any]) -> bool:
    return bool(
        int(state.get("tokens") or 0)
        or int(state.get("next_task_index") or 0)
        or int(state.get("next_shard_index") or 0)
        or state.get("shards")
    )


def _validate_or_upgrade_checkpoint_state(
    source: str,
    state: Dict[str, Any],
    repair_requested: bool = False,
    tasks: Optional[Sequence[ReadTask]] = None,
) -> str:
    """Validate a checkpoint and safely migrate legacy task-boundary states.

    Legacy Stage versions wrote pack checkpoints in the middle of a raw task.
    Such a checkpoint is ambiguous *unless* all raw tasks were already consumed:
    next_task_index == len(tasks) can only be reached after a final task marker.
    That exhausted-at-boundary case is safe to migrate without deleting data.
    """
    ver = int(state.get("checkpoint_version") or 0)
    start = int(state.get("next_task_index") or 0)
    task_count = len(tasks) if tasks is not None else None

    if task_count is not None and start > task_count:
        raise RuntimeError(
            f"{source}: checkpoint next_task_index={start} exceeds current raw task count={task_count}; "
            "raw inputs/order changed; refusing unsafe resume"
        )

    if ver >= CHECKPOINT_VERSION:
        if (not bool(state.get("complete")) and start > 0
                and not str(state.get("last_completed_task_key") or "")):
            if repair_requested:
                return "repair_required"
            raise RuntimeError(
                f"{source}: checkpoint v{ver} is missing last_completed_task_key at "
                f"next_task_index={start}; refusing unsafe resume"
            )
        if task_count is not None and not bool(state.get("complete")) and start == task_count and task_count > 0:
            state["exhausted"] = True
            state["available_tokens"] = int(state.get("tokens") or 0)
            state["raw_task_count"] = task_count
        return "current"

    if int(state.get("tokens") or 0) >= int(state.get("target_tokens") or 0) > 0:
        state["complete"] = True

    if bool(state.get("complete")) or not _legacy_checkpoint_has_progress(state):
        state["checkpoint_version"] = CHECKPOINT_VERSION
        state["checkpoint_mode"] = "task_atomic"
        state.setdefault("last_completed_task_key", "")
        state.setdefault("exhausted", False)
        state.setdefault("available_tokens", int(state.get("tokens") or 0))
        state.setdefault("raw_task_fingerprint", "")
        state.setdefault("raw_task_count", 0)
        return "upgraded"

    # Old code advanced next_task_index to task+1 only on a final marker.  Thus an
    # incomplete source that has consumed *all* current raw tasks is a known-safe
    # boundary even though its 1.25x staging buffer target was not met.
    if tasks is not None and task_count is not None and task_count > 0 and start == task_count:
        state["checkpoint_version"] = CHECKPOINT_VERSION
        state["checkpoint_mode"] = "task_atomic"
        state["last_completed_task_key"] = _task_key(tasks[-1])
        state["exhausted"] = True
        state["available_tokens"] = int(state.get("tokens") or 0)
        state["raw_task_count"] = task_count
        state["raw_task_fingerprint"] = _raw_task_fingerprint(tasks)
        return "upgraded_exhausted"

    if repair_requested:
        return "repair_required"
    raise RuntimeError(
        f"{source}: legacy incomplete checkpoint is ambiguous because older Stage versions "
        f"committed packs in the middle of a task. Refusing a duplicate-prone resume. "
        f"Set REPAIR_INCOMPLETE_SOURCE={source} for one run to reset only this source."
    )


def _purge_source_files(stage_dir: Path, source: str) -> None:
    shutil.rmtree(Path(stage_dir) / "records" / source, ignore_errors=True)
    shutil.rmtree(Path(stage_dir) / ".spool" / source, ignore_errors=True)


def _repair_legacy_incomplete_source(
    stage_dir: Path, manifest: Dict[str, Any], source: str, *, delete_files: bool = True,
) -> None:
    if source not in manifest.get("sources", {}):
        raise ValueError(f"unknown source for repair: {source}")
    state = manifest["sources"][source]
    if bool(state.get("complete")):
        raise RuntimeError(f"refusing to reset completed source {source}")
    quota = int(state["target_tokens"])
    manifest["sources"][source] = _empty_source_state(quota)
    manifest["complete"] = False
    if delete_files:
        _purge_source_files(stage_dir, source)


def _cleanup_uncommitted_source_files(stage_dir: Path, source: str, state: Mapping[str, Any]) -> None:
    root = Path(stage_dir) / "records" / source
    if not root.exists():
        return
    listed = {str(Path(m["path"]).resolve()) for m in state.get("shards", []) if m.get("path")}
    next_index = int(state.get("next_shard_index") or 0)
    pat = re.compile(rf"^{re.escape(source)}-(\d+)\.pack$")
    for p in root.iterdir():
        if ".tmp." in p.name:
            p.unlink(missing_ok=True)
            continue
        m = pat.match(p.name)
        if not m:
            continue
        rp = str(p.resolve())
        idx = int(m.group(1))
        if rp in listed:
            continue
        if idx >= next_index:
            p.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"{source}: unlisted pack exists before checkpoint boundary: {p}")


def _verify_task_resume_boundary(tasks: Sequence[ReadTask], state: Mapping[str, Any], source: str) -> None:
    start = int(state.get("next_task_index") or 0)
    if start <= 0:
        return
    last_key = str(state.get("last_completed_task_key") or "")
    if not last_key:
        return
    prev = next((t for t in tasks if int(t.index) == start - 1), None)
    if prev is None or _task_key(prev) != last_key:
        raise RuntimeError(
            f"{source}: raw task order changed since the last checkpoint; "
            f"expected previous task key {last_key!r}. Refusing to skip or duplicate raw data."
        )


def _record_or_verify_task_layout(
    tasks: Sequence[ReadTask], state: Dict[str, Any], source: str
) -> bool:
    """Return True when state changed; reject in-place raw data mutations on resume."""
    fp = _raw_task_fingerprint(tasks)
    count = len(tasks)
    old_fp = str(state.get("raw_task_fingerprint") or "")
    old_count = int(state.get("raw_task_count") or 0)
    if old_fp and old_fp != fp:
        raise RuntimeError(
            f"{source}: raw input fingerprint changed since checkpoint; refusing unsafe resume "
            f"(task_count was {old_count}, now {count})"
        )
    changed = (old_fp != fp) or (old_count != count)
    state["raw_task_fingerprint"] = fp
    state["raw_task_count"] = count
    return changed


def _initial_manifest(args: argparse.Namespace, roots: Mapping[str, Path], tokenizer_path: Path) -> Dict[str, Any]:
    quotas = compute_quotas(args.target_tokens)
    return {
        "format_version": 5,
        "strict_pack_contract": STRICT_PACK_CONTRACT,
        "code_pack_format": "PTCODE5",
        "script_revision": SCRIPT_REVISION,
        "created_at_unix": time.time(),
        "target_tokens": args.target_tokens,
        "quotas": quotas,
        "seed": args.seed,
        "workers": args.workers,
        "io_readers": args.io_readers,
        "pack_mib": args.pack_mib,
        "min_pack_mib": args.min_pack_mib,
        "pack_safety": args.pack_safety,
        "spool_block_mib": args.spool_block_mib,
        "prefetch_safety": args.prefetch_safety,
        "task_queue_depth": args.task_queue_depth,
        "ram_limit_gib": args.ram_limit_gib,
        "count_batch_mib": args.count_batch_mib,
        "parquet_batch_rows": args.parquet_batch_rows,
        "max_code_file_mib": args.max_code_file_mib,
        "stall_timeout_s": args.stall_timeout_s,
        "tokenizer": str(tokenizer_path),
        "tokenizer_sha256": _sha256_file(tokenizer_path),
        "roots": {k: str(v) for k, v in roots.items()},
        "base_stage_dir": str(args.base_stage_dir) if args.base_stage_dir else None,
        "base_stage_manifest_sha256": str(getattr(args, "_base_stage_manifest_sha256", "") or ""),
        "stack_generation": args.stack_generation,
        "code_languages": list(CODE_LANGUAGES),
        "sources": {s: _empty_source_state(quotas[s]) for s in SOURCE_ORDER},
        "complete": False,
    }


def _merge_counter_dict(dst: Dict[str, int], src: Mapping[str, int]) -> None:
    for k, v in src.items():
        dst[k] = int(dst.get(k, 0)) + int(v)


def run_source(
    args: argparse.Namespace,
    source: str,
    root: Path,
    tokenizer: Any,
    manifest: Dict[str, Any],
    manifest_path: Path,
) -> None:
    state = manifest["sources"][source]
    quota = int(state["target_tokens"])
    if state.get("complete") or int(state.get("tokens", 0)) >= quota:
        state["complete"] = True
        return
    _cleanup_uncommitted_source_files(args.stage_dir, source, state)

    source_kind = "math" if source.startswith("math_") else "general"
    if source == "code":
        all_tasks = build_stack_tasks(root, args.stack_generation, args.seed)
    else:
        all_tasks = build_generic_tasks(root, source, source_kind, args.seed)
    if not all_tasks:
        if int(state.get("tokens") or 0) > 0:
            raise RuntimeError(f"{source}: raw inputs disappeared after checkpoint; refusing unsafe resume")
        raise RuntimeError(f"{source}: no input tasks discovered under {root}")
    _validate_or_upgrade_checkpoint_state(source, state, repair_requested=False, tasks=all_tasks)
    layout_changed = _record_or_verify_task_layout(all_tasks, state, source)
    _verify_task_resume_boundary(all_tasks, state, source)
    if layout_changed:
        atomic_write_json(manifest_path, manifest)
    start_task = int(state.get("next_task_index", 0))
    base_skip = 0
    base_manifest = getattr(args, "_base_stage_manifest", None)
    if base_manifest is not None:
        base_root = str((base_manifest.get("roots") or {}).get(source) or "")
        current_root = str(Path(root).resolve())
        try: base_root_resolved = str(Path(base_root).expanduser().resolve()) if base_root else ""
        except Exception: base_root_resolved = base_root
        if base_root_resolved == current_root:
            bstate=(base_manifest.get("sources") or {}).get(source) or {}
            base_skip=max(0,int(bstate.get("next_task_index") or 0))
            if base_skip>len(all_tasks):
                raise RuntimeError(f"{source}: base consumed task index {base_skip} exceeds current task count {len(all_tasks)}")
            start_task=max(start_task,base_skip)
            state["base_skipped_tasks"]=base_skip
            state["base_stage_source_tokens"]=int(bstate.get("tokens") or 0)
        else:
            state["base_skipped_tasks"]=0
    if start_task >= len(all_tasks):
        state["exhausted"] = True
        state["available_tokens"] = int(state.get("tokens") or 0)
        atomic_write_json(manifest_path, manifest)
        print(
            f"WARN: {source} raw source exhausted at a safe task boundary: "
            f"{int(state.get('tokens', 0)):,}/{quota:,} staging-buffer tokens. "
            "Continuing; downstream plan/trim remains strict on final quotas."
        )
        return
    tasks = [t for t in all_tasks if t.index >= start_task]
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    remaining0 = max(0, quota - int(state.get("tokens", 0)))
    observed_bpt = (
        float(state.get("text_bytes", 0)) / float(state.get("tokens", 0))
        if int(state.get("tokens", 0)) > 0 else None
    )
    hard_pack_bytes = int(args.pack_mib * 2**20)
    min_pack_bytes = int(args.min_pack_mib * 2**20)
    spool_block_bytes = int(args.spool_block_mib * 2**20)
    initial_pack_target = adaptive_pack_target_bytes(
        source, remaining0, hard_pack_bytes, min_pack_bytes,
        safety=args.pack_safety, observed_bytes_per_token=observed_bpt,
    )
    effective_prefetch = adaptive_prefetch_count(
        source, remaining0, min(args.prefetch, args.workers), spool_block_bytes,
        safety=args.prefetch_safety, observed_bytes_per_token=observed_bpt,
    )

    print(f"\n=== {source} ===")
    print(f"root           : {root}")
    print(f"target tokens  : {quota:,}")
    print(f"existing tokens: {int(state.get('tokens', 0)):,}")
    print(f"tasks          : {len(tasks):,} from index {start_task:,} (base-skip={base_skip:,})")
    print(
        f"adaptive       : pack={initial_pack_target/2**20:.1f} MiB, "
        f"active prefetch={effective_prefetch}/{args.prefetch}, "
        f"spool block={args.spool_block_mib:.1f} MiB"
    )

    io_sem = threading.Semaphore(max(1, args.io_readers))
    max_code_bytes = int(args.max_code_file_mib * 2**20)
    ram_guard = MemoryGuard(int(args.ram_limit_gib * 2**30))
    spool_dir = args.stage_dir / ".spool" / source
    if spool_dir.exists():
        shutil.rmtree(spool_dir)
    spool_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()

    def fn(task: ReadTask, q: "queue.Queue[TaskResult]", stop: threading.Event) -> None:
        process_task_stream(
            task, q, stop, io_sem, max_code_bytes, spool_dir,
            args.parquet_batch_rows, ram_guard, spool_block_bytes,
        )

    acc = ShardAccumulator(
        source,
        args.stage_dir / "records",
        initial_pack_target,
        start_index=int(state.get("next_shard_index", 0)),
        fsync=args.fsync,
        hard_cap_bytes=hard_pack_bytes,
    )
    pending_task_metas: List[Dict[str, Any]] = []
    current_task_index: Optional[int] = None
    current_task_key = ""
    t0 = time.perf_counter()

    def update_adaptive_target() -> None:
        if acc.has_data:
            return
        remaining = max(0, quota - int(state["tokens"]))
        if remaining <= 0:
            return
        bpt = (
            float(state["text_bytes"]) / float(state["tokens"])
            if int(state["tokens"]) > 0 else None
        )
        acc.set_target_bytes(adaptive_pack_target_bytes(
            source, remaining, hard_pack_bytes, min_pack_bytes,
            safety=args.pack_safety, observed_bytes_per_token=bpt,
        ))

    def commit_task(metas: Sequence[Dict[str, Any]], task_index: int, task_key: str) -> bool:
        # Count every pack first.  Only after the full task is known-good do we mutate
        # the manifest, so a process crash can never leave a half-committed task.
        new_tokens = int(state["tokens"])
        new_records = int(state["records"])
        new_text_bytes = int(state["text_bytes"])
        subtype_tokens = Counter({str(k): int(v) for k, v in state.get("subtype_tokens", {}).items()})
        subtype_records = Counter({str(k): int(v) for k, v in state.get("subtype_records", {}).items()})
        committed: List[Dict[str, Any]] = []
        log_rows = []
        hit = False
        for pos, raw_meta in enumerate(metas):
            meta = dict(raw_meta)
            remaining = max(0, quota - new_tokens)
            if remaining <= 0:
                Path(meta["path"]).unlink(missing_ok=True)
                continue
            c = count_pack(
                Path(meta["path"]), tokenizer, args.count_batch_size, token_limit=remaining,
                max_batch_bytes=int(args.count_batch_mib * 2**20), ram_guard=ram_guard,
            )
            meta["count"] = c
            meta["task_index"] = int(task_index)
            meta["task_key"] = str(task_key)
            meta["checkpoint_version"] = CHECKPOINT_VERSION
            committed.append(meta)
            new_tokens += int(c["tokens"])
            new_records += int(c["records"])
            new_text_bytes += int(c["text_bytes"])
            _merge_counter_dict(subtype_tokens, c["subtype_tokens"])
            _merge_counter_dict(subtype_records, c["subtype_records"])
            log_rows.append((meta, c, new_tokens))
            if new_tokens >= quota:
                hit = True
                for unused in metas[pos + 1:]:
                    Path(unused["path"]).unlink(missing_ok=True)
                break

        state["shards"].extend(committed)
        state["tokens"] = int(new_tokens)
        state["records"] = int(new_records)
        state["text_bytes"] = int(new_text_bytes)
        state["subtype_tokens"] = dict(subtype_tokens)
        state["subtype_records"] = dict(subtype_records)
        state["next_task_index"] = int(task_index) + 1
        state["next_shard_index"] = int(acc.index)
        state["complete"] = bool(new_tokens >= quota)
        state["exhausted"] = False
        state["available_tokens"] = int(new_tokens)
        state["checkpoint_version"] = CHECKPOINT_VERSION
        state["checkpoint_mode"] = "task_atomic"
        state["last_completed_task_key"] = str(task_key)
        manifest["updated_at_unix"] = time.time()
        atomic_write_json(manifest_path, manifest)
        for meta, c, total in log_rows:
            print(
                f"  shard {Path(meta['path']).name}: "
                f"{meta['text_bytes']/2**20:8.1f} MiB packed, "
                f"{c['text_bytes']/2**20:8.1f} MiB counted, "
                f"{c['tokens']:,} tokens, {c['tokens_per_second']:,.0f} tok/s -> "
                f"{int(total):,}/{quota:,}"
            )
        return bool(hit)

    stream = ordered_stream_results(
        tasks, fn, args.workers, effective_prefetch, stop_event,
        queue_depth=args.task_queue_depth, stall_timeout_s=args.stall_timeout_s,
    )
    stream_closed = False
    try:
        for result in stream:
            if current_task_index is None:
                current_task_index = int(result.index)
                current_task_key = str(result.task_key)
            if int(result.index) != current_task_index or str(result.task_key) != current_task_key:
                raise RuntimeError(
                    f"{source}: ordered stream crossed task boundary without final marker: "
                    f"current={current_task_index}/{current_task_key} got={result.index}/{result.task_key}"
                )
            acc.add(result)
            pending_task_metas.extend(acc.pop_completed())
            if not result.final:
                continue
            tail = acc.flush()
            if tail:
                pending_task_metas.append(tail)
            hit = commit_task(pending_task_metas, current_task_index, current_task_key)
            pending_task_metas = []
            current_task_index = None
            current_task_key = ""
            if hit:
                stop_event.set()
                break
            update_adaptive_target()
    except BaseException:
        try:
            stream.close()
            stream_closed = True
        except Exception:
            pass
        acc.abort()
        _cleanup_uncommitted_source_files(args.stage_dir, source, state)
        shutil.rmtree(spool_dir, ignore_errors=True)
        raise
    finally:
        if not stream_closed:
            try:
                stream.close()
            except Exception:
                pass

    if state.get("complete"):
        acc.abort()
        _cleanup_uncommitted_source_files(args.stage_dir, source, state)
        shutil.rmtree(spool_dir, ignore_errors=True)
        elapsed = time.perf_counter() - t0
        print(
            f"{source} COMPLETE: {int(state['tokens']):,} tokens "
            f"(overshoot {int(state['tokens']) - quota:+,}) in {elapsed/60:.2f} min"
        )
        return

    # Every successful task is flushed and committed at its boundary; no partial
    # accumulator is allowed to survive here.
    acc.abort()
    _cleanup_uncommitted_source_files(args.stage_dir, source, state)
    shutil.rmtree(spool_dir, ignore_errors=True)
    # Reaching here with max_tasks=None means every raw task was consumed cleanly.
    if args.max_tasks is None and int(state.get("next_task_index") or 0) >= len(all_tasks):
        state["exhausted"] = True
        state["available_tokens"] = int(state.get("tokens") or 0)
        atomic_write_json(manifest_path, manifest)
        print(
            f"WARN: {source} raw source exhausted: {int(state['tokens']):,}/{quota:,} "
            "staging-buffer tokens. Downstream final quotas remain fail-closed."
        )
    else:
        print(f"WARN: {source} not yet complete: {int(state['tokens']):,}/{quota:,}")

def build_report(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    sources = manifest["sources"]
    total = sum(int(sources[s].get("tokens", 0)) for s in SOURCE_ORDER)
    target = int(manifest["target_tokens"])
    return {
        "script_revision": SCRIPT_REVISION,
        "created_at_unix": time.time(),
        "target_tokens": target,
        "actual_tokens": total,
        "overshoot_tokens": total - target,
        "complete": all(bool(sources[s].get("complete")) for s in SOURCE_ORDER),
        "sources": {
            s: {
                "target_tokens": int(sources[s]["target_tokens"]),
                "tokens": int(sources[s].get("tokens", 0)),
                "overshoot_tokens": int(sources[s].get("tokens", 0)) - int(sources[s]["target_tokens"]),
                "records": int(sources[s].get("records", 0)),
                "text_bytes": int(sources[s].get("text_bytes", 0)),
                "subtype_tokens": dict(sources[s].get("subtype_tokens", {})),
                "subtype_records": dict(sources[s].get("subtype_records", {})),
                "shards": len(sources[s].get("shards", [])),
                "complete": bool(sources[s].get("complete")),
                "exhausted": bool(sources[s].get("exhausted", False)),
                "available_tokens": int(sources[s].get("available_tokens", sources[s].get("tokens", 0))),
            }
            for s in SOURCE_ORDER
        },
    }


def recount_stage(args: argparse.Namespace, tokenizer: Any) -> Dict[str, Any]:
    manifest_path = args.stage_dir / "stage_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    quotas = {k: int(v) for k, v in manifest["quotas"].items()}
    ram_guard = MemoryGuard(int(args.ram_limit_gib * 2**30))
    for source in SOURCE_ORDER:
        state = manifest["sources"][source]
        fresh = _empty_source_state(quotas[source])
        for meta in state.get("shards", []):
            remaining = max(0, quotas[source] - fresh["tokens"])
            if remaining == 0:
                break
            c = count_pack(
                Path(meta["path"]), tokenizer, args.count_batch_size, token_limit=remaining,
                max_batch_bytes=int(args.count_batch_mib * 2**20), ram_guard=ram_guard,
            )
            fresh["tokens"] += int(c["tokens"])
            fresh["records"] += int(c["records"])
            fresh["text_bytes"] += int(c["text_bytes"])
            _merge_counter_dict(fresh["subtype_tokens"], c["subtype_tokens"])
            _merge_counter_dict(fresh["subtype_records"], c["subtype_records"])
        fresh["complete"] = fresh["tokens"] >= quotas[source]
        # Preserve physical shard metadata and the exact task-boundary checkpoint.
        fresh["shards"] = state.get("shards", [])
        fresh["next_task_index"] = state.get("next_task_index", 0)
        fresh["next_shard_index"] = state.get("next_shard_index", len(fresh["shards"]))
        fresh["checkpoint_version"] = state.get("checkpoint_version", 0)
        fresh["checkpoint_mode"] = state.get("checkpoint_mode", "legacy")
        fresh["last_completed_task_key"] = state.get("last_completed_task_key", "")
        manifest["sources"][source] = fresh
    manifest["complete"] = all(bool(manifest["sources"][s].get("complete")) for s in SOURCE_ORDER)
    manifest["updated_at_unix"] = time.time()
    atomic_write_json(manifest_path, manifest)
    report = build_report(manifest)
    atomic_write_json(args.stage_dir / "token_count_report.json", report)
    return report


def run_all(args: argparse.Namespace) -> Dict[str, Any]:
    args.stage_dir = args.stage_dir.expanduser().resolve()
    manifest_path = args.stage_dir / "stage_manifest.json"
    if args.rebuild_stage and args.stage_dir.exists():
        shutil.rmtree(args.stage_dir)
    args.stage_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, tokenizer_path = load_frozen_tokenizer(args.tokenizer, args.tokenizer_threads)
    roots = source_roots(args)
    base_manifest = None
    args._base_stage_manifest_sha256 = ""
    if args.base_stage_dir:
        bp=args.base_stage_dir/"stage_manifest.json"
        if not bp.is_file():raise RuntimeError(f"base stage manifest missing: {bp}")
        base_manifest=json.loads(bp.read_text(encoding="utf-8"))
        if base_manifest.get("strict_pack_contract") != STRICT_PACK_CONTRACT or int(base_manifest.get("format_version") or 0) != 5:
            raise RuntimeError(f"base stage contract incompatible: {bp}")
        if int(base_manifest.get("seed") or -1) != int(args.seed):
            raise RuntimeError(f"base stage seed mismatch: base={base_manifest.get('seed')} current={args.seed}")
        if str(base_manifest.get("stack_generation") or "") != str(args.stack_generation):
            raise RuntimeError("base stage Stack generation mismatch")
        if str(base_manifest.get("tokenizer_sha256") or "") != _sha256_file(tokenizer_path):
            raise RuntimeError("base stage tokenizer mismatch")
        args._base_stage_manifest_sha256=_sha256_file(bp)
        args._base_stage_manifest=base_manifest

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "target_tokens": args.target_tokens,
            "strict_pack_contract": STRICT_PACK_CONTRACT,
            "seed": args.seed,
            "stack_generation": args.stack_generation,
            "tokenizer": str(tokenizer_path),
            "code_pack_format": "PTCODE5",
            "quotas": compute_quotas(args.target_tokens),
            "roots": {k: str(v) for k, v in roots.items()},
            "base_stage_dir": str(args.base_stage_dir) if args.base_stage_dir else None,
            "base_stage_manifest_sha256": str(getattr(args, "_base_stage_manifest_sha256", "") or ""),
        }
        actual = {k: manifest.get(k) for k in expected}
        if actual != expected:
            raise RuntimeError(f"stage contract mismatch: actual={actual}, expected={expected}; use --rebuild-stage")
        old_code_cap = manifest.get("max_code_file_mib", 4.0)
        if float(old_code_cap) != float(args.max_code_file_mib):
            raise RuntimeError(
                f"stage contract mismatch: max_code_file_mib={old_code_cap} vs {args.max_code_file_mib}; "
                "changing code filtering mid-stage is unsafe"
            )
        tokenizer_sha = _sha256_file(tokenizer_path)
        old_tokenizer_sha = str(manifest.get("tokenizer_sha256") or "")
        if old_tokenizer_sha and old_tokenizer_sha != tokenizer_sha:
            raise RuntimeError("stage contract mismatch: tokenizer bytes changed at the same path")
        old_langs = manifest.get("code_languages")
        if old_langs is not None and list(old_langs) != list(CODE_LANGUAGES):
            raise RuntimeError("stage contract mismatch: CODE_LANGUAGES order changed")
        manifest["tokenizer_sha256"] = tokenizer_sha
        manifest["code_languages"] = list(CODE_LANGUAGES)
        manifest["script_revision"] = SCRIPT_REVISION
        manifest["ram_limit_gib"] = args.ram_limit_gib
        manifest["count_batch_mib"] = args.count_batch_mib
        manifest["parquet_batch_rows"] = args.parquet_batch_rows
        manifest["pack_mib"] = args.pack_mib
        manifest["min_pack_mib"] = args.min_pack_mib
        manifest["pack_safety"] = args.pack_safety
        manifest["spool_block_mib"] = args.spool_block_mib
        manifest["prefetch_safety"] = args.prefetch_safety
        manifest["task_queue_depth"] = args.task_queue_depth
        manifest["max_code_file_mib"] = args.max_code_file_mib
        manifest["stall_timeout_s"] = args.stall_timeout_s
    else:
        manifest = _initial_manifest(args, roots, tokenizer_path)
        atomic_write_json(manifest_path, manifest)

    selected = set(args.sources.split(",")) if args.sources else set(SOURCE_ORDER)
    unknown = selected.difference(SOURCE_ORDER)
    if unknown:
        raise ValueError(f"unknown --sources: {sorted(unknown)}")

    repair_sources = {x.strip() for x in str(args.repair_incomplete_source or "").split(",") if x.strip()}
    unknown_repair = repair_sources.difference(SOURCE_ORDER)
    if unknown_repair:
        raise ValueError(f"unknown repair source(s): {sorted(unknown_repair)}")
    backup_done = False
    # Validate only sources this invocation may touch (plus explicit repair targets).
    # This makes --sources useful for surgical recovery instead of being blocked by
    # unrelated legacy checkpoints elsewhere in the manifest.
    for source in SOURCE_ORDER:
        if source not in selected and source not in repair_sources:
            continue
        state = manifest["sources"][source]
        tasks = None
        if not bool(state.get("complete")) and _legacy_checkpoint_has_progress(state):
            sk = "math" if source.startswith("math_") else "general"
            tasks = (
                build_stack_tasks(roots[source], args.stack_generation, args.seed)
                if source == "code"
                else build_generic_tasks(roots[source], source, sk, args.seed)
            )
        action = _validate_or_upgrade_checkpoint_state(
            source, state, repair_requested=source in repair_sources, tasks=tasks
        )
        if action == "repair_required":
            if not backup_done:
                backup = manifest_path.with_name(
                    f"stage_manifest.before_v7_repair.{int(time.time())}.json"
                )
                shutil.copy2(manifest_path, backup)
                print(f"legacy checkpoint backup: {backup}")
                backup_done = True
            print(f"repairing legacy incomplete source {source}: resetting only this source")
            # Persist reset before deleting old packs: an interruption can only leave
            # unlisted orphans, never a manifest pointing at deleted data.
            _repair_legacy_incomplete_source(args.stage_dir, manifest, source, delete_files=False)
            atomic_write_json(manifest_path, manifest)
            _purge_source_files(args.stage_dir, source)
    atomic_write_json(manifest_path, manifest)

    for source in SOURCE_ORDER:
        if source not in selected:
            continue
        run_source(args, source, roots[source], tokenizer, manifest, manifest_path)

    manifest["complete"] = all(bool(manifest["sources"][s].get("complete")) for s in SOURCE_ORDER)
    manifest["updated_at_unix"] = time.time()
    atomic_write_json(manifest_path, manifest)
    report = build_report(manifest)
    atomic_write_json(args.stage_dir / "token_count_report.json", report)
    return report


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="V7: deadlock-safe streaming, task-atomic resume, task-aware legacy migration and capacity-safe staging into hard-capped SSD packs.",
    )
    p.add_argument("--phase", choices=["all", "count"], default="all")
    p.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--stack-root", type=Path, default=DEFAULT_STACK_ROOT)
    p.add_argument("--nemotron-root", type=Path, default=None)
    p.add_argument("--fineweb-root", type=Path, default=DEFAULT_FINEWEB_ROOT)
    p.add_argument("--finemath-root", type=Path, default=None)
    p.add_argument("--pes2o-root", type=Path, default=None)
    p.add_argument("--wiki-root", type=Path, default=None)
    p.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    p.add_argument("--base-stage-dir", type=Path, default=None, help="immutable prior stage manifest; skip already-consumed task prefix for matching roots")
    p.add_argument("--stack-generation", default=DEFAULT_STACK_GENERATION)

    p.add_argument("--workers", type=int, default=32, help="ingest worker threads")
    p.add_argument("--io-readers", type=int, default=2, help="maximum simultaneous HDD/file reads")
    p.add_argument("--prefetch", type=int, default=32, help="maximum in-flight ordered ingest tasks")
    p.add_argument("--tokenizer-threads", type=int, default=32, help="Rayon threads for encode_batch")
    p.add_argument("--count-batch-size", type=int, default=256)
    p.add_argument("--count-batch-mib", type=float, default=64.0, help="max UTF-8 MiB passed to one tokenizer encode_batch")
    p.add_argument("--parquet-batch-rows", type=int, default=128, help="rows decoded at a time inside each Parquet row group")
    p.add_argument("--ram-limit-gib", type=float, default=70.0, help="cooperative process RSS upper limit")
    p.add_argument("--pack-mib", type=int, default=256, help="hard maximum final .pack size in MiB")
    p.add_argument("--min-pack-mib", type=float, default=8.0, help="smallest adaptive pack target near a token quota")
    p.add_argument("--pack-safety", type=float, default=1.20, help="raw-byte safety multiplier for adaptive early counting")
    p.add_argument("--spool-block-mib", type=float, default=16.0, help="maximum worker SSD spool block before it is emitted")
    p.add_argument("--prefetch-safety", type=float, default=1.50, help="raw-byte safety multiplier for adaptive task prefetch")
    p.add_argument("--task-queue-depth", type=int, default=2, help="spool blocks allowed ahead per active task")
    p.add_argument("--stall-timeout-s", type=float, default=DEFAULT_STALL_TIMEOUT_S, help="fail with diagnostics after this many seconds without ordered progress; 0 disables")
    p.add_argument("--repair-incomplete-source", default=os.environ.get("REPAIR_INCOMPLETE_SOURCE", ""), help="comma-separated legacy incomplete source(s) to reset safely before resume")
    p.add_argument("--max-code-file-mib", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--sources", default="", help="optional comma list, e.g. general,code")
    p.add_argument("--max-tasks", type=int, default=None, help="debug: cap tasks processed per source per run")
    p.add_argument("--fsync", action="store_true", help="fsync every completed pack shard")
    p.add_argument("--rebuild-stage", action="store_true")
    return p


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.target_tokens <= 0:
        raise ValueError("--target-tokens must be > 0")
    for name in ("workers", "io_readers", "prefetch", "tokenizer_threads", "count_batch_size", "parquet_batch_rows", "pack_mib", "task_queue_depth"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    for name in ("count_batch_mib", "min_pack_mib", "spool_block_mib"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be > 0")
    if args.min_pack_mib > args.pack_mib:
        raise ValueError("--min-pack-mib must be <= --pack-mib")
    if args.pack_safety <= 1.0:
        raise ValueError("--pack-safety must be > 1")
    if args.prefetch_safety <= 1.0:
        raise ValueError("--prefetch-safety must be > 1")
    if args.ram_limit_gib <= 1:
        raise ValueError("--ram-limit-gib must be > 1")
    if args.max_code_file_mib <= 0:
        raise ValueError("--max-code-file-mib must be > 0")
    if float(args.stall_timeout_s) < 0:
        raise ValueError("--stall-timeout-s must be >= 0")
    args.stage_dir = args.stage_dir.expanduser().resolve()
    args.base_stage_dir = args.base_stage_dir.expanduser().resolve() if args.base_stage_dir else None
    return args


def print_report(report: Mapping[str, Any]) -> None:
    print("\n=== TOKEN COUNT SUMMARY ===")
    for source in SOURCE_ORDER:
        x = report["sources"][source]
        pct = 100.0 * x["tokens"] / x["target_tokens"] if x["target_tokens"] else 0.0
        print(
            f"{source:16s} {x['tokens']:>14,d} / {x['target_tokens']:>14,d} "
            f"({pct:7.3f}%)  records={x['records']:,}  raw={x['text_bytes']/2**30:.3f} GiB"
        )
        if source == "code" and x.get("subtype_tokens"):
            total = sum(int(v) for v in x["subtype_tokens"].values()) or 1
            for lang, n in sorted(x["subtype_tokens"].items(), key=lambda kv: -int(kv[1])):
                print(f"  {lang:14s} {int(n):>14,d}  {100.0*int(n)/total:6.2f}%")
    print(f"TOTAL            {report['actual_tokens']:>14,d} / {report['target_tokens']:>14,d}")
    print(f"complete: {report['complete']}")



# =============================================================================
# V5 code-pack overlay
# =============================================================================
# Non-code sources intentionally remain byte-compatible with V3 PTPACK1.
# Stack-v3 uses PTCODE5 so repository/file provenance survives SSD staging.

CODE_PACK_MAGIC = b"PTCODE5\n"
CODE_PACK_HEADER_BYTES = len(CODE_PACK_MAGIC)
CODE_FRAME = struct.Struct("<BBBII")  # kind, subtype, flags, metadata_len, UTF-8 content_len
CODE_KIND_REPO = 1
CODE_KIND_FILE = 2
CODE_FLAG_COUNT_FOR_QUOTA = 1 << 0
CODE_FLAG_META = 1 << 1

CODE_META_EXACT = {
    "readme", "readme.md", "readme.rst", "readme.txt",
    "license", "license.md", "license.txt", "copying", "notice",
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "pipfile",
    "package.json", "tsconfig.json", "deno.json", "deno.jsonc",
    "cmakelists.txt", "makefile", "meson.build", "build.gradle", "build.gradle.kts",
    "pom.xml", "go.mod", "go.sum", "cargo.toml", "dockerfile", "compose.yml",
    "compose.yaml", ".editorconfig", ".gitignore", ".gitattributes",
}
CODE_META_PREFIXES = ("readme", "license", "copying", "notice", "requirements")
CODE_META_DIR_PARTS = ("/.github/", "/.gitlab/", "/ci/", "/config/", "/configs/")


@dataclass(frozen=True)
class CodeV4Frame:
    kind: int
    subtype: int
    flags: int
    metadata: Mapping[str, Any]
    text: str = ""


def is_code_meta_path(path: str) -> bool:
    p = "/" + (path or "").replace("\\", "/").lower().lstrip("/")
    base = p.rsplit("/", 1)[-1]
    return (
        base in CODE_META_EXACT
        or any(base.startswith(x) for x in CODE_META_PREFIXES)
        or any(x in p for x in CODE_META_DIR_PARTS)
    )


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    # Arrow can occasionally surface date/time/scalar wrapper objects.
    return str(value)


def _code_json_bytes(metadata: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(dict(metadata)), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def encode_code_v4_frame(frame: CodeV4Frame) -> bytes:
    if frame.kind not in {CODE_KIND_REPO, CODE_KIND_FILE}:
        raise ValueError(f"invalid PTCODE5 frame kind {frame.kind}")
    if not 0 <= int(frame.subtype) <= 255:
        raise ValueError(f"subtype out of uint8 range: {frame.subtype}")
    if not 0 <= int(frame.flags) <= 255:
        raise ValueError(f"flags out of uint8 range: {frame.flags}")
    meta = _code_json_bytes(frame.metadata)
    text = frame.text.encode("utf-8")
    if len(meta) > 0xFFFFFFFF or len(text) > 0xFFFFFFFF:
        raise ValueError("PTCODE5 metadata/content exceeds uint32 frame length")
    return CODE_FRAME.pack(
        int(frame.kind), int(frame.subtype), int(frame.flags), len(meta), len(text)
    ) + meta + text


def _read_code_v4_frame_parts(f) -> Optional[Tuple[int, int, int, bytes, bytes]]:
    header = f.read(CODE_FRAME.size)
    if not header:
        return None
    if len(header) != CODE_FRAME.size:
        raise RuntimeError("truncated PTCODE5 frame header")
    kind, subtype, flags, meta_n, text_n = CODE_FRAME.unpack(header)
    meta = f.read(meta_n)
    if len(meta) != meta_n:
        raise RuntimeError("truncated PTCODE5 metadata payload")
    text = f.read(text_n)
    if len(text) != text_n:
        raise RuntimeError("truncated PTCODE5 text payload")
    return int(kind), int(subtype), int(flags), meta, text


def iter_code_v4_frames(path: Path | str) -> Iterator[CodeV4Frame]:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(CODE_PACK_HEADER_BYTES)
        if magic != CODE_PACK_MAGIC:
            raise RuntimeError(f"invalid PTCODE5 magic/version: {path}")
        while True:
            parts = _read_code_v4_frame_parts(f)
            if parts is None:
                return
            kind, subtype, flags, meta, text = parts
            try:
                metadata = json.loads(meta.decode("utf-8")) if meta else {}
                payload = text.decode("utf-8", errors="strict")
            except Exception as e:
                raise RuntimeError(f"invalid PTCODE5 frame in {path}: {e}") from e
            yield CodeV4Frame(kind, subtype, flags, metadata, payload)


def _iter_code_v4_countable_payloads(path: Path) -> Iterator[Tuple[int, str]]:
    """Fast count path: skip JSON parsing and all non-quota V4 payloads."""
    with Path(path).open("rb") as f:
        if f.read(CODE_PACK_HEADER_BYTES) != CODE_PACK_MAGIC:
            raise RuntimeError(f"invalid PTCODE5 magic/version: {path}")
        while True:
            header = f.read(CODE_FRAME.size)
            if not header:
                return
            if len(header) != CODE_FRAME.size:
                raise RuntimeError(f"truncated PTCODE5 frame header: {path}")
            kind, subtype, flags, meta_n, text_n = CODE_FRAME.unpack(header)
            # Counting does not need metadata at all.
            if meta_n:
                f.seek(meta_n, os.SEEK_CUR)
            if kind == CODE_KIND_FILE and (flags & CODE_FLAG_COUNT_FOR_QUOTA):
                payload = f.read(text_n)
                if len(payload) != text_n:
                    raise RuntimeError(f"truncated PTCODE5 file content: {path}")
                yield int(subtype), payload.decode("utf-8", errors="strict")
            else:
                if text_n:
                    f.seek(text_n, os.SEEK_CUR)


def _code_frame_stats(records: Iterable[CodeV4Frame]) -> Dict[str, int]:
    out = Counter()
    for r in records:
        if r.kind == CODE_KIND_REPO:
            out["repo_manifests"] += 1
        elif r.kind == CODE_KIND_FILE:
            if r.flags & CODE_FLAG_COUNT_FOR_QUOTA:
                out["countable_files"] += 1
                out["countable_text_bytes"] += len(r.text.encode("utf-8"))
            else:
                out["aux_files"] += 1
                out["aux_text_bytes"] += len(r.text.encode("utf-8"))
    return dict(out)


def write_code_v4_pack(path: Path | str, records: Iterable[CodeV4Frame], fsync: bool = False) -> Dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frames = 0
    repo_manifests = 0
    countable_files = 0
    aux_files = 0
    text_bytes = 0
    aux_text_bytes = 0
    subtype_records = Counter()
    with tmp.open("wb") as f:
        f.write(CODE_PACK_MAGIC)
        for r in records:
            raw = encode_code_v4_frame(r)
            f.write(raw)
            frames += 1
            if r.kind == CODE_KIND_REPO:
                repo_manifests += 1
            elif r.kind == CODE_KIND_FILE:
                n = len(r.text.encode("utf-8"))
                if r.flags & CODE_FLAG_COUNT_FOR_QUOTA:
                    countable_files += 1
                    text_bytes += n
                    subtype_records[int(r.subtype)] += 1
                else:
                    aux_files += 1
                    aux_text_bytes += n
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)
    return {
        "path": str(path),
        "format": "PTCODE5",
        "frames": frames,
        "repo_manifests": repo_manifests,
        "countable_files": countable_files,
        "aux_files": aux_files,
        "records": countable_files,
        "text_bytes": text_bytes,
        "aux_text_bytes": aux_text_bytes,
        "file_bytes": path.stat().st_size,
        "subtype_records": {str(k): int(v) for k, v in sorted(subtype_records.items())},
    }


def _repo_key(repo_id: str, repo_path: str, commit_id: str) -> str:
    raw = f"{repo_id}\0{repo_path}\0{commit_id}".encode("utf-8", errors="surrogatepass")
    return hashlib.blake2b(raw, digest_size=16, person=b"ptcode4repo").hexdigest()


def _tree_entry(file_obj: Mapping[str, Any]) -> Dict[str, Any]:
    keys = (
        "file_path", "language", "size_bytes", "content_id", "license_type",
        "is_vendor", "file_timestamp",
    )
    return {k: _json_safe(file_obj.get(k)) for k in keys if file_obj.get(k) is not None}


def _valid_v4_file_text(
    text: Any, *, min_bytes: int, max_code_file_bytes: int, reject_minified: bool,
) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    if reject_minified and looks_minified(text):
        return None
    try:
        payload = text.encode("utf-8")
    except UnicodeEncodeError:
        return None
    n = len(payload)
    if n < min_bytes or n > int(max_code_file_bytes) or n > 0xFFFFFFFF:
        return None
    if payload.count(b"\x00") > max(2, n // 10_000):
        return None
    return text


def stack_row_to_code_v4_records(
    row: Mapping[str, Any], *, max_code_file_bytes: int,
) -> List[CodeV4Frame]:
    """Convert one Stack-v3 repository row into a repo manifest + retained files.

    The full filtered path tree is preserved even for unsupported languages. File
    content is stored only for the 28 selected programming languages plus repository meta/
    config/documentation files. Only primary-language frames carry the quota flag.
    """
    repo_path = str(row.get("repo_path") or "")
    repo_id = str(row.get("repo_id") if row.get("repo_id") is not None else "")
    commit_id = str(row.get("commit_id") or "")
    key = _repo_key(repo_id, repo_path, commit_id)
    raw_files = [f for f in (row.get("files") or []) if isinstance(f, Mapping)]

    tree: List[Dict[str, Any]] = []
    retained: List[CodeV4Frame] = []
    for file_index, f in enumerate(raw_files):
        file_path = str(f.get("file_path") or "")
        if not file_path or bool(f.get("is_vendor")) or is_bad_code_path(file_path):
            continue
        tree.append(_tree_entry(f))
        language = canonical_code_language(f.get("language"))
        subtype = code_language_id(language)
        meta_file = is_code_meta_path(file_path)
        if subtype is None and not meta_file:
            continue
        text = _valid_v4_file_text(
            f.get("content"),
            min_bytes=8 if subtype is not None else 1,
            max_code_file_bytes=max_code_file_bytes,
            reject_minified=subtype is not None,
        )
        if text is None:
            continue
        flags = 0
        if subtype is not None:
            flags |= CODE_FLAG_COUNT_FOR_QUOTA
        if meta_file:
            flags |= CODE_FLAG_META
        file_metadata = {
            str(k): _json_safe(v) for k, v in f.items() if k != "content"
        }
        retained.append(CodeV4Frame(
            CODE_KIND_FILE,
            int(subtype or 0),
            flags,
            {
                "repo_key": key,
                "repo_id": repo_id,
                "repo_path": repo_path,
                "commit_id": commit_id,
                "file_index": int(file_index),
                "file_path": file_path,
                "language": language,
                "file_metadata": file_metadata,
            },
            text,
        ))

    if not retained:
        return []
    tree.sort(key=lambda x: str(x.get("file_path") or ""))
    repo_metadata = {
        str(k): _json_safe(v) for k, v in row.items() if k != "files"
    }
    manifest = CodeV4Frame(
        CODE_KIND_REPO, 0, 0,
        {
            "repo_key": key,
            "repo_id": repo_id,
            "repo_path": repo_path,
            "commit_id": commit_id,
            "github_metadata": _json_safe(row.get("github_metadata") or {}),
            "num_files": int(row.get("num_files") or len(raw_files)),
            "tree": tree,
            "repo_metadata": repo_metadata,
        },
        "",
    )
    return [manifest] + retained


def iter_code_v4_repositories(paths: Iterable[Path | str]) -> Iterator[Dict[str, Any]]:
    """Reconstruct repo objects across one or many PTCODE5 shards.

    Output uses the same basic dictionary contract consumed by semantic_process_v1:
    repo_id/repo_path/commit_id/github_metadata/files. `tree` remains available for
    the exact original filtered directory view without needing to infer it from
    retained file payloads.
    """
    manifests: Dict[str, Dict[str, Any]] = {}
    files: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    seen_files: Dict[str, set] = {}
    for path in paths:
        for frame in iter_code_v4_frames(path):
            meta = dict(frame.metadata)
            key = str(meta.get("repo_key") or "")
            if not key:
                continue
            if key not in manifests:
                order.append(key)
                manifests[key] = {
                    "repo_key": key,
                    "repo_id": str(meta.get("repo_id") or ""),
                    "repo_path": str(meta.get("repo_path") or ""),
                    "commit_id": str(meta.get("commit_id") or ""),
                    "github_metadata": {},
                    "tree": [],
                    "repo_metadata": {},
                }
                files[key] = []
                seen_files[key] = set()
            if frame.kind == CODE_KIND_REPO:
                manifests[key].update({
                    "repo_id": str(meta.get("repo_id") or manifests[key]["repo_id"]),
                    "repo_path": str(meta.get("repo_path") or manifests[key]["repo_path"]),
                    "commit_id": str(meta.get("commit_id") or manifests[key]["commit_id"]),
                    "github_metadata": meta.get("github_metadata") or {},
                    "tree": meta.get("tree") or [],
                    "repo_metadata": meta.get("repo_metadata") or {},
                })
            elif frame.kind == CODE_KIND_FILE:
                fm = dict(meta.get("file_metadata") or {})
                file_path = str(meta.get("file_path") or fm.get("file_path") or "")
                identity = (file_path, str(fm.get("content_id") or ""), hashlib.sha256(frame.text.encode("utf-8")).hexdigest())
                if identity in seen_files[key]:
                    continue
                seen_files[key].add(identity)
                fm.update({
                    "file_path": file_path,
                    "language": meta.get("language") or fm.get("language"),
                    "content": frame.text,
                    "is_vendor": bool(fm.get("is_vendor", False)),
                    "count_for_quota": bool(frame.flags & CODE_FLAG_COUNT_FOR_QUOTA),
                    "is_meta": bool(frame.flags & CODE_FLAG_META),
                })
                files[key].append(fm)
    for key in order:
        if not files.get(key):
            continue
        repo = dict(manifests[key])
        repo["files"] = sorted(files[key], key=lambda x: str(x.get("file_path") or ""))
        repo["tree"] = sorted(repo.get("tree") or [], key=lambda x: str(x.get("file_path") or ""))
        yield repo


# Keep aliases to V3 behavior for non-code sources.
_v3_count_pack = count_pack
_v3_iter_pack_records = iter_pack_records
_v3_process_task_stream = process_task_stream
_V3ShardAccumulator = ShardAccumulator


def iter_pack_records(path: Path) -> Iterator[Tuple[int, str]]:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(max(PACK_HEADER_BYTES, CODE_PACK_HEADER_BYTES))
    if magic.startswith(CODE_PACK_MAGIC):
        yield from _iter_code_v4_countable_payloads(path)
    else:
        yield from _v3_iter_pack_records(path)


def count_pack(
    path: Path,
    tokenizer: Any,
    batch_size: int = 256,
    token_limit: Optional[int] = None,
    max_batch_bytes: int = 64 << 20,
    ram_guard: Optional[MemoryGuard] = None,
) -> Dict[str, Any]:
    path = Path(path)
    with path.open("rb") as f:
        magic = f.read(max(PACK_HEADER_BYTES, CODE_PACK_HEADER_BYTES))
    if not magic.startswith(CODE_PACK_MAGIC):
        return _v3_count_pack(path, tokenizer, batch_size, token_limit, max_batch_bytes, ram_guard)

    t0 = time.perf_counter()
    total_tokens = 0
    total_records = 0
    total_text_bytes = 0
    subtype_tokens = Counter()
    subtype_records = Counter()
    subtype_text_bytes = Counter()
    batch: List[Tuple[int, str, int]] = []
    batch_bytes = 0
    done = False

    def consume(rows: List[Tuple[int, str, int]]) -> None:
        nonlocal total_tokens, total_records, total_text_bytes, done
        if not rows or done:
            return
        raw_bytes = sum(x[2] for x in rows)
        reserve_bytes = min(raw_bytes * 4, 2 << 30)
        cm = ram_guard.reserve(reserve_bytes) if ram_guard is not None else nullcontext()
        with cm:
            texts = [x[1] for x in rows]
            enc = tokenizer.encode_batch(texts, add_special_tokens=False)
            for (subtype, _text, b), e in zip(rows, enc):
                n = len(e.ids)
                total_tokens += n
                total_records += 1
                total_text_bytes += b
                subtype_tokens[subtype] += n
                subtype_records[subtype] += 1
                subtype_text_bytes[subtype] += b
                if token_limit is not None and total_tokens >= int(token_limit):
                    done = True
                    break
            del enc, texts

    for subtype, text in _iter_code_v4_countable_payloads(path):
        if done:
            break
        nbytes = len(text.encode("utf-8"))
        if batch and (len(batch) >= batch_size or batch_bytes + nbytes > max_batch_bytes):
            consume(batch)
            batch = []
            batch_bytes = 0
            if done:
                break
        batch.append((subtype, text, nbytes))
        batch_bytes += nbytes
        if len(batch) >= batch_size or batch_bytes >= max_batch_bytes:
            consume(batch)
            batch = []
            batch_bytes = 0
    if batch and not done:
        consume(batch)

    elapsed = time.perf_counter() - t0
    def subtype_name(i: int) -> str:
        return ID_TO_CODE_LANGUAGE.get(i, "generic" if i == 0 else f"subtype_{i}")
    return {
        "path": str(path),
        "file_bytes": path.stat().st_size,
        "text_bytes": total_text_bytes,
        "records": total_records,
        "tokens": total_tokens,
        "load_seconds": 0.0,
        "elapsed_seconds": elapsed,
        "tokens_per_second": (total_tokens / elapsed) if elapsed else None,
        "mib_per_second": (total_text_bytes / 2**20 / elapsed) if elapsed else None,
        "subtype_tokens": {subtype_name(k): int(v) for k, v in sorted(subtype_tokens.items())},
        "subtype_records": {subtype_name(k): int(v) for k, v in sorted(subtype_records.items())},
        "subtype_text_bytes": {subtype_name(k): int(v) for k, v in sorted(subtype_text_bytes.items())},
        "limit_reached": token_limit is not None and total_tokens >= int(token_limit),
        "overshoot_tokens": max(0, total_tokens - int(token_limit)) if token_limit is not None else 0,
        "format": "PTCODE5",
    }


class _CodeV4StreamingSpoolWriter:
    def __init__(self, task: ReadTask, spool_dir: Path, block_bytes: int, emit, stop_event: threading.Event):
        self.task = task
        self.spool_dir = spool_dir
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        self.block_bytes = max(1, int(block_bytes))
        self.emit = emit
        self.stop_event = stop_event
        self.chunk_index = 0
        self.f = None
        self.path: Optional[Path] = None
        self.packed_bytes = 0
        self.frames = 0
        self.records = 0  # quota-bearing primary code files only
        self.text_bytes = 0
        self.sub_records = Counter()
        self.sub_bytes = Counter()

    def _open(self) -> None:
        if self.f is not None:
            return
        self.path = self.spool_dir / (
            f"task-{self.task.index:08d}-chunk-{self.chunk_index:06d}-"
            f"{os.getpid()}-{threading.get_ident()}.code4"
        )
        self.f = self.path.open("wb")

    def _reset(self) -> None:
        self.f = None
        self.path = None
        self.packed_bytes = 0
        self.frames = 0
        self.records = 0
        self.text_bytes = 0
        self.sub_records.clear()
        self.sub_bytes.clear()

    def _emit_current(self, final: bool) -> bool:
        if self.f is None or self.frames == 0:
            if final:
                return bool(self.emit(TaskResult(
                    self.task.index, _task_key(self.task), "", 0, 0, 0, {}, {},
                    chunk_index=self.chunk_index, final=True,
                )))
            return True
        assert self.path is not None
        self.f.close()
        event = TaskResult(
            self.task.index, _task_key(self.task), str(self.path), self.packed_bytes,
            self.records, self.text_bytes, dict(self.sub_records), dict(self.sub_bytes),
            chunk_index=self.chunk_index, final=final,
        )
        ok = bool(self.emit(event))
        if not ok:
            self.path.unlink(missing_ok=True)
        self.chunk_index += 1
        self._reset()
        return ok

    def add_frame(self, frame: CodeV4Frame) -> bool:
        if self.stop_event.is_set():
            return False
        raw = encode_code_v4_frame(frame)
        if self.frames and self.packed_bytes + len(raw) > self.block_bytes:
            if not self._emit_current(final=False):
                return False
        self._open()
        assert self.f is not None
        self.f.write(raw)
        self.packed_bytes += len(raw)
        self.frames += 1
        if frame.kind == CODE_KIND_FILE and (frame.flags & CODE_FLAG_COUNT_FOR_QUOTA):
            n = len(frame.text.encode("utf-8"))
            self.records += 1
            self.text_bytes += n
            self.sub_records[int(frame.subtype)] += 1
            self.sub_bytes[int(frame.subtype)] += n
        return True

    def finish(self) -> None:
        if self.stop_event.is_set():
            self.abort_current()
            return
        self._emit_current(final=True)

    def abort_current(self) -> None:
        if self.f is not None:
            try:
                self.f.close()
            except Exception:
                pass
        if self.path is not None:
            self.path.unlink(missing_ok=True)
        self._reset()


def process_task_stream(
    task: ReadTask,
    out_queue: "queue.Queue[TaskResult]",
    stop_event: threading.Event,
    io_sem: threading.Semaphore,
    max_code_file_bytes: int,
    spool_dir: Path,
    parquet_batch_rows: int,
    ram_guard: MemoryGuard,
    spool_block_bytes: int,
) -> None:
    if task.kind != "stack_parquet_rg":
        return _v3_process_task_stream(
            task, out_queue, stop_event, io_sem, max_code_file_bytes,
            spool_dir, parquet_batch_rows, ram_guard, spool_block_bytes,
        )

    path = Path(task.path)
    emit = lambda event: _queue_emit(out_queue, stop_event, event)
    out = _CodeV4StreamingSpoolWriter(task, spool_dir, spool_block_bytes, emit, stop_event)
    try:
        pq = _import_pyarrow_parquet()
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        wanted = [x for x in ("repo_path", "repo_id", "commit_id", "github_metadata", "num_files", "files") if x in available]
        if "files" not in wanted:
            raise RuntimeError(f"Stack-v3 parquet missing files column: {path}")
        batches = iter(pf.iter_batches(
            batch_size=max(1, int(parquet_batch_rows)), row_groups=[task.row_group],
            columns=wanted, use_threads=True,
        ))
        while not stop_event.is_set():
            try:
                with ram_guard.reserve(512 << 20):
                    with io_sem:
                        batch = next(batches)
                    rows = batch.to_pylist()
            except StopIteration:
                break
            should_stop = False
            try:
                for row in rows:
                    if stop_event.is_set():
                        should_stop = True
                        break
                    frames = stack_row_to_code_v4_records(row, max_code_file_bytes=max_code_file_bytes)
                    for frame in frames:
                        if not out.add_frame(frame):
                            should_stop = True
                            break
                    if should_stop:
                        break
            finally:
                del rows, batch
            if should_stop:
                break
        out.finish()
    except Exception as e:
        out.abort_current()
        if not stop_event.is_set():
            _queue_emit(out_queue, stop_event, TaskResult(
                task.index, _task_key(task), "", 0, 0, 0, {}, {},
                chunk_index=out.chunk_index, final=True,
                error=f"{type(e).__name__}: {e}",
            ))
        raise


class _CodeV4ShardAccumulator:
    def __init__(self, source: str, out_dir: Path, target_bytes: int, start_index: int = 0,
                 fsync: bool = False, hard_cap_bytes: Optional[int] = None):
        self.source = source
        self.out_dir = out_dir
        minimum = CODE_PACK_HEADER_BYTES + CODE_FRAME.size + 1
        self.target_bytes = max(minimum, int(target_bytes))
        self.hard_cap_bytes = max(self.target_bytes, int(hard_cap_bytes) if hard_cap_bytes is not None else self.target_bytes)
        self.index = int(start_index)
        self.fsync = fsync
        self.f = None
        self.tmp_path: Optional[Path] = None
        self.file_bytes = 0
        self.frames = 0
        self.records = 0
        self.text_bytes = 0
        self.aux_text_bytes = 0
        self.repo_manifests = 0
        self.aux_files = 0
        self.subtype_records = Counter()
        self.subtype_text_bytes = Counter()
        self._completed: List[Dict[str, Any]] = []

    def set_target_bytes(self, value: int) -> None:
        minimum = CODE_PACK_HEADER_BYTES + CODE_FRAME.size + 1
        self.target_bytes = min(self.hard_cap_bytes, max(minimum, int(value)))

    def _ensure_open(self) -> None:
        if self.f is not None:
            return
        d = self.out_dir / self.source
        d.mkdir(parents=True, exist_ok=True)
        final = d / f"{self.source}-{self.index:05d}.pack"
        self.tmp_path = final.with_name(final.name + f".tmp.{os.getpid()}")
        self.f = self.tmp_path.open("wb")
        self.f.write(CODE_PACK_MAGIC)
        self.file_bytes = CODE_PACK_HEADER_BYTES

    @property
    def has_data(self) -> bool:
        return self.frames > 0

    @property
    def full(self) -> bool:
        return self.file_bytes >= self.target_bytes

    def _flush_current(self) -> Optional[Dict[str, Any]]:
        if not self.has_data:
            return None
        assert self.f is not None and self.tmp_path is not None
        final = self.out_dir / self.source / f"{self.source}-{self.index:05d}.pack"
        self.f.flush()
        if self.fsync:
            os.fsync(self.f.fileno())
        self.f.close()
        os.replace(self.tmp_path, final)
        meta = {
            "path": str(final), "format": "PTCODE5",
            "frames": self.frames, "repo_manifests": self.repo_manifests,
            "countable_files": self.records, "aux_files": self.aux_files,
            "records": self.records, "text_bytes": self.text_bytes,
            "aux_text_bytes": self.aux_text_bytes, "file_bytes": final.stat().st_size,
            "subtype_records": {str(k): int(v) for k, v in sorted(self.subtype_records.items())},
            "subtype_text_bytes": {str(k): int(v) for k, v in sorted(self.subtype_text_bytes.items())},
        }
        if meta["file_bytes"] > self.hard_cap_bytes:
            raise RuntimeError(f"hard PTCODE5 pack cap violated: {meta['file_bytes']} > {self.hard_cap_bytes}: {final}")
        meta.update(_strict_pack_integrity(final, self.source, self.hard_cap_bytes))
        self.index += 1
        self.f = None
        self.tmp_path = None
        self.file_bytes = 0
        self.frames = 0
        self.records = 0
        self.text_bytes = 0
        self.aux_text_bytes = 0
        self.repo_manifests = 0
        self.aux_files = 0
        self.subtype_records.clear()
        self.subtype_text_bytes.clear()
        return meta

    def _append_parts(self, header: bytes, kind: int, subtype: int, flags: int, meta: bytes, text: bytes) -> None:
        frame_size = len(header) + len(meta) + len(text)
        if CODE_PACK_HEADER_BYTES + frame_size > self.hard_cap_bytes:
            raise ValueError(
                f"single PTCODE5 frame ({frame_size} bytes + header) exceeds hard pack cap "
                f"{self.hard_cap_bytes}; increase --pack-mib or lower repository/file size"
            )
        if self.has_data and self.file_bytes + frame_size > self.target_bytes:
            done = self._flush_current()
            if done:
                self._completed.append(done)
        self._ensure_open()
        assert self.f is not None
        self.f.write(header); self.f.write(meta); self.f.write(text)
        self.file_bytes += frame_size
        self.frames += 1
        if kind == CODE_KIND_REPO:
            self.repo_manifests += 1
        elif kind == CODE_KIND_FILE:
            if flags & CODE_FLAG_COUNT_FOR_QUOTA:
                self.records += 1
                self.text_bytes += len(text)
                self.subtype_records[int(subtype)] += 1
                self.subtype_text_bytes[int(subtype)] += len(text)
            else:
                self.aux_files += 1
                self.aux_text_bytes += len(text)
        if self.file_bytes >= self.target_bytes:
            done = self._flush_current()
            if done:
                self._completed.append(done)

    def _consume_spool(self, path: Path) -> None:
        with path.open("rb") as src:
            while True:
                header = src.read(CODE_FRAME.size)
                if not header:
                    break
                if len(header) != CODE_FRAME.size:
                    raise RuntimeError(f"truncated PTCODE5 worker spool header: {path}")
                kind, subtype, flags, meta_n, text_n = CODE_FRAME.unpack(header)
                meta = src.read(meta_n)
                text = src.read(text_n)
                if len(meta) != meta_n or len(text) != text_n:
                    raise RuntimeError(f"truncated PTCODE5 worker spool payload: {path}")
                self._append_parts(header, int(kind), int(subtype), int(flags), meta, text)

    def add(self, r: TaskResult) -> None:
        if not r.packed_bytes:
            if r.spool_path:
                Path(r.spool_path).unlink(missing_ok=True)
            return
        if not r.spool_path:
            raise RuntimeError(f"task {r.index} has PTCODE5 bytes but no spool path")
        spool = Path(r.spool_path)
        try:
            self._consume_spool(spool)
        finally:
            spool.unlink(missing_ok=True)

    def pop_completed(self) -> List[Dict[str, Any]]:
        out, self._completed = self._completed, []
        return out

    def flush(self) -> Optional[Dict[str, Any]]:
        return self._flush_current()

    def abort(self) -> None:
        if self.f is not None:
            try: self.f.close()
            except Exception: pass
        if self.tmp_path is not None:
            self.tmp_path.unlink(missing_ok=True)
        self.f = None; self.tmp_path = None
        self.file_bytes = 0; self.frames = 0; self.records = 0; self.text_bytes = 0
        self.aux_text_bytes = 0; self.repo_manifests = 0; self.aux_files = 0
        self.subtype_records.clear(); self.subtype_text_bytes.clear()


class ShardAccumulator:
    """V4 dispatcher: PTCODE5 for code, original PTPACK1 for every other source."""
    def __init__(self, source: str, out_dir: Path, target_bytes: int, start_index: int = 0,
                 fsync: bool = False, hard_cap_bytes: Optional[int] = None):
        impl = _CodeV4ShardAccumulator if source == "code" else _V3ShardAccumulator
        self._impl = impl(source, out_dir, target_bytes, start_index, fsync, hard_cap_bytes)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._impl, name)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    print("=== strict pretraining stage ===")
    print(f"revision        : {SCRIPT_REVISION}")
    print(f"phase           : {args.phase}")
    print(f"target tokens   : {args.target_tokens:,}")
    print(f"workers/readers : {args.workers}/{args.io_readers}")
    print(f"pack hard cap   : {args.pack_mib} MiB")
    print(f"spool block     : {args.spool_block_mib:.1f} MiB")
    print(f"RAM limit       : {args.ram_limit_gib:.1f} GiB")
    print(f"count batch     : {args.count_batch_mib:.1f} MiB / {args.count_batch_size} records")
    print(f"parquet batch   : {args.parquet_batch_rows} rows")
    print(f"stall timeout   : {args.stall_timeout_s:.1f} s")
    print(f"stage dir       : {args.stage_dir}")
    print("code pack       : PTCODE5 (repo/file metadata preserved)")
    if args.phase == "all":
        report = run_all(args)
    else:
        tok, _ = load_frozen_tokenizer(args.tokenizer, args.tokenizer_threads)
        report = recount_stage(args, tok)
    print_report(report)
    if args.phase == "all" and args.max_tasks is None:
        selected = set(args.sources.split(",")) if args.sources else set(SOURCE_ORDER)
        incomplete = [x for x in SOURCE_ORDER if x in selected and not report["sources"][x]["complete"]]
        if incomplete:
            print(
                f"WARN: staging buffer target not fully met for {incomplete}. "
                "This is not silently rebalanced or repeated; build.py plan/trim will "
                "fail closed if the requested final source/language quotas cannot be met."
            )
    return 0




# Public V5 names. The on-disk frame layout stays deliberately frame-compatible
# with V4; the magic/version and subtype vocabulary changed, so readers must
# still check PTCODE5 explicitly.
CodeV5Frame = CodeV4Frame
encode_code_v5_frame = encode_code_v4_frame
iter_code_v5_frames = iter_code_v4_frames
write_code_v5_pack = write_code_v4_pack
stack_row_to_code_v5_records = stack_row_to_code_v4_records
iter_code_v5_repositories = iter_code_v4_repositories


if __name__ == "__main__":
    raise SystemExit(main())
