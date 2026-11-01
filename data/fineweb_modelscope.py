#!/usr/bin/env python3
"""Verify and, when requested, repair a pinned ModelScope FineWeb-Edu shard selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable

DATASET_ID = "HuggingFaceFW/fineweb-edu"
SOURCE_PREFIX = "sample/350BT"
REVISION = "master"
DEFAULT_ROOT = Path("/mnt/D/data/fineweb-edu-ms-200B")
SHARD_RE = re.compile(r"^sample/350BT/\d{3}_\d{5}\.parquet$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            delete=False,
        ) as f:
            tmp = Path(f.name)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _validate_entry(entry: dict[str, Any]) -> tuple[str, int, str]:
    path = entry.get("path")
    size = entry.get("size")
    sha = entry.get("sha256")
    if not isinstance(path, str) or not SHARD_RE.fullmatch(path):
        raise ValueError(f"not a FineWeb-Edu {SOURCE_PREFIX} parquet shard: {path!r}")
    if type(size) is not int or size <= 0:
        raise ValueError(f"invalid shard size for {path}: {size!r}")
    if not isinstance(sha, str) or not SHA_RE.fullmatch(sha):
        raise ValueError(f"missing/invalid ModelScope sha256 for {path}: {sha!r}")
    return path, size, sha


def remote_shards(api: Any) -> dict[str, dict[str, Any]]:
    shards: dict[str, dict[str, Any]] = {}
    page = 1
    while True:
        rows = api.get_dataset_files(
            DATASET_ID, revision=REVISION, root_path=SOURCE_PREFIX,
            recursive=True, page_number=page, page_size=500,
        )
        if not isinstance(rows, list):
            raise RuntimeError("ModelScope returned an invalid dataset file listing")
        for row in rows:
            path = row.get("Path")
            if row.get("Type") != "blob" or not isinstance(path, str) or not path.endswith(".parquet"):
                continue
            entry = {"path": path, "size": row.get("Size"), "sha256": row.get("Sha256")}
            name, _, _ = _validate_entry(entry)
            if name in shards:
                raise RuntimeError(f"duplicate ModelScope shard path: {name}")
            shards[name] = entry
        if len(rows) < 500:
            break
        page += 1
    if not shards:
        raise RuntimeError(f"no parquet shards in ModelScope {DATASET_ID}/{SOURCE_PREFIX}")
    return shards


def selected_shards(root: Path, remote: dict[str, dict[str, Any]], file_count: int) -> list[dict[str, Any]]:
    manifest = root / "selection_manifest.json"
    new_selection = None
    if manifest.is_file():
        selection = json.loads(manifest.read_text(encoding="utf-8"))
        if selection.get("repo_id") != DATASET_ID or selection.get("source_prefix") != SOURCE_PREFIX:
            raise RuntimeError(f"selection manifest is not {DATASET_ID}/{SOURCE_PREFIX}: {manifest}")
        if selection.get("revision", REVISION) != REVISION:
            raise RuntimeError(f"selection manifest revision mismatch: {manifest}")
        files = selection.get("selected_files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(f"empty/invalid selection manifest: {manifest}")
    else:
        if file_count <= 0 or file_count > len(remote):
            raise ValueError(f"--file-count must be between 1 and {len(remote)}")
        files = [remote[path] for path in sorted(remote)[:file_count]]
        new_selection = {
            "repo_id": DATASET_ID, "source_prefix": SOURCE_PREFIX, "revision": REVISION,
            "selected_files": files, "selected_parquet_bytes": sum(e["size"] for e in files),
        }
    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for entry in files:
        path, size, sha = _validate_entry(entry)
        if path in seen_paths or sha in seen_hashes:
            raise RuntimeError(f"repeated shard path or identical shard contents in selection: {path}")
        seen_paths.add(path)
        seen_hashes.add(sha)
        actual = remote.get(path)
        if actual is None or actual["size"] != size or actual["sha256"] != sha:
            raise RuntimeError(f"ModelScope shard changed or is unavailable: {path}")
    if new_selection is not None:
        atomic_json(manifest, new_selection)
    return files


def _check_no_extra_shards(source_root: Path, selected: set[str]) -> None:
    for shard in source_root.rglob("*.parquet"):
        relative = shard.relative_to(source_root)
        if ".cache" in relative.parts:
            continue
        if f"{SOURCE_PREFIX}/{relative.as_posix()}" not in selected:
            raise RuntimeError(f"unselected parquet would be ingested by Stage: {shard}")


def _check_file(path: Path, entry: dict[str, Any]) -> str | None:
    if path.is_symlink():
        return "symlink instead of downloaded file"
    if not path.is_file():
        return "missing"
    if path.stat().st_size != entry["size"]:
        return "size mismatch"
    if sha256_file(path) != entry["sha256"]:
        return "sha256 mismatch"
    return None


def audit(
    root: Path, files: list[dict[str, Any]], *,
    download_missing: bool = False, limit: int | None = None,
    download_file: Callable[..., str] | None = None,
) -> tuple[int, int]:
    root = root.expanduser().resolve()
    manifest = root / "selection_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be > 0")
    if not files:
        raise ValueError("no shards selected for audit")
    source_root = root / SOURCE_PREFIX
    report_path = root / "verification_report.json"
    report_path.unlink(missing_ok=True)
    selected = {_validate_entry(e)[0] for e in files}
    if len(selected) != len(files):
        raise RuntimeError("selection contains duplicate shard paths")
    _check_no_extra_shards(source_root, selected)
    if download_missing and download_file is None:
        from modelscope.hub.file_download import dataset_file_download
        download_file = dataset_file_download

    checked: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    repaired = 0
    for index, entry in enumerate(files[:limit], start=1):
        path, _, _ = _validate_entry(entry)
        dest = root / path
        problem = _check_file(dest, entry)
        if problem and download_missing:
            scratch = source_root / ".cache"
            scratch.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="repair-", dir=scratch) as temp:
                downloaded = Path(download_file(
                    DATASET_ID, path, revision=REVISION, local_dir=temp,
                ))
                if _check_file(downloaded, entry):
                    raise RuntimeError(f"download from ModelScope failed checksum verification: {path}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(downloaded, dest)
            repaired += 1
            problem = None
        if problem:
            failures.append(f"{path}: {problem}")
        else:
            stat = dest.stat()
            checked[path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        if index % 10 == 0 or index == len(files):
            print(f"FineWeb-Edu checked {index}/{len(files)} shards (repaired {repaired})", flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} FineWeb-Edu shard(s) missing or corrupt: "
            + "; ".join(failures[:5])
            + (". Run with --download-missing to repair them." if not download_missing else "")
        )
    if len(checked) == len(files):
        atomic_json(report_path, {
            "manifest_sha256": sha256_file(manifest),
            "files": checked,
        })
    return len(checked), repaired


def assert_verified_selection(source_root: Path) -> None:
    source_root = source_root.expanduser().resolve()
    root = source_root.parent.parent
    if source_root != root / SOURCE_PREFIX:
        raise RuntimeError(f"not a {SOURCE_PREFIX} data directory: {source_root}")
    manifest = root / "selection_manifest.json"
    report_path = root / "verification_report.json"
    if not manifest.is_file() or not report_path.is_file():
        raise RuntimeError(f"ModelScope FineWeb-Edu must be verified before staging: python fineweb_modelscope.py --root {root}")
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    if selection.get("repo_id") != DATASET_ID or selection.get("source_prefix") != SOURCE_PREFIX:
        raise RuntimeError(f"wrong ModelScope FineWeb-Edu selection: {manifest}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != sha256_file(manifest):
        raise RuntimeError("ModelScope selection changed since its checksum audit")
    files = selection.get("selected_files")
    if not isinstance(files, list) or not files:
        raise RuntimeError(f"empty ModelScope selection: {manifest}")
    selected = {_validate_entry(e)[0] for e in files}
    if len(selected) != len(files) or set(report.get("files", {})) != selected:
        raise RuntimeError("ModelScope verification does not cover every unique selected shard")
    _check_no_extra_shards(source_root, selected)
    for path in selected:
        shard = root / path
        if shard.is_symlink() or not shard.is_file():
            raise RuntimeError(f"ModelScope shard missing or replaced since its checksum audit: {path}")
        stat = shard.stat()
        if report["files"][path] != {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
            raise RuntimeError(f"ModelScope shard changed since its checksum audit: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--file-count", type=int, default=270, help="new selection size; ignored if a selection already exists")
    parser.add_argument("--limit", type=int, default=None, help="audit only the first N files (never marks selection verified)")
    parser.add_argument("--download-missing", action="store_true", help="repair only missing/corrupt shards from ModelScope")
    args = parser.parse_args()

    from modelscope.hub.api import HubApi
    root = args.root.expanduser().resolve()
    remote = remote_shards(HubApi())
    files = selected_shards(root, remote, args.file_count)
    checked, repaired = audit(root, files, download_missing=args.download_missing, limit=args.limit)
    print(
        f"ModelScope {DATASET_ID}/{SOURCE_PREFIX}: verified {checked}/{len(files)} "
        f"unique shards, repaired {repaired}; source={root / SOURCE_PREFIX}"
    )
    if checked < len(files):
        print("Partial audit only; Stage will refuse this source until every shard is verified.")


if __name__ == "__main__":
    main()
