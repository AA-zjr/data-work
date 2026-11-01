import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import fineweb_modelscope as ms
from stage import list_source_files


class FakeApi:
    def __init__(self, rows):
        self.rows = rows

    def get_dataset_files(self, repo_id, **kwargs):
        return self.rows if kwargs["page_number"] == 1 else []


class FineWebModelScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = "sample/350BT/000_00000.parquet"
        self.payload = b"PAR1test sample bytesPAR1"
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.remote = ms.remote_shards(FakeApi([{
            "Path": self.path, "Type": "blob", "Size": len(self.payload), "Sha256": self.sha,
        }]))
        self.files = ms.selected_shards(self.root, self.remote, 1)

    def test_verify_repair_and_reject_changed_file(self):
        def download(dataset_id, path, **kwargs):
            dest = Path(kwargs["local_dir"]) / path
            dest.parent.mkdir(parents=True)
            dest.write_bytes(self.payload)
            return str(dest)

        with self.assertRaisesRegex(RuntimeError, "missing"):
            ms.audit(self.root, self.files)
        self.assertEqual(ms.audit(self.root, self.files, download_missing=True, download_file=download), (1, 1))
        ms.assert_verified_selection(self.root / ms.SOURCE_PREFIX)
        self.assertEqual(ms.audit(self.root, self.files), (1, 0))
        path = self.root / self.path
        path.write_bytes(b"X" * len(self.payload))
        with self.assertRaisesRegex(RuntimeError, "sha256 mismatch"):
            ms.audit(self.root, self.files)
        with self.assertRaisesRegex(RuntimeError, "must be verified"):
            ms.assert_verified_selection(self.root / ms.SOURCE_PREFIX)

    def test_reject_unselected_file_and_ignore_cache(self):
        source = self.root / ms.SOURCE_PREFIX
        selected = self.root / self.path
        selected.parent.mkdir(parents=True)
        selected.write_bytes(self.payload)
        cache = source / ".cache" / "partial.parquet"
        cache.parent.mkdir(parents=True)
        cache.write_bytes(self.payload)
        self.assertEqual(list_source_files(source), [selected])
        extra = source / "000_00001.parquet"
        extra.write_bytes(self.payload)
        with self.assertRaisesRegex(RuntimeError, "unselected parquet"):
            ms.audit(self.root, self.files)

    def test_pin_remote_metadata_and_deduplicate_selection(self):
        with self.assertRaisesRegex(RuntimeError, "changed"):
            ms.selected_shards(self.root, {
                self.path: {"path": self.path, "size": len(self.payload), "sha256": "f" * 64},
            }, 1)
        manifest = self.root / "selection_manifest.json"
        selection = json.loads(manifest.read_text())
        selection["selected_files"].append(dict(selection["selected_files"][0]))
        manifest.write_text(json.dumps(selection))
        with self.assertRaisesRegex(RuntimeError, "repeated shard"):
            ms.selected_shards(self.root, self.remote, 1)

    def test_reject_identical_shards_before_creating_selection(self):
        manifest = self.root / "selection_manifest.json"
        manifest.unlink()
        second_path = "sample/350BT/000_00001.parquet"
        remote = dict(self.remote)
        remote[second_path] = {"path": second_path, "size": len(self.payload), "sha256": self.sha}
        with self.assertRaisesRegex(RuntimeError, "identical shard contents"):
            ms.selected_shards(self.root, remote, 2)
        self.assertFalse(manifest.exists())

    def test_reject_symlink_to_other_dataset(self):
        source = self.root / ms.SOURCE_PREFIX
        source.mkdir(parents=True)
        external = self.root / "old_hf.parquet"
        external.write_bytes(self.payload)
        (self.root / self.path).symlink_to(external)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            ms.audit(self.root, self.files)

    def test_partial_audit_does_not_mark_verified(self):
        second_path = "sample/350BT/000_00001.parquet"
        files = self.files + [{"path": second_path, "size": len(self.payload), "sha256": "a" * 64}]
        (self.root / "selection_manifest.json").write_text(json.dumps({
            "repo_id": ms.DATASET_ID, "source_prefix": ms.SOURCE_PREFIX, "selected_files": files,
        }))
        dest = self.root / self.path
        dest.parent.mkdir(parents=True)
        dest.write_bytes(self.payload)
        self.assertEqual(ms.audit(self.root, files, limit=1), (1, 0))
        with self.assertRaisesRegex(RuntimeError, "must be verified"):
            ms.assert_verified_selection(self.root / ms.SOURCE_PREFIX)


if __name__ == "__main__":
    unittest.main()
