# data-work

A reproducible pipeline for building large-scale, deduplicated, structure-aware
pretraining corpora — and the exact code used to build
[**250b-clean-pretrain**](https://modelscope.cn/datasets/zzzibbr/250b-clean-pretrain)
(250B tokens, 148.3M samples, 10,013 shards, 1.13 TB).

The output format keeps document / repository structure instead of flattening
everything into a single concatenated token stream, so FIM (fill-in-the-middle)
and other training-time transforms can be applied **online** rather than baked in.

## Pipeline

```
raw sources
  │  stage.py      strict PTSTAGE1 packs (tokenized, provenance-tagged)
  ▼
stage/*.pack
  │  build.py --phase inventory    record index + quality signals
  │  build.py --phase dedup        exact + near-dup (MinHash/LSH) with pack-ordered I/O
  │  build.py --phase semantic     chunks, repo parsing (tree-sitter), AST/semantic FIM candidates
  │  build.py --phase plan         quota allocation across sources / languages / repos
  │  build.py --phase render       SBSTRUCT1 structured samples
  │  build.py --phase trim         exact token-budget trim -> .trainpack shards
  ▼
out/*.trainpack         (SBSTRUCT1: msgpack records, structure preserved)
   └ fim.py             online causal / AST-FIM / semantic-FIM / test-aware transforms
```

Every stage is resumable: stage signatures are derived from the config, the input
manifest and the previous stage's `_SUCCESS.json`, and completed partitions are
reused instead of recomputed.

## Repository layout

```
data/
  stage.py               raw -> strict PTSTAGE1
  build.py               inventory/dedup/semantic/plan/render/trim (single file, no package deps)
  fim.py, fim.json       online FIM transforms and their default policy
  validate.py            SBSTRUCT1 + tokenizer + FIM validator (supports a full 250B reseed)
  loader.py              minimal PyTorch IterableDataset reference
  fineweb_modelscope.py  pinned FineWeb-Edu shard manifest with SHA-256 verification
  run.sh                 one-shot stage + build
tokenizer/
  tokenizer_pipeline.py  BPE training script
  benchmark_tokenizer.py compression benchmarks
  tokenizer_64k/         64k BPE tokenizer actually used (13 special tokens, ids 0-12)
tests/
  test_fineweb_modelscope.py
```

## Quick start

```bash
pip install tokenizers pyarrow zstandard msgpack numpy tree-sitter tree-sitter-language-pack modelscope

# 1B tokens by default; the stage is sized to 1.25x the final token budget
./data/run.sh 1000000000

# verify
python data/validate.py --data-dir "$OUT_DIR" \
  --tokenizer tokenizer/tokenizer_64k/tokenizer.json --transform-samples 10000
```

Paths and parallelism are environment-overridable:

```bash
TOKENIZER=... STAGE_DIR=... WORK_DIR=... OUT_DIR=... \
WORKERS=32 CODE_WORKERS=12 RENDER_WORKERS=16 ./data/run.sh
```

`FINEWEB_ROOT` points at the ModelScope mirror of `HuggingFaceFW/fineweb-edu`
`sample/350BT`; `fineweb_modelscope.py` verifies every shard by SHA-256 before
`stage.py` consumes it. Other sources (Nemotron-CC-Math, FineMath, peS2o,
English Wikipedia, The Stack v3) are discovered under the raw root — see
`data/README.md` for the full directory contract.

## Output format

`SBSTRUCT1` — magic `SBSTRUCT1\n`, then repeated little-endian `<I` length +
MessagePack records:

```python
{
  "sample_id": ...,
  "candidate_id": ...,
  "source": "general | math_nemotron | math_finemath | academic | wiki | code",
  "tokens": int,            # equals "base_tokens"
  "score": float,
  "capabilities": [...],
  "primary_language": "Python",
  "structured": {
      "kind": "document",
      "parts": [...],              # ordered text parts with provenance
      "fim_candidate_indices": [...]   # spans eligible for FIM, chosen at train time
  }
}
```

Code records use `kind: "code"` with `repo`, `target` and `context` blocks instead.
`fim.py` turns these into `causal | ordinary_fim | math_fim | ast_fim |
semantic_fim | test_aware_fim` training sequences with the frozen 64k tokenizer.

## Notes and limitations

- Quotas are enforced exactly at trim time; every source/language targets its
  configured share of base tokens, with repo-level caps so one repository cannot
  dominate the code mixture.
- Dedup is exact + MinHash/LSH near-dup across sources; code is additionally
  deduplicated at repository/file level, not only by sample id.
- The released 250B corpus is English-only (~99.9% Latin script) and code-heavy
  (38.7%); see the dataset card for the measured duplication and language-mix
  caveats before using it as-is.
- `build.py` is intentionally dependency-light but large; it is the single source
  of truth for stage contents and cache signatures.

## License

MIT — see [LICENSE](LICENSE). The datasets this pipeline was used on carry their
own licenses; the released 250B mix is distributed under `license: other` with
per-source attribution in its dataset card.
