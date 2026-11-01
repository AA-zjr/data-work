#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOKENS="${1:-${TOKENS:-1000000000}}"
STAGE_TOKENS="${STAGE_TOKENS:-$(( (TOKENS * 125 + 99) / 100 ))}"
WORKERS="${WORKERS:-32}"
CODE_WORKERS="${CODE_WORKERS:-12}"
RENDER_WORKERS="${RENDER_WORKERS:-16}"

TOKENIZER="${TOKENIZER:-/home/zjr1236/data/tokenizer_64k/tokenizer.json}"
FINEWEB_ROOT="${FINEWEB_ROOT:-/mnt/D/data/fineweb-edu-ms-200B/sample/350BT}"
STAGE_DIR="${STAGE_DIR:-/home/zjr1236/data/pretrain_read_count_1p25b_strict}"
WORK_DIR="${WORK_DIR:-/home/zjr1236/data/semantic_block_1b_strict}"
OUT_DIR="${OUT_DIR:-/home/zjr1236/data/pretrain_semantic_structured_1b}"

stage_flags=()
build_flags=()
if [[ -n "${BASE_WORK_DIR:-}" ]]; then
  build_flags+=(--base-work-dir "$BASE_WORK_DIR")
fi
if [[ -n "${BASE_STAGE_DIR:-}" ]]; then
  stage_flags+=(--base-stage-dir "$BASE_STAGE_DIR")
fi
if [[ "${REBUILD:-0}" == "1" ]]; then
  stage_flags+=(--rebuild-stage)
  build_flags+=(--rebuild)
fi

printf 'final=%s  stage=%s\n' "$TOKENS" "$STAGE_TOKENS"
printf 'stage=%s\nwork=%s\nout=%s\n' "$STAGE_DIR" "$WORK_DIR" "$OUT_DIR"
printf 'fineweb=%s\n' "$FINEWEB_ROOT"

if [[ "${SKIP_STAGE:-0}" != "1" ]]; then
  python "$HERE/stage.py" \
    --phase all \
    --target-tokens "$STAGE_TOKENS" \
    --tokenizer "$TOKENIZER" \
    --fineweb-root "$FINEWEB_ROOT" \
    --stage-dir "$STAGE_DIR" \
    --workers "$WORKERS" \
    --tokenizer-threads "$WORKERS" \
    "${stage_flags[@]}"
fi

python "$HERE/build.py" \
  --phase all \
  --stage-dir "$STAGE_DIR" \
  --work-dir "$WORK_DIR" \
  --output-dir "$OUT_DIR" \
  --tokenizer "$TOKENIZER" \
  --target-tokens "$TOKENS" \
  --workers "$WORKERS" \
  --code-workers "$CODE_WORKERS" \
  --render-workers "$RENDER_WORKERS" \
  --output-format structured_trainpack \
  "${build_flags[@]}"

printf '\nDONE: %s\n' "$OUT_DIR"
printf 'Validate: python %q --data-dir %q --tokenizer %q --transform-samples 10000\n' \
  "$HERE/validate.py" "$OUT_DIR" "$TOKENIZER"
