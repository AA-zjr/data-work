# Pretraining Data Pipeline

正式链路：`PTSTAGE1/PTCODE5 -> inventory -> dedup -> semantic blocks -> content plan -> SBSTRUCT1 -> online FIM`。

## 文件

- `fineweb_modelscope.py`：固定魔搭 FineWeb-Edu `sample/350BT` 分片清单，按 SHA-256 校验；仅按需补下载缺失或损坏的分片。
- `stage.py`：从原始数据构建 strict PTSTAGE1。默认 staging 为最终 token 数的 1.25x；Stack 保留 repo/file provenance。
- `build.py`：去重、AST/语义关系、repo context、结构化样本、精确 base-token trim。
- `run.sh`：一键运行 Stage + Build。
- `fim.py` / `fim.json`：训练时动态 Code/Math FIM。
- `validate.py`：SBSTRUCT1 + tokenizer + FIM 严格验证。
- `loader.py`：PyTorch DataLoader 接入参考。

## 默认数据比例

最终 base tokens：FineWeb-Edu 45%，Nemotron Math 12.5%，FineMath 12.5%，Stack-v3 20%，peS2o 7%，English Wikipedia 3%。

Stack 支持 28 种编程语言，并保留固定语言配比。Code 20% 内部比例为：Python 17%，C 5%，C++ 9%，C# 5%，Java 9%，JavaScript 8%，TypeScript 9%，TSX 2%，Go 8%，Rust 5%，Ruby 2%，PHP 3%，Kotlin 2%，Swift 2%，Scala 1.5%，Lua 1.5%，Shell 3%，Dart 1%，Julia 0.8%，R 1.5%，Haskell 0.8%，Elixir 0.7%，Erlang 0.5%，OCaml 0.5%，Clojure 0.5%，Objective-C 0.7%，Zig 0.5%，Solidity 0.5%。Planner 和最终 trim 都按这套比例控制。

## 直接运行

FineWeb-Edu 默认改为魔搭 `HuggingFaceFW/fineweb-edu` 的 `sample/350BT`，本机目录 `/mnt/D/data/fineweb-edu-ms-200B/sample/350BT`。先校验选中的 270 个分片（已有文件不会重新下载）；损坏或缺失时才补下载：

```bash
python fineweb_modelscope.py --limit 1  # 快速抽检；不能用于正式 Stage
python fineweb_modelscope.py             # 全量校验并生成 Stage 所需的验证记录
python fineweb_modelscope.py --download-missing  # 仅在校验失败时运行
```

清单仅选同一个 `sample/350BT` 子集，拒绝重复 SHA、未选中的 Parquet 和发生变更的文件；不要把 `sample/100BT`、HF shuffled 或其他副本放进该目录。完整 SHA 校验后 Stage 再次检查文件大小/修改时间，修改文件后必须重新校验。魔搭源与已处理的旧 HF 数据可能仍有**文档级重合**，不能靠换下载站点解决；`build.py` 默认对 base + augmentation 进行跨源精确及近重复去重（不要使用 `--near-dedup off`）。

默认构建最终 1B，Stage 自动准备 1.25B：

```bash
chmod +x run.sh
./run.sh
```

只改最终 token 数：

```bash
./run.sh 1000000000
```

常用可选覆盖参数：

```bash
WORKERS=32 CODE_WORKERS=12 RENDER_WORKERS=16 ./run.sh
SKIP_STAGE=1 ./run.sh
REBUILD=1 ./run.sh
```

在原 250B 底座上增加魔搭 FineWeb-Edu 时，**不要**覆盖旧的 Stage/Work/输出目录；旧 HF Stage 的 raw checkpoint 不能在更换原始数据目录后原地续跑。只处理 FineWeb-Edu 的 Stage + 跨旧底座去重（不会启动已知缺量的完整渲染流程）：

```bash
python stage.py --sources general --target-tokens 300000000000 \
  --base-stage-dir "$PWD/stage_312b" --stage-dir /mnt/D/data/stage_ms_aug
python build.py --phase dedup --target-tokens 250000000000 \
  --base-work-dir "$PWD/work_250b" --stage-dir /mnt/D/data/stage_ms_aug \
  --work-dir /mnt/D/data/work_ms_aug --output-dir /mnt/D/data/pretrain_ms_250b
```

已有 `stage_aug_v13`/`work_aug_v13` 仍保留，但此示例从原始 `stage_312b`/`work_250b` 底座重新构建增量，**不会**把两次增量叠加成一份。旧 HF 底座已写入的内容不会被新数据修复；如果已确认其文档错误而不仅是下载不完整，需单独重建底座。当前 250B 全流程还有 Wiki/Code 数据量缺口，FineWeb 补足不能单独解决这些缺口。补齐其余数据后可用独立的 `STAGE_DIR` / `WORK_DIR` / `OUT_DIR` 配合 `BASE_STAGE_DIR` / `BASE_WORK_DIR` 运行 `run.sh`。

路径也可以直接用环境变量覆盖：`TOKENIZER`、`FINEWEB_ROOT`、`STAGE_DIR`、`WORK_DIR`、`OUT_DIR`，增量构建还可设置 `BASE_STAGE_DIR`、`BASE_WORK_DIR`。默认是：

```text
TOKENIZER=/home/zjr1236/data/tokenizer_64k/tokenizer.json
FINEWEB_ROOT=/mnt/D/data/fineweb-edu-ms-200B/sample/350BT
STAGE_DIR=/home/zjr1236/data/pretrain_read_count_1p25b_strict
WORK_DIR=/home/zjr1236/data/semantic_block_1b_strict
OUT_DIR=/home/zjr1236/data/pretrain_semantic_structured_1b
```

`stage.py` 默认会使用 `/home/zjr1236/data/raw` 查找其他数据，Stack 为 `/mnt/D/data/stack-v3`，Nemotron 为 `/home/zjr1236/data/raw/Nemotron-CC-Math-v1/4plus_MIND`；Wikipedia 只解析 English snapshot。FineWeb 默认不再自动发现旧 HF `fineweb_edu_100bt`。如用其他路径保存魔搭分片，下载与 Stage 都需指定对应目录（Stage 的 `FINEWEB_ROOT` 指向清单根目录下的 `sample/350BT`）。

## 巨型仓库保护

`build.py` 对巨型 monorepo 使用分层保护，不改变 28 语言固定配比：

- `>20,000` 个文件的 repo 进入 giant-repo 模式；repo 元数据行使用确定性 bottom-k，最多保留 12,000 个文件描述，避免单仓库撑爆 worker RAM。
- giant repo 最多分析 4,096 个 target 文件，并优先保证仓库中已有语言至少有代表样本；最多再读取 512 个 import 邻居作为 coarse related context。
- 大目录不再使用 `tree[:200]` 字典序截断。超过 500 个路径时先生成带文件计数的两层 hierarchical tree summary；最终 16K 样本优先保留 target 所在目录 neighborhood，再填全局目录摘要。
- 单 target 所属目录最多保留 24 个 neighborhood paths；最终仍受 `tree_limit=200`、`max_related=8`、`max_sample_tokens=16384` 约束。
- 单仓库 planner 上限仍为 Code quota 的 1%，因此巨型仓库不会主导训练分布。

这些阈值位于 `build.py` 的 `DEFAULT_CONFIG['semantic']`，正常使用 `./run.sh` 不需要额外参数。升级仅使 semantic 及其下游 cache 失效，inventory/dedup cache contract 保持兼容。

## 验证

```bash
python validate.py \
  --data-dir /home/zjr1236/data/pretrain_semantic_structured_1b \
  --tokenizer /home/zjr1236/data/tokenizer_64k/tokenizer.json \
  --transform-samples 10000
```

一次性全量 base-token 审计：

```bash
python validate.py \
  --data-dir /home/zjr1236/data/pretrain_semantic_structured_1b \
  --tokenizer /home/zjr1236/data/tokenizer_64k/tokenizer.json \
  --transform-samples 10000 \
  --full-token-check
```

## FIM

预处理不提前改写 FIM。SBSTRUCT1 保存 target/context/AST FIM candidates；训练时 `fim.py` 决定 causal / AST FIM / semantic FIM / test-aware FIM。默认 Code FIM 总预算 50%，Math FIM 25%。**使用 `fim.py` 后不要再执行旧 FIM 分支，否则会重复插入 PSM special tokens。**

## 依赖

```bash
pip install tokenizers pyarrow zstandard msgpack numpy tree-sitter tree-sitter-language-pack modelscope
```
