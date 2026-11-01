#!/usr/bin/env python3
"""Minimal integration with the existing training-side FIM contract.

Important: once OnlineFIMTransformer/StructuredFIMCollator is used, DO NOT run the old
`encode_fim_text()` branch again on the returned ids. This module already inserts the
same PSM markers exactly once.
"""
from pathlib import Path
from torch.utils.data import DataLoader
from fim import (
    StructuredTrainpackIterableDataset,
    StructuredFIMCollator,
    config_from_legacy_fim_rates,
    load_hf_tokenizer,
)

TOKENIZER = Path('/home/zjr1236/data/tokenizer_64k/tokenizer.json')
DATA = Path('/home/zjr1236/data/pretrain_semantic_structured_1b')


def build_loader(batch_size: int = 4, num_workers: int = 4, epoch: int = 0):
    tokenizer = load_hf_tokenizer(TOKENIZER)
    # Preserves the old training defaults: --code-fim-rate 0.50 / --math-fim-rate 0.25.
    # The 50% code FIM budget is split into AST-only, semantic-context, and test-aware modes.
    cfg = config_from_legacy_fim_rates(
        max_length=16384,
        seed=20260830,
        code_fim_rate=0.50,
        math_fim_rate=0.25,
        ast_share=0.20,
        semantic_share=0.64,
        test_share=0.16,
    )
    packs = sorted(DATA.rglob('*.trainpack'))
    if not packs:
        raise RuntimeError(f'no structured trainpacks found under {DATA}')
    dataset = StructuredTrainpackIterableDataset(packs, validate=True)
    collator = StructuredFIMCollator(tokenizer, cfg, epoch=epoch, pad_to_multiple_of=8)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collator, pin_memory=True)


if __name__ == '__main__':
    loader = build_loader(batch_size=2, num_workers=0)
    batch = next(iter(loader))
    print({k: tuple(v.shape) for k, v in batch.items()})
