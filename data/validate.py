#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, sys, time
from collections import Counter
from pathlib import Path
from fim import (
    OnlineFIMConfig, OnlineFIMTransformer, canonical_causal_text,
    iter_structured_trainpack, load_hf_tokenizer, validate_tokenizer_contract,
)


def main(argv=None):
    p=argparse.ArgumentParser(description='Strict validator for SBSTRUCT1 trainpacks and online FIM transforms')
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--tokenizer',type=Path,required=True)
    p.add_argument('--max-length',type=int,default=16384)
    p.add_argument('--transform-samples',type=int,default=5000,help='number of records to exercise across all eligible modes; 0 disables')
    p.add_argument('--full-token-check',action='store_true',help='re-tokenize every record and verify stored base token count')
    p.add_argument('--seed',type=int,default=20260830)
    p.add_argument('--report',type=Path,default=None)
    a=p.parse_args(argv)
    root=a.data_dir.expanduser().resolve();paths=sorted(root.rglob('*.trainpack'))
    if not paths:raise SystemExit(f'no .trainpack files under {root}')
    tok=load_hf_tokenizer(a.tokenizer);contract=validate_tokenizer_contract(tok);tr=OnlineFIMTransformer(tok,OnlineFIMConfig(max_length=a.max_length,seed=a.seed))
    counts=Counter();mode_counts=Counter();cap_counts=Counter();samples=[];t0=time.perf_counter()
    for path in paths:
        for row in iter_structured_trainpack(path,validate=True):
            counts['records']+=1;counts['stored_base_tokens']+=int(row.get('tokens') or 0);counts[f"source::{row.get('source')}"]+=1
            for cap in row.get('capabilities') or []:cap_counts[str(cap)]+=1
            if a.full_token_check:
                n=len(tok.encode(canonical_causal_text(row),add_special_tokens=False).ids)
                if n!=int(row.get('tokens') or -1):raise RuntimeError(f'base token count mismatch sample={row.get("sample_id")}: stored={row.get("tokens")} actual={n}')
            if a.transform_samples>0:
                # Deterministic reservoir sampling so large datasets do not need to stay in RAM.
                k=a.transform_samples
                if len(samples)<k:samples.append(row)
                else:
                    j=random.Random(a.seed+counts['records']).randrange(counts['records'])
                    if j<k:samples[j]=row
    for row in samples:
        source=str(row['source']);caps=set(row.get('capabilities') or [])
        modes=['causal']
        if source=='code':modes += [x for x in ('ordinary_fim','ast_fim','semantic_fim','test_aware_fim') if x in caps]
        elif 'math_fim' in caps:modes.append('math_fim')
        for mode in modes:
            r=tr.transform(row,epoch=0,mode_override=mode)
            if not r.input_ids or len(r.input_ids)>a.max_length:raise RuntimeError(f'invalid online transform sample={row["sample_id"]} mode={mode} len={len(r.input_ids)}')
            mode_counts[mode]+=1;counts['transformed_tokens']+=len(r.input_ids)
    report={'complete':True,'schema':'SBSTRUCT1','files':len(paths),'records':counts['records'],'stored_base_tokens':counts['stored_base_tokens'],
            'tokenizer_contract':contract,'capability_sample_counts':dict(cap_counts),'validated_mode_transforms':dict(mode_counts),
            'full_token_check':bool(a.full_token_check),'transform_samples':len(samples),'elapsed_s':time.perf_counter()-t0}
    rp=a.report or (root/'structured_validation_report.json');rp.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True));print(f'VALIDATION PASS -> {rp}')
    return 0

if __name__=='__main__':raise SystemExit(main())
