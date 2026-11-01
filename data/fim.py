#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Mapping as _MappingABC
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import hashlib
import math
import re
import struct
import msgpack

STRUCT_TRAIN_MAGIC = b'SBSTRUCT1\n'
FRAME = struct.Struct('<I')
SCHEMA = 'SBSTRUCT1'

SPECIAL_TOKENS: Tuple[str, ...] = (
    '<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>',
    '<|fim_prefix|>', '<|fim_middle|>', '<|fim_suffix|>',
    '<|repo_start|>', '<|repo_end|>', '<|file_start|>', '<|file_end|>',
    '<|document_start|>', '<|document_end|>',
)
EXPECTED_SPECIAL_IDS = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
RESERVED_CONTROL_TOKENS = frozenset(SPECIAL_TOKENS)
_RESERVED_RE = re.compile('|'.join(re.escape(t) for t in SPECIAL_TOKENS))


def _ids(tokenizer: Any, text: str) -> List[int]:
    return list(tokenizer.encode(text, add_special_tokens=False).ids)


def _stable_u64(*parts: Any) -> int:
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode('utf-8', 'surrogatepass'))
        h.update(b'\0')
    return int.from_bytes(h.digest(), 'little')


def validate_tokenizer_contract(tokenizer: Any, expected: Mapping[str, int] = EXPECTED_SPECIAL_IDS) -> Dict[str, Any]:
    got: Dict[str, int] = {}
    for token, expected_id in expected.items():
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise RuntimeError(f'tokenizer missing required special token {token!r}')
        token_id = int(token_id)
        if token_id != int(expected_id):
            raise RuntimeError(f'special token id mismatch for {token!r}: expected {expected_id}, got {token_id}')
        encoded = list(tokenizer.encode(token, add_special_tokens=False).ids)
        if encoded != [token_id]:
            raise RuntimeError(f'special token is not atomic: {token!r} -> {encoded!r}')
        left = list(tokenizer.encode('A', add_special_tokens=False).ids)
        right = list(tokenizer.encode('B', add_special_tokens=False).ids)
        embedded = list(tokenizer.encode('A'+token+'B', add_special_tokens=False).ids)
        expected_embedded = left + [token_id] + right
        if embedded != expected_embedded:
            raise RuntimeError(f'special token is not boundary-stable when embedded: {token!r} -> {embedded!r}, expected {expected_embedded!r}')
        got[token] = token_id
    return {'special_token_ids': got, 'count': len(got)}


def load_hf_tokenizer(path: Path | str) -> Any:
    try:
        from tokenizers import Tokenizer
    except Exception as exc:
        raise RuntimeError('tokenizers is required: pip install tokenizers') from exc
    tok = Tokenizer.from_file(str(Path(path).expanduser().resolve()))
    validate_tokenizer_contract(tok)
    return tok


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, _MappingABC):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _walk_strings(v)


def _validate_spans(text: str, spans: Sequence[Mapping[str, Any]]) -> None:
    for i, span in enumerate(spans):
        try:
            a = int(span['start']); b = int(span['end'])
        except Exception as exc:
            raise ValueError(f'invalid FIM span #{i}: missing integer start/end') from exc
        if not (0 <= a < b <= len(text)):
            raise ValueError(f'invalid FIM span #{i}: [{a},{b}) outside text length {len(text)}')


def validate_structured_record(record: Mapping[str, Any]) -> None:
    if record.get('schema') != SCHEMA:
        raise ValueError(f'unsupported structured schema: {record.get("schema")!r}')
    if int(record.get('sample_id', -1)) < 0:
        raise ValueError('sample_id must be a non-negative integer')
    source = str(record.get('source') or '')
    structured = record.get('structured')
    if not isinstance(structured, Mapping):
        raise ValueError('structured payload must be a mapping')
    # Raw payloads must never contain control-token literals. The renderer, not source text,
    # owns those tokens; dropping collisions prevents accidental control-token injection.
    for s in _walk_strings(structured):
        m = _RESERVED_RE.search(s)
        if m:
            raise ValueError(f'reserved control token {m.group(0)!r} appears in raw structured payload')
    kind = structured.get('kind')
    if source == 'code':
        if kind != 'code': raise ValueError('code record requires structured.kind="code"')
        target = structured.get('target')
        if not isinstance(target, Mapping): raise ValueError('code record missing target')
        text = str(target.get('text') or '')
        if not text: raise ValueError('code target text is empty')
        _validate_spans(text, list(target.get('fim_candidates') or []))
        for ctx in structured.get('context') or []:
            if not isinstance(ctx, Mapping) or not str(ctx.get('text') or ''):
                raise ValueError('code context entries require non-empty text')
    else:
        if kind != 'document': raise ValueError('non-code record requires structured.kind="document"')
        parts = list(structured.get('parts') or [])
        if not parts: raise ValueError('document record requires at least one part')
        for p in parts:
            if not isinstance(p, Mapping) or not str(p.get('text') or ''):
                raise ValueError('document parts require non-empty text')
        for i in structured.get('fim_candidate_indices') or []:
            if not 0 <= int(i) < len(parts): raise ValueError(f'invalid document FIM part index: {i}')


def iter_structured_trainpack(path: Path | str, *, validate: bool = True) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    with path.open('rb') as f:
        magic = f.read(len(STRUCT_TRAIN_MAGIC))
        if magic != STRUCT_TRAIN_MAGIC:
            raise RuntimeError(f'invalid structured trainpack magic {path}: {magic!r}')
        while True:
            hdr = f.read(FRAME.size)
            if not hdr: return
            if len(hdr) != FRAME.size: raise RuntimeError(f'truncated structured frame header: {path}')
            (n,) = FRAME.unpack(hdr)
            payload = f.read(n)
            if len(payload) != n: raise RuntimeError(f'truncated structured frame payload: {path}')
            row = msgpack.unpackb(payload, raw=False, strict_map_key=False)
            if validate: validate_structured_record(row)
            yield row


def canonical_causal_text(record: Mapping[str, Any], *, max_tree_paths: int = 200, max_contexts: int = 8, validate: bool = True) -> str:
    """Canonical causal serialization used for exact preprocessing token accounting.

    Raw payloads are validated to contain no reserved control-token literals, so this textual
    serialization is unambiguous and tokenizes to the same logical control structure used online.
    """
    if validate:
        validate_structured_record(record)
    source=str(record['source']); s=record['structured']
    if source!='code':
        joined='\n\n'.join(str(x['text']) for x in s['parts'])
        return '<|document_start|>'+joined+'<|document_end|>'
    repo=s.get('repo') or {}; contexts=list(s.get('context') or [])[:max_contexts]; target=s['target']
    if not contexts:
        return '<|file_start|>'+f"\npath={target.get('path','')}\nsymbol={target.get('symbol','')}\n"+str(target['text'])+'<|file_end|>'
    out=['<|repo_start|>',f"\nrepo={repo.get('repo_path','')}\n"]
    tree=list(repo.get('tree') or [])[:max_tree_paths]
    if tree: out.append('tree:\n'+'\n'.join(str(x) for x in tree)+'\n')
    for ctx in contexts:
        role=str(ctx.get('role') or 'related')
        out.extend(['<|file_start|>',f"\nrole={role}\npath={ctx.get('path','')}\nsymbol={ctx.get('symbol','')}\n",str(ctx.get('text') or ''),'<|file_end|>'])
    out.extend(['<|file_start|>',f"\npath={target.get('path','')}\nsymbol={target.get('symbol','')}\n",str(target['text']),'<|file_end|>','<|repo_end|>'])
    return ''.join(out)


@dataclass(frozen=True)
class FimPlan:
    middle_start: int
    middle_end: int
    kind: str = 'fim'


def special_id(tokenizer: Any, token: str) -> int:
    value=tokenizer.token_to_id(token)
    if value is None: raise RuntimeError(f'tokenizer missing special token {token}')
    return int(value)


def encode_fim_text(tokenizer: Any, text: str, plan: FimPlan) -> List[int]:
    """Exact PSM compatibility with the existing training FIM contract."""
    a=int(plan.middle_start);b=int(plan.middle_end)
    if not 0<=a<b<=len(text): raise ValueError('invalid FIM character span')
    return ([special_id(tokenizer,'<|fim_prefix|>')] + _ids(tokenizer,text[:a]) +
            [special_id(tokenizer,'<|fim_suffix|>')] + _ids(tokenizer,text[b:]) +
            [special_id(tokenizer,'<|fim_middle|>')] + _ids(tokenizer,text[a:b]))


@dataclass(frozen=True)
class OnlineFIMConfig:
    max_length: int = 16384
    seed: int = 20260830
    # Keep ordinary FIM disabled by default because the user's existing training path already
    # performs ordinary FIM. Set a positive weight when this transformer is the sole FIM path.
    code_mode_probs: Mapping[str, float] = field(default_factory=lambda: {
        'causal': 0.50,
        'ordinary_fim': 0.0,
        'ast_fim': 0.10,
        'semantic_fim': 0.32,
        'test_aware_fim': 0.08,
    })
    math_fim_probability: float = 0.25
    ordinary_min_tokens: int = 16
    ordinary_min_middle_tokens: int = 4
    ordinary_max_middle_fraction: float = 0.50
    add_bos: bool = False
    add_eos: bool = False
    max_tree_paths: int = 200
    max_contexts: int = 8


def config_from_legacy_fim_rates(*, max_length: int = 16384, seed: int = 20260830, code_fim_rate: float = 0.50, math_fim_rate: float = 0.25,
                                 ast_share: float = 0.20, semantic_share: float = 0.64, test_share: float = 0.16) -> OnlineFIMConfig:
    """Map the existing training CLI rates onto online semantic modes without double-applying FIM.

    The legacy code FIM path was already AST-span based, so ordinary_fim stays at zero by default.
    ast_share + semantic_share + test_share must sum to 1.
    """
    vals=(float(code_fim_rate),float(math_fim_rate),float(ast_share),float(semantic_share),float(test_share))
    if not (0.0 <= vals[0] <= 1.0 and 0.0 <= vals[1] <= 1.0): raise ValueError('FIM rates must be in [0,1]')
    if any(x<0 for x in vals[2:]) or abs(sum(vals[2:])-1.0)>1e-9: raise ValueError('ast/semantic/test shares must be nonnegative and sum to 1')
    return OnlineFIMConfig(max_length=max_length,seed=seed,math_fim_probability=math_fim_rate,code_mode_probs={
        'causal':1.0-code_fim_rate,'ordinary_fim':0.0,'ast_fim':code_fim_rate*ast_share,'semantic_fim':code_fim_rate*semantic_share,'test_aware_fim':code_fim_rate*test_share})


def load_online_fim_config(path: Path | str) -> OnlineFIMConfig:
    import json
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    allowed=set(OnlineFIMConfig.__dataclass_fields__)
    unknown=set(data)-allowed
    if unknown: raise ValueError(f'unknown OnlineFIMConfig keys: {sorted(unknown)}')
    return OnlineFIMConfig(**data)


@dataclass
class TransformResult:
    input_ids: List[int]
    mode: str
    sample_id: int
    num_tokens: int
    context_roles: List[str] = field(default_factory=list)
    target_span: Optional[Tuple[int, int]] = None
    token_span: Optional[Tuple[int, int]] = None
    reconstructed_target: Optional[str] = None


class OnlineFIMTransformer:
    def __init__(self, tokenizer: Any, config: Optional[OnlineFIMConfig] = None, *, validate_tokenizer: bool = True):
        self.tokenizer = tokenizer
        self.config = config or OnlineFIMConfig()
        if self.config.max_length <= 0: raise ValueError('max_length must be positive')
        if validate_tokenizer: validate_tokenizer_contract(tokenizer)
        self.sid = {s: int(tokenizer.token_to_id(s)) for s in SPECIAL_TOKENS}

    def _rng_fraction(self, record: Mapping[str, Any], epoch: int, key: str) -> float:
        return _stable_u64(self.config.seed, epoch, int(record['sample_id']), key) / 2**64

    def _eligible_code_modes(self, record: Mapping[str, Any]) -> List[str]:
        caps = set(record.get('capabilities') or [])
        target = record['structured']['target']
        spans = list(target.get('fim_candidates') or [])
        contexts = list(record['structured'].get('context') or [])
        tests = any(str(x.get('role')) in {'test', 'tested_by'} for x in contexts)
        out = ['causal']
        target_tokens = len(_ids(self.tokenizer, str(target['text'])))
        if 'ordinary_fim' in caps and target_tokens >= self.config.ordinary_min_tokens: out.append('ordinary_fim')
        if spans and 'ast_fim' in caps: out.append('ast_fim')
        if spans and contexts and 'semantic_fim' in caps: out.append('semantic_fim')
        if spans and tests and 'test_aware_fim' in caps: out.append('test_aware_fim')
        return out

    def _choose_code_mode(self, record: Mapping[str, Any], epoch: int) -> str:
        eligible = self._eligible_code_modes(record)
        weights = {m: max(0.0, float(self.config.code_mode_probs.get(m, 0.0))) for m in eligible}
        total = sum(weights.values())
        if total <= 0:
            # Fail closed to a valid deterministic mode rather than inventing an unavailable semantic task.
            return 'ordinary_fim' if 'ordinary_fim' in eligible else 'causal'
        x = self._rng_fraction(record, epoch, 'mode') * total
        acc = 0.0
        for mode in ('causal','ordinary_fim','ast_fim','semantic_fim','test_aware_fim'):
            if mode not in weights: continue
            acc += weights[mode]
            if x < acc: return mode
        return next(reversed(weights))

    def _code_contexts(self, record: Mapping[str, Any], mode: str) -> List[Mapping[str, Any]]:
        if mode in {'ordinary_fim', 'ast_fim'}: return []
        contexts = list(record['structured'].get('context') or [])
        if mode == 'test_aware_fim':
            contexts.sort(key=lambda x: (0 if str(x.get('role')) in {'test','tested_by'} else 1, -float(x.get('score',0.0)), str(x.get('path',''))))
        else:
            contexts.sort(key=lambda x: (-float(x.get('score',0.0)), str(x.get('role','')), str(x.get('path',''))))
        return contexts[: self.config.max_contexts]

    def _append_text(self, out: List[int], text: str) -> None:
        out.extend(_ids(self.tokenizer, text))

    def _code_prefix(self, record: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]]) -> Tuple[List[int], List[str]]:
        s = record['structured']; repo = s.get('repo') or {}; out: List[int] = []; roles: List[str] = []
        if contexts:
            out.append(self.sid['<|repo_start|>'])
            self._append_text(out, f"\nrepo={repo.get('repo_path','')}\n")
            tree = list(repo.get('tree') or [])[: self.config.max_tree_paths]
            if tree: self._append_text(out, 'tree:\n' + '\n'.join(str(x) for x in tree) + '\n')
            for ctx in contexts:
                out.append(self.sid['<|file_start|>'])
                role = str(ctx.get('role') or 'related'); roles.append(role)
                self._append_text(out, f"\nrole={role}\npath={ctx.get('path','')}\nsymbol={ctx.get('symbol','')}\n")
                self._append_text(out, str(ctx.get('text') or ''))
                out.append(self.sid['<|file_end|>'])
        return out, roles

    def _target_header_ids(self, target: Mapping[str, Any]) -> List[int]:
        out = [self.sid['<|file_start|>']]
        self._append_text(out, f"\npath={target.get('path','')}\nsymbol={target.get('symbol','')}\n")
        return out

    def _code_causal(self, record: Mapping[str, Any], contexts: Sequence[Mapping[str, Any]]) -> TransformResult:
        s = record['structured']; target = s['target']; out, roles = self._code_prefix(record, contexts)
        out.extend(self._target_header_ids(target)); self._append_text(out, str(target['text'])); out.append(self.sid['<|file_end|>'])
        if contexts: out.append(self.sid['<|repo_end|>'])
        out = self._finish(out)
        return TransformResult(out,'causal',int(record['sample_id']),len(out),roles,reconstructed_target=str(target['text']))

    def _choose_ast_span(self, record: Mapping[str, Any], epoch: int) -> Mapping[str, Any]:
        spans = list(record['structured']['target'].get('fim_candidates') or [])
        if not spans: raise ValueError('AST/semantic FIM requested without fim_candidates')
        # Prefer body/statement-like spans, then spans near a useful medium size.
        preferred = {'body','statement','return_statement','assignment','expression_statement','block','compound_statement'}
        spans.sort(key=lambda x: (0 if str(x.get('kind')) in preferred else 1, abs((int(x['end'])-int(x['start']))-512), int(x['start']), int(x['end'])))
        top = spans[: min(8, len(spans))]
        return top[_stable_u64(self.config.seed, epoch, int(record['sample_id']), 'ast-span') % len(top)]

    def _fim_from_char_span(self, record: Mapping[str, Any], mode: str, contexts: Sequence[Mapping[str, Any]], span: Mapping[str, Any]) -> TransformResult:
        target = record['structured']['target']; text = str(target['text']); a = int(span['start']); b = int(span['end'])
        if not (0 <= a < b <= len(text)): raise ValueError(f'invalid FIM span [{a},{b}) for target length {len(text)}')
        prefix, middle, suffix = text[:a], text[a:b], text[b:]
        out, roles = self._code_prefix(record, contexts)
        header = self._target_header_ids(target)
        tail = [self.sid['<|file_end|>']] + ([self.sid['<|repo_end|>']] if contexts else [])
        # Exact existing PSM contract: FIM_PREFIX + prefix + FIM_SUFFIX + suffix + FIM_MIDDLE + middle.
        fim: List[int] = [self.sid['<|fim_prefix|>']]
        fim.extend(out); fim.extend(header); fim.extend(_ids(self.tokenizer, prefix))
        fim.append(self.sid['<|fim_suffix|>']); fim.extend(_ids(self.tokenizer, suffix)); fim.extend(tail)
        fim.append(self.sid['<|fim_middle|>']); fim.extend(_ids(self.tokenizer, middle))
        fim = self._finish(fim)
        return TransformResult(fim,mode,int(record['sample_id']),len(fim),roles,(a,b),None,prefix+middle+suffix)

    def _ordinary_fim(self, record: Mapping[str, Any], epoch: int) -> TransformResult:
        target = record['structured']['target']; text = str(target['text']); ids = _ids(self.tokenizer,text); n=len(ids)
        if n < self.config.ordinary_min_tokens: return self._code_causal(record,[])
        max_mid=max(self.config.ordinary_min_middle_tokens,int(n*self.config.ordinary_max_middle_fraction))
        max_mid=min(max_mid,max(self.config.ordinary_min_middle_tokens,n-2))
        min_mid=min(self.config.ordinary_min_middle_tokens,max_mid)
        span_len=min_mid + (_stable_u64(self.config.seed,epoch,int(record['sample_id']),'ordinary-len') % max(1,max_mid-min_mid+1))
        max_start=max(0,n-span_len)
        start=_stable_u64(self.config.seed,epoch,int(record['sample_id']),'ordinary-start') % (max_start+1)
        end=start+span_len
        header=self._target_header_ids(target);tail=[self.sid['<|file_end|>']]
        out=[self.sid['<|fim_prefix|>']]+header+ids[:start]+[self.sid['<|fim_suffix|>']]+ids[end:]+tail+[self.sid['<|fim_middle|>']]+ids[start:end]
        out=self._finish(out)
        return TransformResult(out,'ordinary_fim',int(record['sample_id']),len(out),[],None,(start,end),text)

    def _document(self, record: Mapping[str, Any], epoch: int, mode_override: Optional[str]) -> TransformResult:
        s=record['structured']; parts=list(s['parts']); texts=[str(x['text']) for x in parts]; source=str(record['source'])
        can_fim=source.startswith('math_') and bool(s.get('fim_candidate_indices'))
        mode=mode_override or ('math_fim' if can_fim and self._rng_fraction(record,epoch,'math-mode') < self.config.math_fim_probability else 'causal')
        if mode=='math_fim' and can_fim:
            choices=[int(x) for x in s['fim_candidate_indices']]
            idx=choices[_stable_u64(self.config.seed,epoch,int(record['sample_id']),'math-span')%len(choices)]
            prefix='\n\n'.join(texts[:idx]);middle=texts[idx];suffix='\n\n'.join(texts[idx+1:])
            out=[self.sid['<|fim_prefix|>'],self.sid['<|document_start|>']]+_ids(self.tokenizer,prefix)+[self.sid['<|fim_suffix|>']]+_ids(self.tokenizer,suffix)+[self.sid['<|document_end|>'],self.sid['<|fim_middle|>']]+_ids(self.tokenizer,middle)
            out=self._finish(out)
            return TransformResult(out,'math_fim',int(record['sample_id']),len(out),[],None,None,'\n\n'.join(texts))
        joined='\n\n'.join(texts);out=[self.sid['<|document_start|>']]+_ids(self.tokenizer,joined)+[self.sid['<|document_end|>']];out=self._finish(out)
        return TransformResult(out,'causal',int(record['sample_id']),len(out),[],None,None,joined)

    def _finish(self, ids: List[int]) -> List[int]:
        if self.config.add_bos: ids=[self.sid['<|bos|>']]+ids
        if self.config.add_eos: ids=ids+[self.sid['<|eos|>']]
        if len(ids)>self.config.max_length:
            raise ValueError(f'online transform exceeds max_length: {len(ids)} > {self.config.max_length}')
        return ids

    def transform(self, record: Mapping[str, Any], *, epoch: int = 0, mode_override: Optional[str] = None) -> TransformResult:
        validate_structured_record(record)
        if str(record['source']) != 'code': return self._document(record,epoch,mode_override)
        mode=mode_override or self._choose_code_mode(record,epoch)
        contexts=self._code_contexts(record,mode)
        # If the full semantic context is too long, drop lowest-priority contexts deterministically.
        if mode=='causal':
            while True:
                try:return self._code_causal(record,contexts)
                except ValueError as e:
                    if 'max_length' not in str(e) or not contexts: raise
                    contexts=contexts[:-1]
        if mode=='ordinary_fim': return self._ordinary_fim(record,epoch)
        span=self._choose_ast_span(record,epoch)
        while True:
            try:return self._fim_from_char_span(record,mode,contexts,span)
            except ValueError as e:
                if 'max_length' not in str(e) or not contexts: raise
                contexts=contexts[:-1]


class StructuredFIMCollator:
    def __init__(self, tokenizer: Any, config: Optional[OnlineFIMConfig] = None, *, epoch: int = 0, pad_to_multiple_of: int = 1, return_metadata: bool = False):
        self.transformer=OnlineFIMTransformer(tokenizer,config);self.epoch=int(epoch);self.pad_to_multiple_of=max(1,int(pad_to_multiple_of));self.return_metadata=return_metadata
        self.pad_id=int(tokenizer.token_to_id('<|pad|>'))
    def set_epoch(self, epoch: int) -> None: self.epoch=int(epoch)
    def __call__(self, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        try: import torch
        except Exception as exc: raise RuntimeError('PyTorch is required for StructuredFIMCollator') from exc
        results=[self.transformer.transform(r,epoch=self.epoch) for r in records]
        max_len=max(len(r.input_ids) for r in results)
        max_len=int(math.ceil(max_len/self.pad_to_multiple_of)*self.pad_to_multiple_of)
        input_ids=[];attention=[];labels=[]
        for r in results:
            pad=max_len-len(r.input_ids);row=r.input_ids+[self.pad_id]*pad;mask=[1]*len(r.input_ids)+[0]*pad;lab=r.input_ids+[-100]*pad
            input_ids.append(row);attention.append(mask);labels.append(lab)
        out={'input_ids':torch.tensor(input_ids,dtype=torch.long),'attention_mask':torch.tensor(attention,dtype=torch.long),'labels':torch.tensor(labels,dtype=torch.long)}
        if self.return_metadata:
            out['sample_id']=[r.sample_id for r in results];out['fim_mode']=[r.mode for r in results];out['num_tokens']=[r.num_tokens for r in results]
        return out


try:
    from torch.utils.data import IterableDataset as _TorchIterableDataset
except Exception:
    class _TorchIterableDataset:  # type: ignore[no-redef]
        pass


class StructuredTrainpackIterableDataset(_TorchIterableDataset):
    """PyTorch IterableDataset adapter. Shards files among DataLoader workers without shared state."""
    def __init__(self, paths: Sequence[Path | str], *, validate: bool = True):
        super().__init__()
        self.paths=[Path(p) for p in paths];self.validate=validate
    def __iter__(self):
        try:
            from torch.utils.data import get_worker_info
            info=get_worker_info()
        except Exception:
            info=None
        paths=self.paths if info is None else self.paths[info.id::info.num_workers]
        for p in paths:
            yield from iter_structured_trainpack(p,validate=self.validate)
