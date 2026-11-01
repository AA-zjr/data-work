#!/usr/bin/env python3
from __future__ import annotations

# Standalone SemanticBlock preprocessing pipeline.
# Revision: 2026-09-16-250b-v11-bounded-repo-relations
# Generated from the tested modular implementation, with no package-relative imports.

# ============================================================================
# BEGIN config.py
# ============================================================================
from collections import OrderedDict

SOURCE_SHARES=OrderedDict([
 ('general',0.45),('math_nemotron',0.125),('math_finemath',0.125),('code',0.20),('academic',0.07),('wiki',0.03)
])
LANGUAGE_RATIOS=OrderedDict([
 ('Python',0.17),('C',0.05),('C++',0.09),('C#',0.05),('Java',0.09),('JavaScript',0.08),('TypeScript',0.09),('TSX',0.02),
 ('Go',0.08),('Rust',0.05),('Ruby',0.02),('PHP',0.03),('Kotlin',0.02),('Swift',0.02),('Scala',0.015),('Lua',0.015),('Shell',0.03),
 ('Dart',0.01),('Julia',0.008),('R',0.015),('Haskell',0.008),('Elixir',0.007),('Erlang',0.005),('OCaml',0.005),('Clojure',0.005),
 ('Objective-C',0.007),('Zig',0.005),('Solidity',0.005)
])
CODE_SAMPLE_RATIOS=OrderedDict([('repo_causal',0.35),('semantic_fim',0.35),('single_causal',0.12),('ast_fim',0.10),('test_aware',0.08)])
DEFAULT_CONFIG={
 'source_shares':dict(SOURCE_SHARES),'language_ratios':dict(LANGUAGE_RATIOS),'code_sample_ratios':dict(CODE_SAMPLE_RATIOS),
 'math_fim_rate':0.25,
 'inventory':{'records_per_shard':8192},
 'dedup':{'near':True,'exact_partitions':1024,'lsh_partitions':1024,'signature_size':32,'bands':8,'max_bucket':512,'thresholds':{'natural':0.88,'math':0.97}},
 'semantic':{'repo_partitions':256,'repo_buckets':32,'chunk_tokens':8192,'max_related':8,'min_quality':0.25,'max_parse_mib':2.0,'max_code_file_mib':8.0,'ts_timeout_s':15.0,'ts_crash_threshold':2,'ts_parser_cache':8,'ts_max_nodes':250000,
             'giant_repo_files':20000,'giant_repo_row_cap':12000,'giant_repo_target_cap':4096,'giant_repo_neighbor_cap':512,'tree_summary_lines':500,'target_tree_paths':24},
 'planner':{'overplan':1.06,'max_repo_fraction':0.01,'plan_shards':128},
 'render':{'max_sample_tokens':16384,'workers':16,'shard_tokens':25000000,'output_format':'structured_trainpack','tree_limit':200}
}

def deep_merge(base,override):
    out={k:(dict(v) if isinstance(v,dict) else v) for k,v in base.items()}
    for k,v in (override or {}).items():
        if isinstance(v,dict) and isinstance(out.get(k),dict):out[k]=deep_merge(out[k],v)
        else:out[k]=v
    return out


# ============================================================================
# BEGIN storage.py
# ============================================================================
import json, os, struct, tempfile, multiprocessing as mp, sys, threading, io
from pathlib import Path
from typing import Any, Iterator, Optional
from concurrent.futures import ThreadPoolExecutor
import msgpack

FRAME = struct.Struct('<I')
MAGIC = b'SBIDX1\n'

class FramedWriter:
    def __init__(self, path: Path | str, *, magic: bytes = MAGIC, atomic: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.atomic = atomic
        self.tmp = self.path.with_name(self.path.name + f'.tmp.{os.getpid()}') if atomic else self.path
        self.f = self.tmp.open('wb', buffering=1<<20)
        self.f.write(magic)
        self.magic = magic
        self.count = 0
        self.bytes = len(magic)
        self.closed = False
    def write(self, obj: Any) -> int:
        payload = msgpack.packb(obj, use_bin_type=True)
        offset = self.f.tell()
        self.f.write(FRAME.pack(len(payload))); self.f.write(payload)
        self.count += 1; self.bytes += FRAME.size + len(payload)
        return offset
    def flush(self):
        self.f.flush()
    def close(self):
        if self.closed: return
        self.f.flush(); self.f.close(); self.closed=True
        if self.atomic: os.replace(self.tmp, self.path)
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        if exc_type is None: self.close()
        else:
            try: self.f.close()
            finally:
                if self.atomic:
                    try: self.tmp.unlink()
                    except FileNotFoundError: pass


def _iter_frames_stream(f, magic: bytes, label: Any = '<bytes>') -> Iterator[Any]:
    got=f.read(len(magic))
    if got != magic: raise RuntimeError(f'invalid framed magic {label}: {got!r}')
    while True:
        h=f.read(FRAME.size)
        if not h: return
        if len(h)!=FRAME.size: raise RuntimeError(f'truncated frame header: {label}@{f.tell()-len(h)}')
        (n,)=FRAME.unpack(h); b=f.read(n)
        if len(b)!=n: raise RuntimeError(f'truncated frame payload: {label}')
        yield msgpack.unpackb(b, raw=False, strict_map_key=False)


def iter_frames(path: Path | str, *, magic: bytes = MAGIC) -> Iterator[Any]:
    path=Path(path)
    # 8 KiB default buffering turns sequential reads into per-8KiB FUSE calls,
    # which is 10-20x slower on the NTFS candidate tree than 1 MiB refills.
    with path.open('rb', buffering=1<<20) as f:
        yield from _iter_frames_stream(f, magic, path)


def iter_frames_bytes(data: bytes, *, magic: bytes = MAGIC, label: Any = '<bytes>') -> Iterator[Any]:
    yield from _iter_frames_stream(io.BytesIO(data), magic, label)


_PACK_HANDLES=OrderedDict()
_PACK_HANDLE_MAX=max(4,int(os.environ.get('SBP_PACK_HANDLES','64')))


def _pack_handle(path:Path|str):
    # One open per pack instead of one per document. On the fuseblk NTFS mount an
    # open costs ~0.4ms and it dominated render's CPU profile (~10 opens/sample).
    key=str(path);f=_PACK_HANDLES.get(key)
    if f is not None:_PACK_HANDLES.move_to_end(key);return f
    f=open(key,'rb',buffering=1<<20)
    _PACK_HANDLES[key]=f
    while len(_PACK_HANDLES)>_PACK_HANDLE_MAX:
        _,_old=_PACK_HANDLES.popitem(last=False);_old.close()
    return f


def read_frame_at(path: Path | str, offset: int, *, magic: bytes = MAGIC) -> Any:
    f=_pack_handle(path)
    f.seek(offset); h=f.read(FRAME.size)
    if len(h)!=FRAME.size: raise RuntimeError(f'bad frame offset {path}@{offset}')
    (n,)=FRAME.unpack(h); b=f.read(n)
    if len(b)!=n: raise RuntimeError(f'bad frame payload {path}@{offset}')
    return msgpack.unpackb(b, raw=False, strict_map_key=False)


def atomic_json(path: Path | str, obj: Any) -> None:
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,sort_keys=True),encoding='utf-8')
    os.replace(tmp,path)


def _rmtree_any(path: Path | str, keep_link: bool=False) -> None:
    """Remove a tree; for a symlinked tree empty the target and keep the link.

    Dedup intermediates can be symlinked onto a different filesystem, where
    shutil.rmtree refuses to operate on the link itself.
    """
    path=Path(path)
    if path.is_symlink():
        target=Path(os.path.realpath(path))
        if target.is_dir():
            for ch in target.iterdir():_rmtree_any(ch)
        elif target.exists():target.unlink(missing_ok=True)
        if keep_link:target.mkdir(parents=True,exist_ok=True)
        else:
            path.unlink(missing_ok=True)
            if target.is_dir():shutil.rmtree(target,ignore_errors=True)
        return
    if path.is_dir():shutil.rmtree(path,ignore_errors=True)
    elif path.exists():path.unlink(missing_ok=True)


def _replace_file(src: Path | str, dst: Path | str) -> None:
    """Atomic replace when both paths share a filesystem, else staged copy."""
    src=Path(src);dst=Path(dst)
    try:os.replace(src,dst)
    except OSError:
        stage=dst.with_name(dst.name+'.partial')
        shutil.copyfile(src,stage);os.replace(stage,dst);src.unlink(missing_ok=True)


def success_marker(stage_dir: Path | str, name: str, payload: Optional[dict]=None) -> Path:
    p=Path(stage_dir)/'_SUCCESS.json'; atomic_json(p,{'stage':name,**(payload or {})}); return p


def process_context():
    forced=os.environ.get('SEMANTIC_MP_START_METHOD')
    if forced:return mp.get_context(forced)
    # 250B-scale preprocessing keeps large Python/native heaps alive in the parent.
    # fork() makes later allocator writes COW those pages into every child, which can
    # turn a modest worker pool into whole-machine memory pressure. Spawn is slower
    # to start but gives each long-lived worker a clean address space and is the safe
    # production default.
    return mp.get_context('spawn')


# ============================================================================
# BEGIN packio.py
# ============================================================================
import json, os, struct
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

PACK_MAGIC=b'PTPACK1\n'; PACK_RECORD=struct.Struct('<BI')
PTCODE4_MAGIC=b'PTCODE4\n'; PTCODE5_MAGIC=b'PTCODE5\n'; CODE_FRAME=struct.Struct('<BBBII')
CODE_KIND_REPO=1; CODE_KIND_FILE=2; CODE_FLAG_COUNT=1<<0; CODE_FLAG_META=1<<1

LANGUAGE_ALIASES={
    'python':'Python','c':'C','c++':'C++','cpp':'C++','c plus plus':'C++','c#':'C#','c sharp':'C#','csharp':'C#',
    'java':'Java','javascript':'JavaScript','javascript jsx':'JavaScript','typescript':'TypeScript','typescript jsx':'TSX','tsx':'TSX',
    'go':'Go','golang':'Go','rust':'Rust','ruby':'Ruby','php':'PHP','kotlin':'Kotlin','swift':'Swift','scala':'Scala','lua':'Lua',
    'shell':'Shell','shell script':'Shell','bash':'Shell','zsh':'Shell','dart':'Dart','julia':'Julia','r':'R','haskell':'Haskell',
    'elixir':'Elixir','erlang':'Erlang','ocaml':'OCaml','clojure':'Clojure','objective-c':'Objective-C','objective c':'Objective-C',
    'objectivec':'Objective-C','zig':'Zig','solidity':'Solidity',
}
CODE_LANGUAGES=(
    'Python','C','C++','C#','Java','JavaScript','TypeScript','TSX','Go','Rust','Ruby','PHP','Kotlin','Swift','Scala','Lua',
    'Shell','Dart','Julia','R','Haskell','Elixir','Erlang','OCaml','Clojure','Objective-C','Zig','Solidity'
)

def canonical_language(value):
    if value is None: return None
    s=str(value).strip(); return LANGUAGE_ALIASES.get(s.lower(), s if s in CODE_LANGUAGES else None)


def stable_u64(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'),digest_size=8).digest(),'little',signed=False)


def scan_ptpack(path: Path | str, source: str) -> Iterator[Dict[str,Any]]:
    """Scan one canonical PTPACK1 shard.

    Final V3/V4/V5 generic shards are record-aligned and each starts with
    PTPACK1\n.  Do not guess legacy/headerless layouts here: stage_manifest.json
    is authoritative and stale/unlisted files are excluded by inventory.
    """
    path=Path(path)
    with path.open('rb') as f:
        prefix=f.read(len(PACK_MAGIC))
        if prefix != PACK_MAGIC:
            raise RuntimeError(
                f'invalid manifest-listed PTPACK1 shard: {path} '
                f'(magic={prefix!r}, expected={PACK_MAGIC!r})'
            )
        size=os.fstat(f.fileno()).st_size
        while True:
            off=f.tell(); h=f.read(PACK_RECORD.size)
            if not h: return
            if len(h)!=PACK_RECORD.size:
                raise RuntimeError(f'truncated PTPACK1: {path}@{off}')
            subtype,n=PACK_RECORD.unpack(h)
            payload_end=f.tell()+int(n)
            if payload_end>size:
                raise RuntimeError(
                    f'truncated PTPACK1 payload: {path}@{off} '
                    f'need={int(n)} remaining={max(0,size-f.tell())}'
                )
            f.seek(int(n),1)
            rid=stable_u64(f'{source}|{path.resolve()}|{off}')
            yield {'kind':'document','record_id':rid,'source':source,'format':'PTPACK1','pack_path':str(path.resolve()),'offset':off,
                   'payload_len':int(n),'subtype':int(subtype),'approx_tokens':max(1,int(n)//4)}


def _code_magic(f,path):
    magic=f.read(8)
    if magic != PTCODE5_MAGIC:
        raise RuntimeError(f'strict PTCODE5 required: {path}: {magic!r}')
    return magic


def scan_code_pack(path: Path | str) -> Iterator[Dict[str,Any]]:
    path=Path(path)
    with path.open('rb') as f:
        magic=_code_magic(f,path); fmt=magic.decode().strip()
        while True:
            off=f.tell(); h=f.read(CODE_FRAME.size)
            if not h: return
            if len(h)!=CODE_FRAME.size: raise RuntimeError(f'truncated code header: {path}@{off}')
            kind,subtype,flags,mn,tn=CODE_FRAME.unpack(h); mb=f.read(mn)
            if len(mb)!=mn: raise RuntimeError(f'truncated code meta: {path}@{off}')
            meta=json.loads(mb.decode('utf-8')) if mb else {}
            f.seek(tn,1)
            repo_key=str(meta.get('repo_key') or '')
            if kind==CODE_KIND_REPO:
                yield {'kind':'repo','repo_key':repo_key,'repo_id':str(meta.get('repo_id') or ''),'repo_path':str(meta.get('repo_path') or ''),
                       'commit_id':str(meta.get('commit_id') or ''),'github_metadata':meta.get('github_metadata') or {},
                       'repo_metadata':meta.get('repo_metadata') or {},'tree':meta.get('tree') or [],'format':fmt,'pack_path':str(path.resolve()),'offset':off}
            elif kind==CODE_KIND_FILE:
                fm=meta.get('file_metadata') or {}; fp=str(meta.get('file_path') or fm.get('file_path') or '')
                lang=canonical_language(meta.get('language') or fm.get('language'))
                rid=stable_u64(f'code|{path.resolve()}|{off}')
                yield {'kind':'code_file','record_id':rid,'source':'code','repo_key':repo_key,'file_path':fp,'language':lang,'format':fmt,
                       'pack_path':str(path.resolve()),'offset':off,'payload_len':int(tn),'subtype':int(subtype),
                       'countable':bool(flags & CODE_FLAG_COUNT),'is_meta':bool(flags & CODE_FLAG_META),'file_metadata':fm,
                       'approx_tokens':max(1,int(tn)//4)}


def read_ptpack_text(path: Path | str, offset: int) -> str:
    f=_pack_handle(path)
    f.seek(offset); h=f.read(PACK_RECORD.size)
    if len(h)!=PACK_RECORD.size: raise RuntimeError(f'bad ptpack locator {path}@{offset}')
    _,n=PACK_RECORD.unpack(h); b=f.read(n)
    if len(b)!=n: raise RuntimeError(f'bad ptpack payload {path}@{offset}')
    return b.decode('utf-8','strict')


def read_code_text(path: Path | str, offset: int) -> str:
    f=_pack_handle(path)
    f.seek(offset); h=f.read(CODE_FRAME.size)
    if len(h)!=CODE_FRAME.size: raise RuntimeError(f'bad code locator {path}@{offset}')
    _,_,_,mn,tn=CODE_FRAME.unpack(h); f.seek(mn,1); b=f.read(tn)
    if len(b)!=tn: raise RuntimeError(f'bad code payload {path}@{offset}')
    return b.decode('utf-8','strict')


def read_text(rec: Mapping[str,Any]) -> str:
    fmt=rec.get('format')
    if fmt=='PTPACK1': return read_ptpack_text(rec['pack_path'],int(rec['offset']))
    if fmt=='PTCODE5': return read_code_text(rec['pack_path'],int(rec['offset']))
    raise ValueError(f'unknown format {fmt!r}')


# ============================================================================
# BEGIN inventory.py
# ============================================================================
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json, shutil, time
from typing import Any, Dict

GENERIC_SOURCES=('general','math_nemotron','math_finemath','academic','wiki')

def _load_strict_stage_manifest(stage: Path) -> Dict[str,Any]:
    p=stage/'stage_manifest.json'
    if not p.is_file():
        raise RuntimeError(f'strict stage requires stage_manifest.json: {p}')
    try:
        m=json.loads(p.read_text(encoding='utf-8'))
    except Exception as e:
        raise RuntimeError(f'cannot read strict stage manifest {p}: {e}') from e
    if m.get('strict_pack_contract') != 'PTSTAGE1':
        raise RuntimeError(
            f"strict stage contract PTSTAGE1 required; got {m.get('strict_pack_contract')!r}. "
            'Regenerate the stage with pretrain_read_count_1p25b_strict.py.'
        )
    if int(m.get('format_version') or 0) != 5 or m.get('code_pack_format') != 'PTCODE5':
        raise RuntimeError(f"strict stage format mismatch: format_version={m.get('format_version')!r} code_pack_format={m.get('code_pack_format')!r}")
    return m


def _source_pack_entries(stage: Path, source: str):
    """Resolve strict manifest-listed packs and retain metadata needed for safe inventory resume."""
    manifest=_load_strict_stage_manifest(stage)
    metas=((manifest.get('sources') or {}).get(source) or {}).get('shards') or []
    if not metas:return []
    out=[];seen=set()
    for meta in metas:
        if not isinstance(meta,dict) or not meta.get('path'):
            raise RuntimeError(f'strict manifest shard for {source} missing path: {meta!r}')
        raw=str(meta['path']); p=Path(raw).expanduser()
        candidates=[p] if p.is_absolute() else [stage/p]
        candidates.append(stage/'records'/source/p.name)
        resolved=next((x.resolve() for x in candidates if x.exists()),None)
        if resolved is None:raise FileNotFoundError(f'strict manifest shard missing for {source}: {raw}')
        st=resolved.stat()
        if meta.get('file_bytes') is not None and int(meta['file_bytes']) != int(st.st_size):
            raise RuntimeError(f'strict shard size mismatch: {resolved}: manifest={meta["file_bytes"]} actual={st.st_size}')
        with resolved.open('rb') as f:magic=f.read(8)
        expected=PTCODE5_MAGIC if source=='code' else PACK_MAGIC
        if magic != expected:raise RuntimeError(f'strict shard magic mismatch: {resolved}: got={magic!r} expected={expected!r}')
        key=str(resolved)
        if key in seen:continue
        seen.add(key);out.append((resolved,dict(meta)))
    return out


def _source_packs(stage: Path, source: str):
    return [p for p,_ in _source_pack_entries(stage,source)]

def _inventory_expected_records(source:str,meta:Mapping[str,Any])->int:
    key='frames' if source=='code' else 'records'
    n=meta.get(key)
    if n is None:
        # Older manifests normally have these fields. Refuse unsafe legacy adoption if not.
        raise RuntimeError(f'inventory resume requires manifest {key} for {source}: {meta.get("path")!r}')
    return max(0,int(n))


def _inventory_expected_bytes(source:str,meta:Mapping[str,Any])->int:
    if source=='code':return max(0,int(meta.get('text_bytes') or 0)+int(meta.get('aux_text_bytes') or 0))
    return max(0,int(meta.get('text_bytes') or 0))


def _inventory_shard_path(base:Path,index:int)->Path:
    return base.with_name(base.name+f'-{index:05d}.mpk')


def _inventory_done_path(base:Path)->Path:
    return base.with_name(base.name+'.done.json')


def _count_framed_records_fast(path:Path)->int:
    """Validate one inventory shard without msgpack decoding; atomic outputs make this sufficient."""
    nrec=0
    with Path(path).open('rb') as f:
        got=f.read(len(MAGIC))
        if got!=MAGIC:raise RuntimeError(f'invalid inventory shard magic {path}: {got!r}')
        size=os.fstat(f.fileno()).st_size
        while True:
            h=f.read(FRAME.size)
            if not h:return nrec
            if len(h)!=FRAME.size:raise RuntimeError(f'truncated inventory frame header: {path}')
            (n,)=FRAME.unpack(h); nxt=f.tell()+int(n)
            if nxt>size:raise RuntimeError(f'truncated inventory frame payload: {path}')
            f.seek(int(n),1);nrec+=1


def _inventory_pack_complete(base:Path|str,expected_records:int,records_per_shard:int,source_path:Path|str|None=None)->Tuple[bool,int]:
    """Return whether a pack has a complete atomic inventory image.

    V6+ uses a done marker. For the user's existing V5 93-GiB inventory, safely adopt
    legacy outputs only when the exact expected shard set exists and the final shard's
    record cardinality matches the manifest. Atomic FramedWriter renames guarantee all
    earlier full shards were closed completely.
    """
    base=Path(base);expected_records=max(0,int(expected_records));rps=max(1,int(records_per_shard))
    expected_shards=(expected_records+rps-1)//rps if expected_records else 0
    marker=_inventory_done_path(base)
    src=Path(source_path).resolve() if source_path is not None else None
    if marker.exists():
        try:
            d=json.loads(marker.read_text(encoding='utf-8'))
            marker_ok=int(d.get('expected_records',-1))==expected_records and int(d.get('records_per_shard',-1))==rps
            if src is not None:
                st=src.stat();marker_ok=marker_ok and str(d.get('source_path',''))==str(src) and int(d.get('source_size',-1))==int(st.st_size) and int(d.get('source_mtime_ns',-1))==int(st.st_mtime_ns)
            if marker_ok:
                paths=[_inventory_shard_path(base,i) for i in range(expected_shards)]
                if all(q.is_file() for q in paths):return True,expected_shards
        except Exception:pass
    if expected_records==0:return True,0
    paths=[_inventory_shard_path(base,i) for i in range(expected_shards)]
    if not all(q.is_file() for q in paths):return False,expected_shards
    # Extra final outputs imply an interrupted/reconfigured scan; do not adopt.
    extras=list(base.parent.glob(base.name+'-*.mpk'))
    if len(extras)!=expected_shards:return False,expected_shards
    try:
        remainder=expected_records-(expected_shards-1)*rps
        if _count_framed_records_fast(paths[-1])!=remainder:return False,expected_shards
        # Spot-check first full shard when there is more than one; catches wrong records_per_shard.
        if expected_shards>1 and _count_framed_records_fast(paths[0])!=rps:return False,expected_shards
    except Exception:return False,expected_shards
    return True,expected_shards


def _cleanup_inventory_pack_outputs(base:Path|str)->None:
    base=Path(base)
    for q in list(base.parent.glob(base.name+'-*.mpk'))+list(base.parent.glob(base.name+'-*.mpk.tmp.*'))+[ _inventory_done_path(base) ]:
        try:q.unlink()
        except FileNotFoundError:pass


def _write_inventory_done(base:Path,source_path:Path,expected_records:int,records_per_shard:int,counts:Mapping[str,Any])->None:
    st=source_path.stat();atomic_json(_inventory_done_path(base),{
        'source_path':str(source_path.resolve()),'source_size':int(st.st_size),'source_mtime_ns':int(st.st_mtime_ns),
        'expected_records':int(expected_records),'records_per_shard':int(records_per_shard),'records':int(counts.get('records',0)),
        'bytes':int(counts.get('bytes',0)),'shards':int(counts.get('shards',0))})


def _scan_task(args):
    source,path,base,records_per_shard,bytes_per_token,*_=args;path=Path(path);base=Path(base)
    _cleanup_inventory_pack_outputs(base)
    counts={'records':0,'bytes':0,'shards':0};writer=None;in_shard=0;shard_i=0
    try:
        it=scan_code_pack(path) if source=='code' else scan_ptpack(path,source)
        for rec in it:
            if rec.get('kind') in {'document','code_file'} and rec.get('payload_len') is not None:
                rec['bytes_per_token']=float(bytes_per_token);rec['approx_tokens']=max(1,int(round(int(rec.get('payload_len') or 0)/max(float(bytes_per_token),1e-6))))
            if writer is None or in_shard>=records_per_shard:
                if writer is not None:writer.close()
                writer=FramedWriter(_inventory_shard_path(base,shard_i));counts['shards']+=1;shard_i+=1;in_shard=0
            writer.write(rec);in_shard+=1;counts['records']+=1;counts['bytes']+=int(rec.get('payload_len') or 0)
    finally:
        if writer is not None:writer.close()
    return source,str(path),counts


def _source_bytes_per_token(stage: Path) -> Dict[str,float]:
    out={s:4.0 for s in (*GENERIC_SOURCES,'code')};p=stage/'stage_manifest.json'
    if not p.exists():return out
    try:data=json.loads(p.read_text(encoding='utf-8'))
    except Exception:return out
    for source,state in (data.get('sources') or {}).items():
        try:
            tokens=int(state.get('tokens') or 0);text_bytes=int(state.get('text_bytes') or 0)
            if tokens>0 and text_bytes>0:out[str(source)]=text_bytes/tokens
        except Exception:pass
    return out


def _bounded_process_results(tasks,fn,workers:int,max_in_flight:int|None=None,mp_context=None):
    """Yield (task,result) while keeping only O(workers) Future objects alive."""
    from concurrent.futures import wait,FIRST_COMPLETED
    workers=max(1,int(workers));limit=max(workers,int(max_in_flight or workers*2));it=iter(tasks)
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp_context or process_context()) as ex:
        active={}
        def fill():
            while len(active)<limit:
                try:t=next(it)
                except StopIteration:return
                _wait_for_memory_headroom()
                active[ex.submit(fn,t)]=t
        fill()
        while active:
            done,_=wait(tuple(active),return_when=FIRST_COMPLETED)
            for fut in done:
                t=active.pop(fut);yield t,fut.result()
            fill()


def run_inventory(stage_dir: Path|str,out_dir: Path|str,workers:int=16,records_per_shard:int=8192,rebuild:bool=False) -> Dict[str,Any]:
    stage=Path(stage_dir).resolve();out=Path(out_dir).resolve()
    if rebuild and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);bpt=_source_bytes_per_token(stage);tasks=[];by_source={};adopted=0
    for source in (*GENERIC_SOURCES,'code'):
        for i,(pack,meta) in enumerate(_source_pack_entries(stage,source)):
            base=out/source/f'{source}-{i:05d}';expected=_inventory_expected_records(source,meta);expected_bytes=_inventory_expected_bytes(source,meta)
            ok,nsh=_inventory_pack_complete(base,expected,records_per_shard,pack)
            d=by_source.setdefault(source,{'records':0,'bytes':0,'packs':0,'inventory_shards':0})
            if ok:
                # Existing V5 output has no done marker. Adopt it once so all future resumes are O(1).
                counts={'records':expected,'bytes':expected_bytes,'shards':nsh};d['records']+=expected;d['bytes']+=expected_bytes;d['packs']+=1;d['inventory_shards']+=nsh;adopted+=1
                if not _inventory_done_path(base).exists():_write_inventory_done(base,pack,expected,records_per_shard,counts)
            else:
                tasks.append((source,pack,base,records_per_shard,float(bpt.get(source,4.0)),expected,expected_bytes))
    total_packs=adopted+len(tasks)
    if not total_packs:raise RuntimeError(f'no input packs found under {stage}')
    requested=max(1,int(workers));raw_cap=os.environ.get('SBP_INVENTORY_WORKERS')
    cap=max(1,int(raw_cap)) if raw_cap not in (None,'') else max(1,min(requested,os.cpu_count() or 8))
    eff=min(requested,cap,max(1,len(tasks) or 1))
    print(f'[inventory] packs={total_packs} resume_ok={adopted} pending={len(tasks)} workers={eff}/{requested} records/shard={records_per_shard}',flush=True)
    t0=time.perf_counter();done=0
    def consume(t,res):
        nonlocal done
        source,path,c=res;expected=int(t[5]);expected_bytes=int(t[6]);base=Path(t[2])
        if int(c['records'])!=expected:raise RuntimeError(f'inventory record mismatch {path}: got={c["records"]} expected={expected}')
        # Report bytes are locator payload bytes; manifest values avoid rescanning adopted legacy output.
        c=dict(c);c['bytes']=expected_bytes
        _write_inventory_done(base,Path(path),expected,records_per_shard,c)
        d=by_source.setdefault(source,{'records':0,'bytes':0,'packs':0,'inventory_shards':0});d['records']+=int(c['records']);d['bytes']+=int(c['bytes']);d['packs']+=1;d['inventory_shards']+=int(c['shards']);done+=1
        if done==len(tasks) or done%max(1,max(1,len(tasks))//10)==0:print(f'[inventory] completed pending={done}/{len(tasks)} total={adopted+done}/{total_packs}',flush=True)
    if tasks:
        if eff<=1:
            for t in tasks:consume(t,_scan_task(t))
        else:
            for t,res in _bounded_process_results(tasks,_scan_task,eff,max_in_flight=eff*2,mp_context=process_context()):consume(t,res)
    report={'stage':'inventory','input':str(stage),'output':str(out),'workers':eff,'requested_workers':requested,'records_per_shard':records_per_shard,'bytes_per_token':bpt,
            'resume_adopted_packs':adopted,'rescanned_packs':len(tasks),'sources':by_source,'elapsed_s':time.perf_counter()-t0}
    atomic_json(out/'inventory_report.json',report);success_marker(out,'inventory',report);return report


# ============================================================================
# BEGIN dedup.py
# ============================================================================
from collections import defaultdict, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib, itertools, math, os, re, shutil, time, unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

WORD_RE=re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ_][\wÀ-ÖØ-öø-ÿ'-]*|\d+(?:\.\d+)?",re.UNICODE)
RICH_RE=re.compile(r"\\[A-Za-z]+|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|===|!==|==|!=|<=|>=|::|->|=>|\+\+|--|&&|\|\||\*\*|[^\s]",re.UNICODE)
SOURCE_PRIORITY={'wiki':0,'academic':1,'math_nemotron':2,'math_finemath':3,'general':4,'code':0}
DEFAULT_THRESHOLDS={'natural':0.88,'math':0.97}
MASK64=(1<<64)-1
_np=None
_np_checked=False

def _get_numpy():
    global _np,_np_checked
    if not _np_checked:
        try:
            import numpy as np
            _np=np
        except Exception:
            _np=None
        _np_checked=True
    return _np


def normalize_exact(text:str,source:str)->str:
    text=unicodedata.normalize('NFKC',text)
    if source=='code':
        # Formatting-insensitive but identifier/number/comment-sensitive lexical fingerprint.
        return '\x1f'.join(RICH_RE.findall(text))
    return ' '.join(text.split())


def family(source:str)->str:
    if source=='code': return 'code'
    if source.startswith('math_'): return 'math'
    return 'natural'


def dedup_tokens(text:str,source:str)->List[str]:
    text=unicodedata.normalize('NFKC',text)
    return (RICH_RE if family(source) in {'math','code'} else WORD_RE).findall(text.lower())


def _mix64(x:int)->int:
    x=(x+0x9E3779B97F4A7C15)&MASK64
    x=((x^(x>>30))*0xBF58476D1CE4E5B9)&MASK64
    x=((x^(x>>27))*0x94D049BB133111EB)&MASK64
    return (x^(x>>31))&MASK64

@lru_cache(maxsize=131072)
def _token_hash(tok:str)->int:
    return int.from_bytes(hashlib.blake2b(tok.encode('utf-8'),digest_size=8).digest(),'little')


def hashed_shingles(tokens:Sequence[str],n:int)->Set[int]:
    if not tokens: return set()
    if len(tokens)<n:
        h=0xCBF29CE484222325
        for t in tokens: h=_mix64(h ^ _token_hash(t))
        return {h}
    out=set()
    # n is only 5/10. A compact token-hash combiner is substantially cheaper than hashing joined strings.
    th=[_token_hash(t) for t in tokens]
    for i in range(len(th)-n+1):
        h=0xCBF29CE484222325 ^ n
        for x in th[i:i+n]: h=_mix64(h ^ x)
        out.add(h)
    return out


def _shingle_features_np(np,th,nt:int,n:int,cap:int)->Tuple[Set[int],int,bool]:
    """Vectorized equivalent of the streaming window fold below (exact same hashes)."""
    import heapq
    if nt==0:return set(),0,False
    if nt<n:
        h=0xCBF29CE484222325
        for x in th.tolist():h=_mix64(h^int(x))
        return {h},1,False
    W=np.lib.stride_tricks.sliding_window_view(th,n);m=nt-n+1
    h=np.full(m,np.uint64(0xCBF29CE484222325^n),dtype=np.uint64)
    for j in range(n):
        x=h^W[:,j]
        x=x+np.uint64(0x9E3779B97F4A7C15)
        x=(x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB)
        h=x^(x>>np.uint64(31))
    uniq=set(h.tolist())
    if len(uniq)<=cap:return uniq,m,False
    return set(heapq.nsmallest(cap,uniq)),m,True


def _bounded_shingle_features(text:str,source:str,n:int,max_features:Optional[int]=None)->Tuple[Set[int],int,bool]:
    """Stream tokenization/shingling with a hard feature-memory bound.

    Normal documents retain the exact unique shingle set. Pathological giant
    documents switch to a deterministic bottom-k sample once the cap is reached.
    This avoids the old text -> token-list -> token-hash-list -> unbounded-set
    expansion that could multiply one document into gigabytes of temporary RAM.
    """
    import heapq
    from collections import deque
    cap=max(256,int(max_features if max_features is not None else os.environ.get('SBP_NEAR_FEATURE_CAP','32768')))
    normalized=unicodedata.normalize('NFKC',text).lower()
    rx=RICH_RE if family(source) in {'math','code'} else WORD_RE
    np=_get_numpy()
    if np is not None:
        lim=4*cap+n
        th=np.fromiter((_token_hash(m.group(0)) for m in itertools.islice(rx.finditer(normalized),lim+1)),dtype=np.uint64)
        if int(th.size)<=lim:return _shingle_features_np(np,th,int(th.size),n,cap)
    win=deque(maxlen=max(1,int(n)));selected=set();heap=[];windows=0;truncated=False
    def keep(h:int):
        nonlocal truncated
        if h in selected:return
        if len(selected)<cap:
            selected.add(h);heapq.heappush(heap,-h);return
        truncated=True
        largest=-heap[0]
        if h>=largest:return
        old=-heapq.heapreplace(heap,-h);selected.remove(old);selected.add(h)
    any_token=False
    for m in rx.finditer(normalized):
        any_token=True;win.append(_token_hash(m.group(0)))
        if len(win)<n:continue
        h=0xCBF29CE484222325 ^ n
        for x in win:h=_mix64(h ^ x)
        windows+=1;keep(h)
    if any_token and windows==0:
        h=0xCBF29CE484222325
        for x in win:h=_mix64(h ^ x)
        windows=1;keep(h)
    return selected,windows,truncated


def minhash_signature(features:Set[int],k:int=32)->Tuple[int,...]:
    if not features: return tuple([MASK64]*k)
    np=_get_numpy()
    if np is not None and len(features)>=16:
        arr=np.fromiter(features,dtype=np.uint64,count=len(features));out=[]
        for seed in range(k):
            x=arr ^ np.uint64(_mix64(0xD6E8FEB86659FD93 ^ seed))
            x=(x+np.uint64(0x9E3779B97F4A7C15))
            x=((x^(x>>np.uint64(30)))*np.uint64(0xBF58476D1CE4E5B9))
            x=((x^(x>>np.uint64(27)))*np.uint64(0x94D049BB133111EB))
            x=x^(x>>np.uint64(31));out.append(int(x.min()))
        return tuple(out)
    vals=list(features);out=[]
    for seed in range(k):
        salt=_mix64(0xD6E8FEB86659FD93 ^ seed);out.append(min(_mix64(v ^ salt) for v in vals))
    return tuple(out)


def band_keys(signature:Sequence[int],bands:int=8)->Iterator[Tuple[int,Tuple[int,...]]]:
    if not signature:return
    width=max(1,len(signature)//bands)
    for b in range(bands):
        s=b*width; e=len(signature) if b==bands-1 else min(len(signature),s+width)
        yield b,tuple(int(x) for x in signature[s:e])


def _band_key_bytes(signature:Sequence[int],bands:int=8)->bytes:
    """Pack the band keys of a minhash signature into bands*8 bytes.

    Each band key is reduced to a 64-bit blake2b digest. Equality of band keys is
    all the near phase needs, and this drops the 32x9-byte msgpack signature from
    every routed row (8 rows per document).
    """
    sig=[int(x) for x in signature]
    if not sig:return b''
    bands=max(1,int(bands));width=max(1,len(sig)//bands);out=bytearray()
    for b in range(bands):
        s=b*width;e=len(sig) if b==bands-1 else min(len(sig),s+width)
        chunk=sig[s:e]
        out+=hashlib.blake2b(struct.pack(f'<{len(chunk)}Q',*chunk),digest_size=8).digest() if chunk else b'\0'*8
    return bytes(out)


def jaccard(a:Set[int],b:Set[int])->float:
    if not a and not b:return 1.0
    if not a or not b:return 0.0
    inter=len(a&b); return inter/(len(a)+len(b)-inter)


def _priority(rec:Mapping[str,Any])->Tuple[int,int]:
    inv=rec.get('inv',rec); return (SOURCE_PRIORITY.get(str(inv.get('source')),99),int(inv.get('record_id',0)))


def _fingerprint_record(inv:Mapping[str,Any],signature_size:int=32,need_near:bool=True)->Dict[str,Any]:
    text=read_text(inv);source=str(inv['source']);norm=normalize_exact(text,source)
    exact=hashlib.blake2b(norm.encode('utf-8'),digest_size=16).digest();fam=family(source);n=10 if fam=='code' else 5
    if (not need_near) or fam=='code':
        sig=();shingle_count=0;truncated=False
    else:
        feats,windows,truncated=_bounded_shingle_features(text,source,n)
        sig=minhash_signature(feats,signature_size)
        shingle_count=int(windows if truncated else len(feats))
    return {'inv':dict(inv),'exact_hash':exact,'scope':'code' if fam=='code' else 'text','family':fam,'shingle_n':n,'shingle_count':shingle_count,'signature':sig,'feature_sampled':bool(truncated)}


def _dedup_worker_limits(requested:int)->Dict[str,int]:
    """Worker caps for large-scale dedup, auto-scaled to the host.

    Every dedup worker is sqlite/bloom-backed with a small bounded cache
    (16 MiB per connection) plus fixed-size feature buffers, so the real
    constraint is total RAM headroom (page cache, OS) rather than a fixed
    1-2 worker pool. Defaults therefore scale with CPU count and available
    memory; environment variables still override each cap explicitly.
    """
    req=max(1,int(requested))
    cpu=os.cpu_count() or 8
    budget=max(4.0,_mem_available_gib()*0.5)
    def cap(name,cpu_div,per_worker_gib):
        raw=os.environ.get(name)
        if raw is not None:
            try:return max(1,int(raw))
            except Exception:pass
        by_cpu=max(1,cpu//cpu_div);by_mem=max(1,int(budget/per_worker_gib))
        return max(1,min(req,by_cpu,by_mem))
    return {
        'fingerprint':cap('SBP_FP_WORKERS',2,0.5),
        'exact':cap('SBP_EXACT_WORKERS',4,1.0),
        'route':1,
        'near':cap('SBP_NEAR_WORKERS',2,0.75),
    }


def _partition_from_run_name(path:Path|str)->int:
    m=re.match(r'^p(\d{5})(?:-r\d+)?$',Path(path).stem)
    if not m:raise ValueError(f'cannot parse partition from {path}')
    return int(m.group(1))


def _mem_available_gib()->float:
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])/1024/1024
    except Exception:pass
    return 1e9


def _wait_for_memory_headroom(min_gib:float|None=None)->None:
    floor=float(os.environ.get('SBP_MIN_AVAILABLE_GIB',str(16.0 if min_gib is None else min_gib)))
    if floor<=0:return
    warned=False
    while True:
        avail=_mem_available_gib()
        if avail>=floor:return
        if not warned:
            print(f'[memory-guard] MemAvailable={avail:.1f} GiB < {floor:.1f} GiB; pausing new worker submissions',flush=True)
            warned=True
        time.sleep(1.0)


def _fingerprint_shard(args):
    """Write exactly one atomic fingerprint run per inventory shard."""
    shard,run_dir,parts,signature_size,need_near=args;shard=Path(shard);run_dir=Path(run_dir)
    dest=run_dir/shard.parent.name/shard.name;dest.parent.mkdir(parents=True,exist_ok=True);count=0
    with FramedWriter(dest) as w:
        for inv in iter_frames(shard):
            if inv.get('kind')=='document' or (inv.get('kind')=='code_file' and inv.get('countable') and inv.get('language')):
                fp=_fingerprint_record(inv,signature_size,need_near=need_near)
                fp['exact_part']=int.from_bytes(fp['exact_hash'][:8],'little')%int(parts)
                w.write(fp);count+=1
    return count,str(dest)


def _route_fingerprint_runs(run_paths:Sequence[Path],exact_root:Path|str,parts:int)->int:
    """Route atomic fingerprint runs into only `parts` global files.

    The previous layout created up to exact_partitions files *per inventory shard*
    (tens of millions of files at 250B). This global LRU router bounds both RAM
    and inode count.
    """
    from collections import OrderedDict
    exact_root=Path(exact_root)
    if exact_root.is_symlink() or exact_root.exists():_rmtree_any(exact_root,keep_link=True)
    exact_root.mkdir(parents=True,exist_ok=True)
    max_open=max(8,int(os.environ.get('SBP_EXACT_OPEN_WRITERS','64')));writers=OrderedDict();count=0
    def writer(part:int):
        w=writers.pop(part,None)
        if w is None:w=_AppendFramedWriter(exact_root/f'p{part:05d}.mpk')
        writers[part]=w
        while len(writers)>max_open:
            _k,old=writers.popitem(last=False);old.close()
        return w
    try:
        for rp in run_paths:
            for fp in iter_frames(rp):
                part=int(fp.pop('exact_part',int.from_bytes(fp['exact_hash'][:8],'little')%int(parts)))
                writer(part).write(fp);count+=1
    finally:
        for w in writers.values():w.close()
    return count


def _exact_partition(args):
    """Exact-dedup one hash partition using SQLite external sort, bounded RAM."""
    import sqlite3
    part,files,out_dir=args;out_dir=Path(out_dir)
    tmpdir=Path(os.environ.get('SBP_EXACT_SQLITE_DIR') or (out_dir/'_exact_sqlite'));tmpdir.mkdir(parents=True,exist_ok=True)
    db=tmpdir/f'p{int(part):05d}.sqlite'
    db.unlink(missing_ok=True)
    conn=sqlite3.connect(str(db),timeout=120.0)
    conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF')
    conn.execute('PRAGMA temp_store=FILE');conn.execute('PRAGMA cache_size=-16384')
    conn.execute('CREATE TABLE r(scope TEXT NOT NULL, h BLOB NOT NULL, pri INTEGER NOT NULL, rid TEXT NOT NULL, payload BLOB NOT NULL)')
    total=0;batch=[];bands=int(os.environ.get('SBP_LSH_BANDS','8'))
    try:
        for p in files:
            for r in iter_frames(p):
                inv=r.get('inv',r);rid=int(inv.get('record_id',0));src=str(inv.get('source') or '')
                compact={'inv':inv,'family':str(r.get('family') or family(src)),'shingle_n':int(r.get('shingle_n',10 if family(src)=='code' else 5)),
                         'shingle_count':int(r.get('shingle_count',0)),'keys':_band_key_bytes(r.get('signature') or (),bands)}
                batch.append((str(r.get('scope','text')),sqlite3.Binary(bytes(r['exact_hash'])),int(_priority(r)[0]),_u64_key(rid),sqlite3.Binary(msgpack.packb(compact,use_bin_type=True))))
                total+=1
                if len(batch)>=1000:
                    conn.executemany('INSERT INTO r VALUES(?,?,?,?,?)',batch);batch.clear()
        if batch:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?)',batch);batch.clear()
        conn.commit();conn.execute('CREATE INDEX idx_r_sort ON r(scope,h,pri,rid)');conn.commit()
        sp=out_dir/'exact_survivors'/f'p{int(part):05d}.mpk';dp=out_dir/'exact_drops'/f'p{int(part):05d}.mpk'
        survivors=drops=0;prev=None;winner=None
        with FramedWriter(sp) as sw,FramedWriter(dp) as dw:
            cur=conn.execute('SELECT scope,h,payload FROM r ORDER BY scope,h,pri,rid')
            for scope,h,payload in cur:
                r=msgpack.unpackb(payload,raw=False,strict_map_key=False);key=(scope,bytes(h))
                if key!=prev:
                    sw.write(r);survivors+=1;winner=r;prev=key
                else:
                    inv=r['inv'];winv=winner['inv']
                    dw.write({'record_id':int(inv['record_id']),'winner_id':int(winv['record_id']),'reason':'exact','source':inv['source']});drops+=1
        atomic_json(out_dir/'exact_stats'/f'p{int(part):05d}.json',{'total':int(total),'survivors':int(survivors),'drops':int(drops)})
        return total,survivors,drops,str(sp),str(dp)
    finally:
        conn.close();db.unlink(missing_ok=True)

class _AppendFramedWriter:
    """Append framed records while writing magic exactly once."""
    def __init__(self,path:Path|str,magic:bytes=MAGIC):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        exists=self.path.exists() and self.path.stat().st_size>0
        self.f=self.path.open('ab' if exists else 'wb');self.magic=magic
        if not exists:self.f.write(magic)
    def write(self,obj:Any)->None:
        payload=msgpack.packb(obj,use_bin_type=True);self.f.write(FRAME.pack(len(payload)));self.f.write(payload)
    def close(self):
        if self.f is not None:self.f.flush();self.f.close();self.f=None


def _route_lsh(args):
    """Route LSH rows directly to disk with a small LRU of open writers."""
    from collections import OrderedDict
    survivor_path,run_root,lsh_parts,bands=args;survivor_path=Path(survivor_path);run_root=Path(run_root)
    base=run_root/survivor_path.stem;base.mkdir(parents=True,exist_ok=True)
    max_open=max(8,int(os.environ.get('SBP_LSH_OPEN_WRITERS','1024')));writers=OrderedDict();count=0
    def writer(part:int):
        w=writers.pop(part,None)
        if w is None:w=_AppendFramedWriter(base/f'p{part:05d}.mpk')
        writers[part]=w
        while len(writers)>max_open:
            _k,old=writers.popitem(last=False);old.close()
        return w
    try:
        for fp in iter_frames(survivor_path):
            if fp['family']=='code' or not fp.get('signature'):continue
            inv=fp['inv'];compact={'record_id':int(inv['record_id']),'source':inv['source'],'family':fp['family'],'shingle_n':int(fp['shingle_n']),
                'shingle_count':int(fp['shingle_count']),'pack_path':inv['pack_path'],'offset':int(inv['offset']),'format':inv['format'],'signature':fp['signature']}
            for band,key in band_keys(fp['signature'],bands):
                raw=repr((band,key)).encode();bucket=hashlib.blake2b(raw,digest_size=8).digest();part=int.from_bytes(bucket,'little')%lsh_parts
                writer(part).write({'band':band,'key':key,'rec':compact});count+=1
    finally:
        for w in writers.values():w.close()
    return count


def _route_lsh_global(survivor_paths:Sequence[Path],run_root:Path|str,lsh_parts:int,bands:int)->int:
    """Single streaming global LSH router writing compact fixed rows.

    Row layout: (band, key8, rid, sub16, recb). The 32-value signature is replaced
    by `bands` 64-bit band-key digests computed in the parallel exact phase, and
    pack paths are interned into a sidecar, so a row shrinks from ~450B to ~130B.
    """
    from collections import OrderedDict
    run_root=Path(run_root)
    if run_root.is_symlink() or run_root.exists():_rmtree_any(run_root,keep_link=True)
    run_root.mkdir(parents=True,exist_ok=True)
    max_open=max(8,int(os.environ.get('SBP_LSH_OPEN_WRITERS','1024')));writers=OrderedDict();count=0
    bands=int(bands);lsh_parts=int(lsh_parts);salt=[_mix64(0x9E3779B97F4A7C15*(b+1)) for b in range(bands)]
    packs={};pack_meta=[];sources={};source_list=[]
    def writer(part:int):
        w=writers.pop(part,None)
        if w is None:w=_AppendFramedWriter(run_root/f'p{part:05d}.mpk')
        writers[part]=w
        while len(writers)>max_open:
            _k,old=writers.popitem(last=False);old.close()
        return w
    try:
        for survivor_path in survivor_paths:
            for fp in iter_frames(survivor_path):
                keys=fp.get('keys')
                if not keys:continue
                if len(keys)!=bands*8:raise RuntimeError(f'band key size mismatch: {len(keys)} != {bands*8}')
                inv=fp['inv'];src=str(inv['source'])
                sid=sources.get(src)
                if sid is None:sid=len(source_list);sources[src]=sid;source_list.append(src)
                pp=str(inv['pack_path']);pid=packs.get(pp)
                if pid is None:pid=len(pack_meta);packs[pp]=pid;pack_meta.append((pp,str(inv['format'])))
                rid=int(inv['record_id'])
                recb=msgpack.packb((sid,int(fp['shingle_n']),int(fp['shingle_count']),pid,int(inv['offset']),keys),use_bin_type=True)
                sub=keys[:16]
                for b in range(bands):
                    k=keys[b*8:(b+1)*8]
                    part=int(_mix64(int.from_bytes(k,'little')^salt[b])%lsh_parts)
                    writer(part).write((b,k,rid,sub,recb));count+=1
    finally:
        for w in writers.values():w.close()
    meta=run_root.parent/'lsh_meta.mpk';tmp=meta.with_name(meta.name+f'.tmp.{os.getpid()}')
    tmp.write_bytes(msgpack.packb({'packs':pack_meta,'sources':source_list,'bands':bands},use_bin_type=True))
    os.replace(tmp,meta)
    return count

def _locator_text(rec:Mapping[str,Any])->str:
    return read_text({'format':rec['format'],'pack_path':rec['pack_path'],'offset':rec['offset']})


def _near_partition(args):
    """Verify one LSH partition with disk-backed grouping and bounded feature cache."""
    import sqlite3
    part,files,out_path,thresholds,max_bucket,bands,meta_path=args;out_path=Path(out_path)
    meta=msgpack.unpackb(Path(meta_path).read_bytes(),raw=False,strict_map_key=False)
    packs=meta['packs'];srcs=meta['sources'];pack_paths=[p[0] for p in packs];pack_fmts=[p[1] for p in packs]
    fam_of=[family(s) for s in srcs]
    tmpdir=Path(os.environ.get('SBP_NEAR_SQLITE_DIR') or (out_path.parent/'_near_sqlite'));tmpdir.mkdir(parents=True,exist_ok=True)
    db=tmpdir/f'p{int(part):05d}.sqlite';db.unlink(missing_ok=True)
    conn=sqlite3.connect(str(db),timeout=120.0)
    conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF');conn.execute('PRAGMA temp_store=FILE');conn.execute('PRAGMA cache_size=-16384')
    conn.execute('CREATE TABLE r(band INTEGER NOT NULL,k BLOB NOT NULL,sub BLOB NOT NULL,rid BLOB NOT NULL,payload BLOB NOT NULL)')
    batch=[]
    for p in files:
        for band,key,rid,sub,recb in iter_frames(p):
            batch.append((int(band),sqlite3.Binary(bytes(key)),sqlite3.Binary(bytes(sub)),sqlite3.Binary(int(rid).to_bytes(8,'big')),sqlite3.Binary(bytes(recb))))
            if len(batch)>=2000:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?)',batch);batch.clear()
    if batch:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?)',batch);batch.clear()
    conn.commit();conn.execute('CREATE INDEX idx_r_group ON r(band,k,sub,rid)');conn.commit()
    candidates=edges_n=oversized=0
    def first_shared_band(ka,kb):
        for bb in range(bands):
            i=bb*8
            if ka[i:i+8]==kb[i:i+8]:return bb
        return None
    def process_group(current_band,rows,writer):
        nonlocal candidates,edges_n
        cache={}
        def feats(r):
            rid=r[0]
            if rid not in cache:
                text=read_text({'format':pack_fmts[r[4]],'pack_path':pack_paths[r[4]],'offset':r[5]})
                cache[rid]=_bounded_shingle_features(text,srcs[r[1]],int(r[2]),int(os.environ.get('SBP_NEAR_VERIFY_FEATURE_CAP','16384')))[0]
            return cache[rid]
        for a,b in itertools.combinations(rows,2):
            if fam_of[a[1]]!=fam_of[b[1]]:continue
            if current_band != first_shared_band(a[6],b[6]):continue
            candidates+=1
            ca,cb=int(a[3]),int(b[3]);fam=fam_of[a[1]];threshold=float(thresholds.get(fam,DEFAULT_THRESHOLDS[fam]))
            if min(ca,cb)/max(ca,cb,1)<threshold:continue
            score=jaccard(feats(a),feats(b))
            if score>=threshold:writer.write({'a':{'record_id':a[0],'source':srcs[a[1]]},'b':{'record_id':b[0],'source':srcs[b[1]]},'score':score});edges_n+=1
    def load_rows(sql,params):
        rows=[]
        for ridb,payload in conn.execute(sql,params):
            sid,n,c,pid,off,keys=msgpack.unpackb(payload,raw=False,strict_map_key=False)
            rows.append((int.from_bytes(ridb,'big'),int(sid),int(n),int(c),int(pid),int(off),keys))
        return rows
    with FramedWriter(out_path) as w:
        groups=conn.execute('SELECT band,k,COUNT(*) FROM r GROUP BY band,k ORDER BY band,k')
        for band,k,n in groups:
            if int(n)<=int(max_bucket):
                process_group(int(band),load_rows('SELECT rid,payload FROM r WHERE band=? AND k=? ORDER BY rid',(band,k)),w)
            else:
                oversized+=1
                subs=list(conn.execute('SELECT sub,COUNT(*) FROM r WHERE band=? AND k=? GROUP BY sub ORDER BY sub',(band,k)))
                for sub,_sn in subs:
                    process_group(int(band),load_rows('SELECT rid,payload FROM r WHERE band=? AND k=? AND sub=? ORDER BY rid LIMIT ?',(band,k,sub,int(max_bucket))),w)
    conn.close();db.unlink(missing_ok=True)
    return {'partition':part,'candidates':candidates,'edges':edges_n,'oversized_buckets':oversized}

def _read_all_drop_ids(paths:Iterable[Path])->Tuple[Set[int],List[dict]]:
    """Legacy small-corpus helper. Production paths use the disk-backed drop index below."""
    ids=set(); rows=[]
    for p in paths:
        for r in iter_frames(p):ids.add(int(r['record_id']));rows.append(r)
    return ids,rows


def _u64_key(value:Any)->str:
    return f'{int(value) & ((1<<64)-1):016x}'


class _Bloom64:
    """Fixed-memory no-false-negative membership prefilter for uint64 record ids."""
    def __init__(self,mib:int=64,hashes:int=3):
        nbytes=max(1,int(mib))*(1<<20);self.bits=bytearray(nbytes);self.nbits=nbytes*8;self.hashes=max(1,int(hashes))
    def _positions(self,value:int):
        x=int(value)&((1<<64)-1)
        h1=_mix64(x ^ 0x9E3779B97F4A7C15);h2=_mix64(x ^ 0xD6E8FEB86659FD93)|1
        for i in range(self.hashes):yield (h1+i*h2+i*i*0x94D049BB133111EB)%self.nbits
    def add(self,value:int)->None:
        for pos in self._positions(value):self.bits[pos>>3]|=1<<(pos&7)
    def maybe(self,value:int)->bool:
        return all(self.bits[pos>>3]&(1<<(pos&7)) for pos in self._positions(value))


def _open_scale_sqlite(path:Path|str):
    import sqlite3
    conn=sqlite3.connect(str(path),timeout=120.0)
    conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF');conn.execute('PRAGMA temp_store=FILE')
    conn.execute('PRAGMA cache_size=-131072');conn.execute('PRAGMA locking_mode=EXCLUSIVE')
    return conn


def _open_plan_sqlite(path:Path|str):
    """Crash-safe planner DB: WAL keeps per-file checkpoints recoverable."""
    import sqlite3
    conn=sqlite3.connect(str(path),timeout=120.0)
    conn.isolation_level=None
    conn.execute('PRAGMA journal_mode=WAL');conn.execute('PRAGMA synchronous=NORMAL');conn.execute('PRAGMA temp_store=FILE')
    conn.execute('PRAGMA cache_size=-2097152');conn.execute('PRAGMA locking_mode=EXCLUSIVE')
    return conn


def _db_existing_keys(conn,table:str,keys:Sequence[str])->Set[str]:
    out=set();keys=list(dict.fromkeys(str(x) for x in keys))
    for i in range(0,len(keys),800):
        part=keys[i:i+800]
        if not part:continue
        q=','.join('?' for _ in part)
        out.update(str(r[0]) for r in conn.execute(f'SELECT record_key FROM {table} WHERE record_key IN ({q})',part))
    return out


def _finalize_dedup_external(out:Path|str,survivor_paths:Sequence[Path],near_paths:Sequence[Path])->Dict[str,Any]:
    """Resolve exact/near drops and filter survivors with bounded RAM.

    Near-edge global ordering is delegated to an on-disk SQLite B-tree. A fixed-size
    Bloom filter avoids a database lookup for the common case where neither endpoint
    has been dropped yet. `drop_index.sqlite` remains as the production membership
    index used by the code semantic stage; the much larger temporary edge DB is removed.
    """
    import sqlite3
    out=Path(out);drop_final=out/'drop_index.sqlite'
    if drop_final.is_symlink():drop_final=Path(os.path.realpath(drop_final))
    tmp_root=Path(os.environ.get('SBP_DEDUP_TMP_DIR') or drop_final.parent);tmp_root.mkdir(parents=True,exist_ok=True)
    drop_tmp=tmp_root/'drop_index.sqlite.tmp';edge_db=tmp_root/'near_edges.sqlite.tmp'
    for q in (drop_tmp,drop_final,edge_db):
        try:q.unlink()
        except FileNotFoundError:pass
    drop=_open_scale_sqlite(drop_tmp)
    drop.execute('CREATE TABLE drops(record_key TEXT PRIMARY KEY,record_id TEXT NOT NULL,winner_id TEXT,reason TEXT NOT NULL,score REAL,source TEXT) WITHOUT ROWID')
    bloom=_Bloom64(int(os.environ.get('SBP_DROP_BLOOM_MIB','64')));exact_dropped=0
    try:
        drop.execute('BEGIN')
        for ep in sorted((out/'exact_drops').glob('*.mpk')):
            for r in iter_frames(ep):
                rid=int(r['record_id']);key=_u64_key(rid);winner=r.get('winner_id')
                cur=drop.execute('INSERT OR IGNORE INTO drops VALUES(?,?,?,?,?,?)',(key,str(rid),None if winner is None else str(int(winner)),'exact',None,str(r.get('source') or '')))
                if cur.rowcount:
                    exact_dropped+=1;bloom.add(rid)
        drop.commit()

        near_dropped=0;edge_count=0
        if near_paths:
            edges=_open_scale_sqlite(edge_db)
            edges.execute('CREATE TABLE edges(sort_pri INTEGER NOT NULL,sort_id TEXT NOT NULL,score REAL NOT NULL,min_id TEXT NOT NULL,a_key TEXT NOT NULL,a_id TEXT NOT NULL,a_source TEXT NOT NULL,b_key TEXT NOT NULL,b_id TEXT NOT NULL,b_source TEXT NOT NULL)')
            edges.execute('BEGIN')
            for np in near_paths:
                for e in iter_frames(np):
                    a,b=e['a'],e['b'];ai,bi=int(a['record_id']),int(b['record_id']);asrc,bsrc=str(a['source']),str(b['source']);ap=(SOURCE_PRIORITY.get(asrc,99),ai);bp=(SOURCE_PRIORITY.get(bsrc,99),bi);mp=ap if ap<=bp else bp
                    edges.execute('INSERT INTO edges VALUES(?,?,?,?,?,?,?,?,?,?)',(int(mp[0]),_u64_key(mp[1]),float(e['score']),_u64_key(min(ai,bi)),_u64_key(ai),str(ai),asrc,_u64_key(bi),str(bi),bsrc));edge_count+=1
                    if edge_count%200000==0:edges.commit();edges.execute('BEGIN')
            edges.commit();edges.execute('CREATE INDEX edge_rank ON edges(sort_pri,sort_id,score DESC,min_id)');edges.commit()
            drop.execute('BEGIN');pending=0
            for row in edges.execute('SELECT a_key,a_id,a_source,b_key,b_id,b_source,score FROM edges ORDER BY sort_pri,sort_id,score DESC,min_id'):
                ak,ais,asrc,bk,bis,bsrc,score=row;ai,bi=int(ais),int(bis)
                if bloom.maybe(ai) or bloom.maybe(bi):
                    if drop.execute('SELECT 1 FROM drops WHERE record_key IN (?,?) LIMIT 1',(ak,bk)).fetchone():continue
                ap=(SOURCE_PRIORITY.get(asrc,99),ai);bp=(SOURCE_PRIORITY.get(bsrc,99),bi)
                if ap<=bp:winner,loser,lsrc=ai,bi,bsrc
                else:winner,loser,lsrc=bi,ai,asrc
                cur=drop.execute('INSERT OR IGNORE INTO drops VALUES(?,?,?,?,?,?)',(_u64_key(loser),str(loser),str(winner),'near',float(score),lsrc))
                if cur.rowcount:
                    near_dropped+=1;pending+=1;bloom.add(loser)
                    if pending>=100000:drop.commit();drop.execute('BEGIN');pending=0
            drop.commit();edges.close()
            try:edge_db.unlink()
            except FileNotFoundError:pass

        drop_count=exact_dropped+near_dropped
        # Compatibility/audit stream; no Python list/set of all drops is created.
        with FramedWriter(out/'drop_ids.mpk') as w:
            for rid,winner,reason,score,source in drop.execute('SELECT record_id,winner_id,reason,score,source FROM drops ORDER BY record_key'):
                rec={'record_id':int(rid),'winner_id':None if winner is None else int(winner),'reason':reason,'source':source}
                if score is not None:rec['score']=float(score)
                w.write(rec)

        surv_root=out/'survivors'
        if surv_root.exists():shutil.rmtree(surv_root)
        counts=defaultdict(int)
        for sp in survivor_paths:
            writers={};batch=[]
            def flush_batch():
                if not batch:return
                maybe=[_u64_key(int(fp['inv']['record_id'])) for fp in batch if bloom.maybe(int(fp['inv']['record_id']))]
                dead=_db_existing_keys(drop,'drops',maybe) if maybe else set()
                for fp in batch:
                    r=fp['inv'];rid=int(r['record_id'])
                    if _u64_key(rid) in dead:continue
                    source=str(r['source']);dest=surv_root/source/sp.name
                    if source not in writers:writers[source]=FramedWriter(dest)
                    writers[source].write(r);counts[source]+=1
                batch.clear()
            try:
                for fp in iter_frames(sp):
                    batch.append(fp)
                    if len(batch)>=2048:flush_batch()
                flush_batch()
            finally:
                for w in writers.values():w.close()
        drop.commit();drop.close();_replace_file(drop_tmp,drop_final)
        return {'exact_dropped':exact_dropped,'near_dropped':near_dropped,'drop_count':drop_count,'near_edges':edge_count,'survivors':dict(counts),'drop_index':str(drop_final),'selection_backend':'sqlite_external'}
    except Exception:
        try:drop.close()
        except Exception:pass
        raise
    finally:
        try:edge_db.unlink()
        except FileNotFoundError:pass


class _DropLookup:
    """Bounded-memory batched membership reader for the persistent dedup drop index."""
    def __init__(self,dedup_dir:Path|str,legacy_source:Optional[str]=None):
        import sqlite3
        self.path=Path(dedup_dir)/'drop_index.sqlite';self.conn=None;self.legacy=None;self.legacy_source=legacy_source
        if self.path.is_file():
            self.conn=sqlite3.connect(f'file:{self.path}?mode=ro',uri=True,timeout=120.0);self.conn.execute('PRAGMA query_only=ON');self.conn.execute('PRAGMA cache_size=-32768')
        else:
            p=Path(dedup_dir)/'drop_ids.mpk';self.legacy={int(r['record_id']) for r in iter_frames(p) if legacy_source is None or str(r.get('source'))==str(legacy_source)} if p.exists() else set()
    def contains_many(self,ids:Iterable[int],source:Optional[str]=None)->Set[int]:
        vals=list(dict.fromkeys(int(x) for x in ids))
        if self.conn is None:return {x for x in vals if x in self.legacy}
        out=set()
        for i in range(0,len(vals),800):
            part=vals[i:i+800];keys=[_u64_key(x) for x in part];q=','.join('?' for _ in keys)
            if source is None:rows=self.conn.execute(f'SELECT record_id FROM drops WHERE record_key IN ({q})',keys)
            else:rows=self.conn.execute(f'SELECT record_id FROM drops WHERE record_key IN ({q}) AND source=?',keys+[str(source)])
            out.update(int(r[0]) for r in rows)
        return out
    def close(self):
        if self.conn is not None:self.conn.close();self.conn=None



_BASE_DROP_FILTER_CACHE={}

class _BaseDropFilter:
    """One-time Bloom front-end for immutable base drop_index.sqlite."""
    def __init__(self,base_dedup_dir:Path|str):
        import sqlite3
        self.path=Path(base_dedup_dir)/'drop_index.sqlite'
        if not self.path.is_file():raise RuntimeError(f'base drop index missing: {self.path}')
        self.conn=sqlite3.connect(f'file:{self.path}?mode=ro',uri=True,timeout=120.0)
        self.conn.execute('PRAGMA query_only=ON');self.conn.execute('PRAGMA cache_size=-32768')
        self.bloom=_Bloom64(int(os.environ.get('SBP_BASE_DROP_BLOOM_MIB','64')))
        self.count=0
        for (rid,) in self.conn.execute('SELECT record_id FROM drops'):
            self.bloom.add(int(rid));self.count+=1
        print(f'[incremental base-ref] drop Bloom ready rows={self.count:,} mib={os.environ.get("SBP_BASE_DROP_BLOOM_MIB","64")}',flush=True)
    def contains_many(self,ids:Iterable[int])->Set[int]:
        vals=list(dict.fromkeys(int(x) for x in ids))
        maybe=[x for x in vals if self.bloom.maybe(x)]
        if not maybe:return set()
        out=set()
        for i in range(0,len(maybe),800):
            part=maybe[i:i+800];keys=[_u64_key(x) for x in part];q=','.join('?' for _ in keys)
            out.update(int(r[0]) for r in self.conn.execute(f'SELECT record_id FROM drops WHERE record_key IN ({q})',keys))
        return out

def _get_base_drop_filter(base_dedup_dir:Path|str)->_BaseDropFilter:
    key=str(Path(base_dedup_dir).resolve());obj=_BASE_DROP_FILTER_CACHE.get(key)
    if obj is None:
        obj=_BaseDropFilter(key);_BASE_DROP_FILTER_CACHE[key]=obj
    return obj

def _base_live_batch(base_lookup, rows:Sequence[Mapping[str,Any]])->List[Mapping[str,Any]]:
    if not rows:return []
    dead=base_lookup.contains_many(int(r['inv']['record_id']) for r in rows)
    return [r for r in rows if int(r['inv']['record_id']) not in dead]


def _exact_partition_incremental(args):
    """Exact dedup for augmentation with immutable base precedence."""
    import sqlite3
    part,aug_files,base_file,base_dedup_dir,out_dir=args;out_dir=Path(out_dir);base_file=Path(base_file)
    tmpdir=Path(os.environ.get('SBP_EXACT_SQLITE_DIR') or (out_dir/'_exact_sqlite'));tmpdir.mkdir(parents=True,exist_ok=True)
    db=tmpdir/f'p{int(part):05d}.sqlite';db.unlink(missing_ok=True)
    conn=sqlite3.connect(str(db),timeout=120.0)
    conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF');conn.execute('PRAGMA temp_store=FILE');conn.execute('PRAGMA cache_size=-16384')
    conn.execute('CREATE TABLE r(scope TEXT NOT NULL,h BLOB NOT NULL,origin INTEGER NOT NULL,pri INTEGER NOT NULL,rid TEXT NOT NULL,payload BLOB NOT NULL)')
    total=0;base_live=0;batch=[]
    def add(r,origin):
        nonlocal total,base_live
        inv=r.get('inv',r);rid=int(inv.get('record_id',0))
        batch.append((str(r.get('scope','text')),sqlite3.Binary(bytes(r['exact_hash'])),int(origin),int(_priority(r)[0]),_u64_key(rid),sqlite3.Binary(msgpack.packb(r,use_bin_type=True))))
        if origin==0:base_live+=1
        else:total+=1
        if len(batch)>=1000:
            conn.executemany('INSERT INTO r VALUES(?,?,?,?,?,?)',batch);batch.clear()
    base_lookup=_get_base_drop_filter(base_dedup_dir)
    try:
        # Snapshot the tiny partition-local augmentation input first. Besides being
        # faster (base rows are only materialized on a matching hash), this makes
        # the worker independent of exact_runs after it starts scanning the large
        # immutable base partition.
        aug_keys=set()
        for ap in aug_files:
            ap=Path(ap)
            if not ap.is_file():raise RuntimeError(f'incremental exact aug input vanished before read: partition={part} path={ap}')
            for r in iter_frames(ap):
                aug_keys.add((str(r.get('scope','text')),bytes(r['exact_hash'])));add(r,1)
        if batch:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?,?)',batch);batch.clear()
        matched=[]
        for r in iter_frames(base_file):
            key=(str(r.get('scope','text')),bytes(r['exact_hash']))
            if key not in aug_keys:continue
            matched.append(r)
            if len(matched)>=2048:
                for x in _base_live_batch(base_lookup,matched):add(x,0)
                matched.clear()
        for x in _base_live_batch(base_lookup,matched):add(x,0)
        if batch:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?,?)',batch);batch.clear()
        conn.commit();conn.execute('CREATE INDEX idx_r_sort ON r(scope,h,origin,pri,rid)');conn.commit()
        sp=out_dir/'exact_survivors'/f'p{int(part):05d}.mpk';dp=out_dir/'exact_drops'/f'p{int(part):05d}.mpk'
        survivors=drops=base_drops=0;prev=None;winner=None;winner_origin=None
        with FramedWriter(sp) as sw,FramedWriter(dp) as dw:
            for scope,h,origin,payload in conn.execute('SELECT scope,h,origin,payload FROM r ORDER BY scope,h,origin,pri,rid'):
                r=msgpack.unpackb(payload,raw=False,strict_map_key=False);key=(scope,bytes(h));origin=int(origin)
                if key!=prev:
                    winner=r;winner_origin=origin;prev=key
                    if origin==1:sw.write(r);survivors+=1
                    continue
                if origin==0:
                    raise RuntimeError(f'base exact-survivor invariant violated in partition {part}: duplicate base hash')
                inv=r['inv'];winv=winner['inv'];reason='exact_base' if winner_origin==0 else 'exact'
                dw.write({'record_id':int(inv['record_id']),'winner_id':int(winv['record_id']),'reason':reason,'source':inv['source']})
                drops+=1;base_drops+=int(winner_origin==0)
        return total,survivors,drops,base_drops,base_live,str(sp),str(dp)
    finally:
        conn.close();db.unlink(missing_ok=True)


def _filter_base_lsh_partition(args):
    """Scan one immutable base exact-survivor partition against an mmap Bloom."""
    import mmap
    base_path,bloom_path,bloom_nbits,bloom_hashes,bands,out_path,base_dedup_dir,aug_families=args
    base_path=Path(base_path);out_path=Path(out_path);stats=Counter()
    lookup=_DropLookup(base_dedup_dir)
    bf=open(bloom_path,'rb');mm=mmap.mmap(bf.fileno(),0,access=mmap.ACCESS_READ)
    def maybe(value:int)->bool:
        x=int(value)&MASK64
        h1=_mix64(x ^ 0x9E3779B97F4A7C15);h2=_mix64(x ^ 0xD6E8FEB86659FD93)|1
        for i in range(int(bloom_hashes)):
            pos=(h1+i*h2+i*i*0x94D049BB133111EB)%int(bloom_nbits)
            if not (mm[pos>>3]&(1<<(pos&7))):return False
        return True
    def compact(fp):
        inv=fp['inv']
        return {'record_id':int(inv['record_id']),'source':inv['source'],'family':fp['family'],'shingle_n':int(fp['shingle_n']),
            'shingle_count':int(fp['shingle_count']),'pack_path':inv['pack_path'],'offset':int(inv['offset']),'format':inv['format'],'signature':fp['signature'],
            'feature_sampled':bool(fp.get('feature_sampled')),'origin':'base'}
    pending=[]
    with FramedWriter(out_path) as w:
        def flush():
            if not pending:return
            dead=lookup.contains_many(int(fp['inv']['record_id']) for fp,_hits in pending)
            for fp,hits in pending:
                rid=int(fp['inv']['record_id'])
                if rid in dead:continue
                rec=compact(fp);stats['base_live_candidates']+=1
                for band,key,bid in hits:
                    w.write({'band':band,'key':key,'bid':int(bid),'rec':rec});stats['base_rows']+=1
            pending.clear()
        for fp in iter_frames(base_path):
            if fp['family']=='code' or fp['family'] not in aug_families or not fp.get('signature'):continue
            stats['base_records_scanned']+=1;hits=[]
            for band,key in band_keys(fp['signature'],bands):
                raw=repr((band,key)).encode();bid=int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),'little')
                if maybe(bid):hits.append((band,key,bid))
            if not hits:continue
            stats['base_prefilter_candidates']+=1;pending.append((fp,hits))
            if len(pending)>=2048:flush()
        flush()
    mm.close();bf.close();lookup.close()
    return dict(stats)


def _route_lsh_incremental(base_survivor_paths:Sequence[Path],aug_survivor_paths:Sequence[Path],base_dedup_dir:Path|str,run_root:Path|str,lsh_parts:int,bands:int)->Dict[str,int]:
    """Parallel query-focused LSH route with immutable base precedence."""
    from collections import OrderedDict
    run_root=Path(run_root)
    if run_root.is_symlink() or run_root.exists():_rmtree_any(run_root,keep_link=True)
    run_root.mkdir(parents=True,exist_ok=True)
    max_open=max(8,int(os.environ.get('SBP_LSH_OPEN_WRITERS','1024')));writers=OrderedDict();stats=Counter()
    aug_bloom=_Bloom64(int(os.environ.get('SBP_AUG_LSH_BLOOM_MIB','128')));aug_families=set()
    def writer(part:int):
        w=writers.pop(part,None)
        if w is None:w=_AppendFramedWriter(run_root/f'p{part:05d}.mpk')
        writers[part]=w
        while len(writers)>max_open:
            _k,old=writers.popitem(last=False);old.close()
        return w
    def compact(fp):
        inv=fp['inv']
        return {'record_id':int(inv['record_id']),'source':inv['source'],'family':fp['family'],'shingle_n':int(fp['shingle_n']),
            'shingle_count':int(fp['shingle_count']),'pack_path':inv['pack_path'],'offset':int(inv['offset']),'format':inv['format'],'signature':fp['signature'],
            'feature_sampled':bool(fp.get('feature_sampled')),'origin':'aug'}
    # Route all augmentation bands and construct the no-false-negative query Bloom.
    for path in aug_survivor_paths:
        if not Path(path).is_file():raise RuntimeError(f'incremental near aug survivor missing: {path}')
        for fp in iter_frames(path):
            if fp['family']=='code' or not fp.get('signature'):continue
            aug_families.add(str(fp['family']));rec=compact(fp);stats['aug_records']+=1
            for band,key in band_keys(fp['signature'],bands):
                raw=repr((band,key)).encode();bid=int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),'little');part=bid%int(lsh_parts)
                aug_bloom.add(bid);writer(part).write({'band':band,'key':key,'rec':rec});stats['aug_rows']+=1
    stats['aug_bloom_mib']=int(os.environ.get('SBP_AUG_LSH_BLOOM_MIB','128'))
    for w in writers.values():w.close()
    writers.clear()
    if not stats['aug_rows']:return dict(stats)

    bloom_path=run_root/'_aug_lsh.bloom'
    with bloom_path.open('wb') as f:f.write(aug_bloom.bits)
    hit_root=run_root/'_base_hits';hit_root.mkdir(parents=True,exist_ok=True)
    scan_workers=max(1,min(len(base_survivor_paths),int(os.environ.get('SBP_BASE_LSH_SCAN_WORKERS','4'))))
    tasks=[]
    for base_path in base_survivor_paths:
        part=_partition_from_run_name(base_path);hit=hit_root/f'p{part:05d}.mpk'
        tasks.append((base_path,bloom_path,aug_bloom.nbits,aug_bloom.hashes,bands,hit,base_dedup_dir,tuple(sorted(aug_families))))
    print(f'[dedup 3/4 near] base query scan partitions={len(tasks)} workers={scan_workers} aug_families={sorted(aug_families)}',flush=True)
    if scan_workers<=1:
        iterator=((t,_filter_base_lsh_partition(t)) for t in tasks)
    else:
        iterator=_bounded_process_results(tasks,_filter_base_lsh_partition,scan_workers,max_in_flight=max(scan_workers*2,8),mp_context=process_context())
    done=0
    for t,res in iterator:
        done+=1
        for k,v in res.items():stats[k]+=int(v)
        if done%64==0 or done==len(tasks):
            print(f'[dedup 3/4 near] base query scan progress={done}/{len(tasks)} hits={stats["base_live_candidates"]:,} rows={stats["base_rows"]:,}',flush=True)
    # Merge only the sparse matched base rows into the same global LSH partitions.
    for hp in sorted(hit_root.glob('p*.mpk')):
        for x in iter_frames(hp):
            part=int(x.pop('bid'))%int(lsh_parts);writer(part).write(x)
    for w in writers.values():w.close()
    shutil.rmtree(hit_root,ignore_errors=True);bloom_path.unlink(missing_ok=True)
    return dict(stats)


def _near_partition_incremental(args):
    """Verify one combined LSH partition without truncating oversized buckets.

    Base-base pairs are irrelevant because base has already been deduplicated.
    For a hot (band,key) bucket, rows are processed in bounded tiles and every
    aug-aug and aug-base pair is covered. No LIMIT-based candidate loss occurs.
    """
    import sqlite3
    part,files,out_path,thresholds,max_bucket,bands=args;out_path=Path(out_path)
    tmpdir=Path(os.environ.get('SBP_NEAR_SQLITE_DIR') or (out_path.parent/'_near_sqlite'));tmpdir.mkdir(parents=True,exist_ok=True)
    db=tmpdir/f'p{int(part):05d}.sqlite';db.unlink(missing_ok=True)
    conn=sqlite3.connect(str(db),timeout=120.0)
    conn.execute('PRAGMA journal_mode=OFF');conn.execute('PRAGMA synchronous=OFF');conn.execute('PRAGMA temp_store=FILE');conn.execute('PRAGMA cache_size=-16384')
    conn.execute('CREATE TABLE r(band INTEGER NOT NULL,k BLOB NOT NULL,origin INTEGER NOT NULL,rid TEXT NOT NULL,payload BLOB NOT NULL)')
    batch=[]
    for p in files:
        for x in iter_frames(p):
            rec=x['rec'];origin=0 if rec.get('origin')=='base' else 1
            batch.append((int(x['band']),sqlite3.Binary(msgpack.packb(tuple(x['key']),use_bin_type=True)),origin,_u64_key(rec['record_id']),sqlite3.Binary(msgpack.packb(rec,use_bin_type=True))))
            if len(batch)>=2000:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?)',batch);batch.clear()
    if batch:conn.executemany('INSERT INTO r VALUES(?,?,?,?,?)',batch);batch.clear()
    conn.commit();conn.execute('CREATE INDEX idx_r_group ON r(band,k,origin,rid)');conn.commit()
    candidates=edges_n=oversized=base_edges=tiled_pairs=0;max_bucket_rows=0

    def first_shared_band(sa,sb):
        bk=dict(band_keys(sb,bands))
        for bb,ka in band_keys(sa,bands):
            if bk.get(bb)==ka:return bb
        return None

    def process_pairs(current_band,rows_a,rows_b,writer,same:bool):
        nonlocal candidates,edges_n,base_edges,tiled_pairs
        cache={}
        def feats(r):
            rid=(str(r.get('origin')),int(r['record_id']))
            if rid not in cache:
                text=_locator_text(r)
                cache[rid]=_bounded_shingle_features(text,r['source'],int(r['shingle_n']),int(os.environ.get('SBP_NEAR_VERIFY_FEATURE_CAP','16384')))[0]
            return cache[rid]
        pairs=itertools.combinations(rows_a,2) if same else itertools.product(rows_a,rows_b)
        for a,b in pairs:
            if a.get('origin')=='base' and b.get('origin')=='base':continue
            if a['family']!=b['family']:continue
            if current_band != first_shared_band(a['signature'],b['signature']):continue
            candidates+=1;tiled_pairs+=1
            ca,cb=int(a['shingle_count']),int(b['shingle_count'])
            threshold=float(thresholds.get(a['family'],DEFAULT_THRESHOLDS[a['family']]))
            if min(ca,cb)/max(ca,cb,1)<threshold:continue
            score=jaccard(feats(a),feats(b))
            if score>=threshold:
                writer.write({'a':a,'b':b,'score':score});edges_n+=1
                if a.get('origin')!=b.get('origin'):base_edges+=1

    def load_tile(band,k,origin,offset,limit):
        return [msgpack.unpackb(x[0],raw=False,strict_map_key=False) for x in
                conn.execute('SELECT payload FROM r WHERE band=? AND k=? AND origin=? ORDER BY rid LIMIT ? OFFSET ?',
                             (band,k,int(origin),int(limit),int(offset)))]

    tile=max(16,min(int(max_bucket),int(os.environ.get('SBP_NEAR_TILE_SIZE','128'))))
    with FramedWriter(out_path) as w:
        groups=conn.execute('SELECT band,k,COUNT(*) FROM r GROUP BY band,k ORDER BY band,k')
        for band,k,n in groups:
            n=int(n);max_bucket_rows=max(max_bucket_rows,n)
            if n<=int(max_bucket):
                rows=[msgpack.unpackb(x[0],raw=False,strict_map_key=False) for x in
                      conn.execute('SELECT payload FROM r WHERE band=? AND k=? ORDER BY rid',(band,k))]
                process_pairs(int(band),rows,rows,w,True)
                continue

            oversized+=1
            aug_n=int(conn.execute('SELECT COUNT(*) FROM r WHERE band=? AND k=? AND origin=1',(band,k)).fetchone()[0])
            base_n=n-aug_n
            if aug_n<=0:continue  # base-only hot bucket; base is immutable/already deduped.

            # Complete aug-aug coverage, tiled to bound RAM.
            for ai in range(0,aug_n,tile):
                arows=load_tile(band,k,1,ai,tile)
                process_pairs(int(band),arows,arows,w,True)
                for aj in range(ai+tile,aug_n,tile):
                    brows=load_tile(band,k,1,aj,tile)
                    process_pairs(int(band),arows,brows,w,False)

                # Complete aug-base coverage. Base-base is intentionally skipped.
                for bi in range(0,base_n,tile):
                    brows=load_tile(band,k,0,bi,tile)
                    process_pairs(int(band),arows,brows,w,False)

    conn.close();db.unlink(missing_ok=True)
    return {'partition':part,'candidates':candidates,'edges':edges_n,'base_edges':base_edges,
            'oversized_buckets':oversized,'max_bucket_rows':max_bucket_rows,'tiled_pairs':tiled_pairs,'tile_size':tile}


def _finalize_dedup_incremental(out:Path|str,aug_survivor_paths:Sequence[Path],near_paths:Sequence[Path])->Dict[str,Any]:
    """Finalize augmentation drops. Base endpoints are immutable and can never enter the drop index."""
    out=Path(out);drop_final=out/'drop_index.sqlite'
    if drop_final.is_symlink():drop_final=Path(os.path.realpath(drop_final))
    tmp_root=Path(os.environ.get('SBP_DEDUP_TMP_DIR') or drop_final.parent);tmp_root.mkdir(parents=True,exist_ok=True)
    drop_tmp=tmp_root/'drop_index.sqlite.tmp';edge_db=tmp_root/'near_edges.sqlite.tmp'
    for q in (drop_tmp,drop_final,edge_db):
        try:q.unlink()
        except FileNotFoundError:pass
    drop=_open_scale_sqlite(drop_tmp)
    drop.execute('CREATE TABLE drops(record_key TEXT PRIMARY KEY,record_id TEXT NOT NULL,winner_id TEXT,reason TEXT NOT NULL,score REAL,source TEXT) WITHOUT ROWID')
    bloom=_Bloom64(int(os.environ.get('SBP_DROP_BLOOM_MIB','64')));exact_dropped=exact_base_dropped=0
    try:
        drop.execute('BEGIN')
        for ep in sorted((out/'exact_drops').glob('*.mpk')):
            for r in iter_frames(ep):
                rid=int(r['record_id']);reason=str(r.get('reason') or 'exact');key=_u64_key(rid);winner=r.get('winner_id')
                cur=drop.execute('INSERT OR IGNORE INTO drops VALUES(?,?,?,?,?,?)',(key,str(rid),None if winner is None else str(int(winner)),reason,None,str(r.get('source') or '')))
                if cur.rowcount:
                    exact_dropped+=1;exact_base_dropped+=int(reason=='exact_base');bloom.add(rid)
        drop.commit()
        near_dropped=near_base_dropped=edge_count=0
        if near_paths:
            edges=_open_scale_sqlite(edge_db)
            edges.execute('CREATE TABLE edges(kind INTEGER NOT NULL,score REAL NOT NULL,min_id TEXT NOT NULL,a BLOB NOT NULL,b BLOB NOT NULL)')
            edges.execute('BEGIN')
            for np in near_paths:
                for e in iter_frames(np):
                    a,b=e['a'],e['b'];kind=0 if a.get('origin')!=b.get('origin') else 1
                    edges.execute('INSERT INTO edges VALUES(?,?,?,?,?)',(kind,float(e['score']),_u64_key(min(int(a['record_id']),int(b['record_id']))),msgpack.packb(a,use_bin_type=True),msgpack.packb(b,use_bin_type=True)));edge_count+=1
                    if edge_count%200000==0:edges.commit();edges.execute('BEGIN')
            edges.commit();edges.execute('CREATE INDEX edge_rank ON edges(kind,score DESC,min_id)');edges.commit()
            drop.execute('BEGIN');pending=0
            for kind,score,ab,bb in edges.execute('SELECT kind,score,a,b FROM edges ORDER BY kind,score DESC,min_id'):
                a=msgpack.unpackb(ab,raw=False,strict_map_key=False);b=msgpack.unpackb(bb,raw=False,strict_map_key=False)
                ao,bo=a.get('origin'),b.get('origin');ai,bi=int(a['record_id']),int(b['record_id'])
                if int(kind)==0:
                    base=a if ao=='base' else b;aug=b if ao=='base' else a;loser=int(aug['record_id'])
                    if bloom.maybe(loser) and drop.execute('SELECT 1 FROM drops WHERE record_key=?',(_u64_key(loser),)).fetchone():continue
                    cur=drop.execute('INSERT OR IGNORE INTO drops VALUES(?,?,?,?,?,?)',(_u64_key(loser),str(loser),str(int(base['record_id'])),'near_base',float(score),str(aug['source'])))
                    if cur.rowcount:near_dropped+=1;near_base_dropped+=1;pending+=1;bloom.add(loser)
                else:
                    if bloom.maybe(ai) or bloom.maybe(bi):
                        if drop.execute('SELECT 1 FROM drops WHERE record_key IN (?,?) LIMIT 1',(_u64_key(ai),_u64_key(bi))).fetchone():continue
                    ap=(SOURCE_PRIORITY.get(str(a['source']),99),ai);bp=(SOURCE_PRIORITY.get(str(b['source']),99),bi)
                    winner,loser,lsrc=(ai,bi,str(b['source'])) if ap<=bp else (bi,ai,str(a['source']))
                    cur=drop.execute('INSERT OR IGNORE INTO drops VALUES(?,?,?,?,?,?)',(_u64_key(loser),str(loser),str(winner),'near',float(score),lsrc))
                    if cur.rowcount:near_dropped+=1;pending+=1;bloom.add(loser)
                if pending>=100000:drop.commit();drop.execute('BEGIN');pending=0
            drop.commit();edges.close()
            try:edge_db.unlink()
            except FileNotFoundError:pass
        with FramedWriter(out/'drop_ids.mpk') as w:
            for rid,winner,reason,score,source in drop.execute('SELECT record_id,winner_id,reason,score,source FROM drops ORDER BY record_key'):
                rec={'record_id':int(rid),'winner_id':None if winner is None else int(winner),'reason':reason,'source':source}
                if score is not None:rec['score']=float(score)
                w.write(rec)
        surv_root=out/'survivors'
        if surv_root.exists():shutil.rmtree(surv_root)
        counts=defaultdict(int)
        for sp in aug_survivor_paths:
            writers={};batch=[]
            def flush_batch():
                if not batch:return
                maybe=[_u64_key(int(fp['inv']['record_id'])) for fp in batch if bloom.maybe(int(fp['inv']['record_id']))]
                dead=_db_existing_keys(drop,'drops',maybe) if maybe else set()
                for fp in batch:
                    r=fp['inv'];rid=int(r['record_id'])
                    if _u64_key(rid) in dead:continue
                    source=str(r['source']);dest=surv_root/source/sp.name
                    if source not in writers:writers[source]=FramedWriter(dest)
                    writers[source].write(r);counts[source]+=1
                batch.clear()
            try:
                for fp in iter_frames(sp):
                    batch.append(fp)
                    if len(batch)>=2048:flush_batch()
                flush_batch()
            finally:
                for w in writers.values():w.close()
        drop.commit();drop.close();_replace_file(drop_tmp,drop_final)
        return {'exact_dropped':exact_dropped,'exact_base_dropped':exact_base_dropped,'near_dropped':near_dropped,'near_base_dropped':near_base_dropped,
                'drop_count':exact_dropped+near_dropped,'near_edges':edge_count,'survivors':dict(counts),'drop_index':str(drop_final),'selection_backend':'sqlite_external_incremental_base_wins'}
    except Exception:
        try:drop.close()
        except Exception:pass
        raise
    finally:
        try:edge_db.unlink()
        except FileNotFoundError:pass


def run_dedup(inventory_dir:Path|str,out_dir:Path|str,workers:int=16,exact_partitions:int=1024,lsh_partitions:int=1024,
              near:bool=True,thresholds:Optional[Mapping[str,float]]=None,signature_size:int=32,bands:int=8,max_bucket:int=512,
              rebuild:bool=False,base_dedup_dir:Path|str|None=None,checkpoint_signature:Optional[str]=None)->Dict[str,Any]:
    inv=Path(inventory_dir).resolve();out=Path(out_dir).resolve();thresholds={**DEFAULT_THRESHOLDS,**(thresholds or {})}
    os.environ['SBP_LSH_BANDS']=str(int(bands))
    if rebuild and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();caps=_dedup_worker_limits(workers)
    base_dd=Path(base_dedup_dir).resolve() if base_dedup_dir else None
    exact_ckpt=out/'incremental_exact_checkpoint.json'
    resume_exact=False
    ck=None
    if base_dd is not None and exact_ckpt.is_file():
        try:ck=json.loads(exact_ckpt.read_text(encoding='utf-8'))
        except Exception as e:raise RuntimeError(f'cannot read incremental exact checkpoint: {exact_ckpt}: {e}') from e
        ck_sig=str(ck.get('checkpoint_signature') or '')
        want_sig=str(checkpoint_signature or '')
        if ck_sig==want_sig and want_sig:
            survivor_paths=sorted((out/'exact_survivors').glob('p*.mpk'));drop_parts=sorted((out/'exact_drops').glob('p*.mpk'))
            if int(ck.get('exact_partitions',-1))!=int(exact_partitions) or len(survivor_paths)!=int(exact_partitions) or len(drop_parts)!=int(exact_partitions):
                raise RuntimeError(f'incremental exact checkpoint incomplete: survivors={len(survivor_paths)} drops={len(drop_parts)} expected={exact_partitions}')
            fps=int(ck['fingerprinted']);exact_dropped=int(ck['exact_dropped']);exact_base_dropped=int(ck.get('exact_base_dropped',0));base_live_compared=int(ck.get('base_live_exact_hits',0))
            resume_exact=True
            print(f'[dedup 1-2/4 exact] RESUME checkpoint signature={want_sig} fingerprinted={fps:,} survivors={int(ck.get("survivors",0)):,} dropped={exact_dropped:,} base_dropped={exact_base_dropped:,}',flush=True)
        else:
            print(f'[dedup 1-2/4 exact] INVALIDATE stale checkpoint old_signature={ck_sig or "<missing>"} new_signature={want_sig or "<missing>"}',flush=True)
    if base_dd is not None and not resume_exact:
        # A checkpoint without the current upstream signature can never be reused:
        # stage augmentation may have added/changed packs after the checkpoint.
        # Clear only augmentation dedup state; immutable base_dedup_dir is untouched.
        for dname in ('fingerprint_runs','exact_runs','exact_survivors','exact_drops','lsh_runs','near_edges','survivors','_exact_sqlite','_near_sqlite'):
            _rmtree_any(out/dname,keep_link=dname in ('exact_runs','lsh_runs'))
        for fname in ('drop_index.sqlite','drop_index.sqlite.tmp','drop_ids.mpk','dedup_report.json','_SUCCESS.json','incremental_exact_checkpoint.json','near_edges.sqlite.tmp'):
            try:(out/fname).unlink()
            except FileNotFoundError:pass
    exact_resume_marker=out/'exact_resume.json';exact_signature=str(checkpoint_signature or '')
    exact_resume_ok=False;exact_complete=False
    if base_dd is None:
        er={}
        if exact_resume_marker.is_file():
            try:er=json.loads(exact_resume_marker.read_text(encoding='utf-8'))
            except Exception:er={}
        exact_resume_ok=(er.get('signature')==exact_signature and int(er.get('partitions',-1))==int(exact_partitions))
        if not exact_resume_ok:
            stale=[d for d in ('exact_survivors','exact_drops','exact_stats') if (out/d).is_dir() and any((out/d).glob('p*.mpk'))]
            if stale:
                print('[dedup 1-2/4 exact] INVALIDATE stale exact outputs (signature/partitions changed)',flush=True)
                for d in stale:shutil.rmtree(out/d,ignore_errors=True)
            (out/'exact_route.json').unlink(missing_ok=True)
        else:
            sps=sorted((out/'exact_survivors').glob('p*.mpk'));dps=sorted((out/'exact_drops').glob('p*.mpk'))
            if len(sps)==int(exact_partitions) and len(dps)==int(exact_partitions):
                exact_complete=True;fps=int(er.get('fingerprinted',0));survivor_paths=sps
                exact_dropped=int(er.get('exact_dropped',0)) if er.get('complete') else sum(sum(1 for _ in iter_frames(p)) for p in dps)
                if not fps:
                    rm={}
                    if (out/'exact_route.json').is_file():
                        try:rm=json.loads((out/'exact_route.json').read_text(encoding='utf-8'))
                        except Exception:rm={}
                    fps=int(rm.get('records',0))
                exact_base_dropped=0;base_live_compared=0
                print(f'[dedup 1-2/4 exact] RESUME complete exact survivors={len(sps)} dropped={exact_dropped:,} fingerprinted={fps:,}',flush=True)
            else:
                print(f'[dedup 1-2/4 exact] partial exact outputs kept survivors={len(sps)} drops={len(dps)} expected={exact_partitions}',flush=True)
    if not resume_exact and not exact_complete:
        shards=sorted(p for p in inv.rglob('*.mpk') if p.is_file())
        fpw=caps['fingerprint'];fp_root=out/'fingerprint_runs'
        sigmark=fp_root/'.resume_signature';sig=_stable_hash([(str(p.relative_to(inv)),int(p.stat().st_size)) for p in shards])
        prev_sig=sigmark.read_text(encoding='utf-8').strip() if sigmark.is_file() else None
        if prev_sig is not None and prev_sig!=sig:shutil.rmtree(fp_root,ignore_errors=True)
        fp_root.mkdir(parents=True,exist_ok=True);sigmark.write_text(sig,encoding='utf-8')
        for q in fp_root.rglob('*.tmp.*'):q.unlink(missing_ok=True)
        done=[fp_root/p.parent.name/p.name for p in shards]
        fp_runs=[d for d in done if d.is_file()];fp_tasks=[(p,fp_root,exact_partitions,signature_size,near) for p,d in zip(shards,done) if not d.is_file()]
        print(f'[dedup 1/4 fingerprint] inventory_shards={len(shards)} resume_runs={len(fp_runs)} pending={len(fp_tasks)} workers={fpw}/{workers} exact_parts={exact_partitions} streaming_runs=1/shard feature_cap={os.environ.get("SBP_NEAR_FEATURE_CAP","32768")}',flush=True)
        if fpw<=1:
            for t in fp_tasks:
                _c,rp=_fingerprint_shard(t);fp_runs.append(Path(rp))
        else:
            for _,res in _bounded_process_results(fp_tasks,_fingerprint_shard,fpw,max_in_flight=max(2,fpw),mp_context=process_context()):fp_runs.append(Path(res[1]))
        fp_runs.sort()
        exact_root=out/'exact_runs';route_marker=out/'exact_route.json';route_ok=False;fps=0
        if route_marker.is_file() and exact_root.is_dir():
            try:rm=json.loads(route_marker.read_text(encoding='utf-8'))
            except Exception:rm={}
            ef=sorted(exact_root.glob('p*.mpk'))
            if rm.get('signature')==exact_signature and int(rm.get('parts',-1))==int(exact_partitions) and len(ef)==int(exact_partitions):
                route_ok=True;fps=int(rm.get('records',0));exact_files=ef
                print(f'[dedup 1/4 fingerprint] RESUME exact routing files={len(ef)} records={fps:,}',flush=True)
        if not route_ok:
            routed=_route_fingerprint_runs(fp_runs,exact_root,exact_partitions);fps=routed
            atomic_json(route_marker,{'signature':exact_signature,'parts':int(exact_partitions),'records':int(routed),'runs':len(fp_runs)})
        print(f'[dedup 1/4 fingerprint] fingerprint COMPLETE records={fps:,} runs={len(fp_runs):,}; routed to {exact_partitions} global exact files',flush=True)
        # keep fingerprint_runs for restart-resume; reclaimed by the semantic-milestone prune
        exact_files=sorted(exact_root.glob('p*.mpk'));exw=caps['exact']
        if base_dd is not None:
            base_exact=base_dd/'exact_survivors';base_drop=base_dd/'drop_index.sqlite'
            if not base_exact.is_dir() or not base_drop.is_file():raise RuntimeError(f'incremental base dedup reference incomplete: {base_dd}')
            base_parts={_partition_from_run_name(p):p for p in base_exact.glob('p*.mpk')}
            if len(base_parts)!=int(exact_partitions) or set(base_parts)!=set(range(int(exact_partitions))):
                raise RuntimeError(f'base exact partition contract mismatch: expected {exact_partitions}, found {len(base_parts)}')
            print(f'[dedup 2/4 exact] incremental base-wins populated_aug_partitions={len(exact_files)} base_partitions={len(base_parts)} workers={exw}/{workers}',flush=True)
            exact_dropped=exact_base_dropped=base_live_compared=0;survivor_paths=[]
            tasks=[(_partition_from_run_name(p),[p],base_parts[_partition_from_run_name(p)],base_dd,out) for p in exact_files]
            if exw<=1:iterator=((t,_exact_partition_incremental(t)) for t in tasks)
            else:iterator=_bounded_process_results(tasks,_exact_partition_incremental,exw,max_in_flight=max(2,exw),mp_context=process_context())
            for _t,x in iterator:
                exact_dropped+=int(x[2]);exact_base_dropped+=int(x[3]);base_live_compared+=int(x[4]);survivor_paths.append(Path(x[5]))
        else:
            sp_root=out/'exact_survivors';dp_root=out/'exact_drops'
            exact_dropped=0;exact_base_dropped=0;base_live_compared=0
            atomic_json(exact_resume_marker,{'signature':exact_signature,'partitions':int(exact_partitions),'complete':False})
            completed={};pending=[]
            for p in exact_files:
                part=_partition_from_run_name(p);sp=sp_root/f'p{part:05d}.mpk';dp=dp_root/f'p{part:05d}.mpk'
                if exact_resume_ok and sp.is_file() and dp.is_file():completed[part]=(sp,dp)
                else:pending.append((part,p))
            for part,(sp,dp) in completed.items():
                stf=out/'exact_stats'/f'p{part:05d}.json';drops_n=None
                if stf.is_file():
                    try:drops_n=int(json.loads(stf.read_text(encoding='utf-8')).get('drops'))
                    except Exception:drops_n=None
                if drops_n is None:drops_n=sum(1 for _ in iter_frames(dp))
                exact_dropped+=drops_n
            tasks=[(part,[p],out) for part,p in pending]
            print(f'[dedup 2/4 exact] populated_partitions={len(exact_files)} resumed_parts={len(completed)} pending_parts={len(tasks)} workers={exw}/{workers} backend=sqlite-external',flush=True)
            if exw<=1:iterator=((t,_exact_partition(t)) for t in tasks)
            else:iterator=_bounded_process_results(tasks,_exact_partition,exw,max_in_flight=max(2,exw),mp_context=process_context())
            for _t,x in iterator:exact_dropped+=int(x[2])
            survivor_paths=sorted(sp_root.glob('p*.mpk'))
            if len(survivor_paths)!=int(exact_partitions):
                raise RuntimeError(f'exact resume incomplete: survivors={len(survivor_paths)} expected={exact_partitions}')
        survivor_paths.sort();_rmtree_any(exact_root,keep_link=True);print(f'[dedup 2/4 exact] COMPLETE dropped={exact_dropped:,} base_dropped={exact_base_dropped:,}',flush=True)
        if base_dd is not None:
            atomic_json(exact_ckpt,{'checkpoint_signature':str(checkpoint_signature or ''),'fingerprinted':int(fps),
                'survivors':sum(1 for sp in survivor_paths for _ in iter_frames(sp)),
                'exact_dropped':int(exact_dropped),'exact_base_dropped':int(exact_base_dropped),
                'base_live_exact_hits':int(base_live_compared),'exact_partitions':int(exact_partitions)})
        else:
            atomic_json(exact_resume_marker,{'signature':exact_signature,'partitions':int(exact_partitions),'fingerprinted':int(fps),'exact_dropped':int(exact_dropped),'complete':True})
            route_marker.unlink(missing_ok=True)
    near_stats=[];near_paths=[];near_complete=False
    near_dir=out/'near_edges';near_marker=out/'near_resume.json'
    if near:
        near_sig=str(checkpoint_signature or '')
        if base_dd is None:
            nm={}
            if near_marker.is_file():
                try:nm=json.loads(near_marker.read_text(encoding='utf-8'))
                except Exception:nm={}
            near_meta_ok=(nm.get('signature')==near_sig and int(nm.get('partitions',-1))==int(lsh_partitions) and int(nm.get('bands',-1))==int(bands))
            if near_meta_ok:
                nps=sorted(near_dir.glob('p*.mpk')) if near_dir.is_dir() else []
                expected=int(nm.get('files',-1))
                if nm.get('complete') and expected>0 and len(nps)==expected and all(p.stat().st_size>0 for p in nps):
                    near_complete=True;near_paths=nps
                    print(f'[dedup 3/4 near] RESUME complete near_edges={len(nps)}',flush=True)
                    _rmtree_any(out/'lsh_runs',keep_link=True);(out/'lsh_route.json').unlink(missing_ok=True)
            else:
                stale=(near_dir.is_dir() and any(near_dir.glob('p*.mpk'))) or near_marker.is_file()
                if stale:
                    print('[dedup 3/4 near] INVALIDATE stale near_edges (signature/partitions changed)',flush=True)
                    shutil.rmtree(near_dir,ignore_errors=True);near_marker.unlink(missing_ok=True)
                (out/'lsh_route.json').unlink(missing_ok=True)
        if not near_complete:
            print(f'[dedup 3/4 near] global streaming LSH route -> {lsh_partitions} files; verify_workers={caps["near"]}/{workers}',flush=True)
            lsh_root=out/'lsh_runs';route_ckpt=out/'incremental_near_route_checkpoint.json';lsh_route_marker=out/'lsh_route.json';lsh_meta=out/'lsh_meta.mpk'
            route_sig=_stable_hash({'checkpoint_signature':near_sig,'lsh_partitions':int(lsh_partitions),'bands':int(bands),'near':True})
            route_resumed=False;lsh_stat={};lsh_rows=0
            if base_dd is not None and route_ckpt.is_file() and lsh_root.is_dir():
                try:route_meta=json.loads(route_ckpt.read_text(encoding='utf-8'))
                except Exception:route_meta={}
                lsh_files_existing=sorted(lsh_root.glob('p*.mpk'))
                if route_meta.get('route_signature')==route_sig and int(route_meta.get('file_count',-1))==len(lsh_files_existing) and len(lsh_files_existing)>0:
                    lsh_stat={str(k):v for k,v in (route_meta.get('lsh_stat') or {}).items()}
                    lsh_rows=int(lsh_stat.get('aug_rows',0))+int(lsh_stat.get('base_rows',0))
                    route_resumed=True
                    print(f'[dedup 3/4 near] RESUME LSH route files={len(lsh_files_existing)} routed_rows={lsh_rows:,}',flush=True)
            elif base_dd is None and lsh_route_marker.is_file() and lsh_root.is_dir():
                try:lm=json.loads(lsh_route_marker.read_text(encoding='utf-8'))
                except Exception:lm={}
                lf=sorted(lsh_root.glob('p*.mpk'))
                if lm.get('signature')==route_sig and int(lm.get('files',-1))==len(lf) and len(lf)>0 and lsh_meta.is_file():
                    route_resumed=True;lsh_rows=int(lm.get('rows',0))
                    print(f'[dedup 3/4 near] RESUME LSH route files={len(lf)} routed_rows={lsh_rows:,}',flush=True)
            if not route_resumed:
                try:route_ckpt.unlink()
                except FileNotFoundError:pass
                if base_dd is not None:
                    base_survivor_paths=sorted((base_dd/'exact_survivors').glob('p*.mpk'))
                    lsh_stat=_route_lsh_incremental(base_survivor_paths,survivor_paths,base_dd,lsh_root,lsh_partitions,bands)
                    lsh_rows=int(lsh_stat.get('aug_rows',0))+int(lsh_stat.get('base_rows',0))
                    atomic_json(route_ckpt,{'route_signature':route_sig,'file_count':len(list(lsh_root.glob('p*.mpk'))),'lsh_stat':lsh_stat})
                else:
                    lsh_stat={};lsh_rows=_route_lsh_global(survivor_paths,lsh_root,lsh_partitions,bands)
                    atomic_json(lsh_route_marker,{'signature':route_sig,'files':len(list(lsh_root.glob('p*.mpk'))),'rows':int(lsh_rows)})
            if base_dd is None:
                atomic_json(near_marker,{'signature':near_sig,'partitions':int(lsh_partitions),'bands':int(bands),'complete':False})
            lsh_files=sorted(lsh_root.glob('p*.mpk'))
            near_resume=os.environ.get('SBP_NEAR_RESUME','0')=='1'
            if near_resume:
                near_dir.mkdir(parents=True,exist_ok=True)
                for _q in near_dir.glob('*.tmp.*'):_q.unlink(missing_ok=True)
                _rmtree_any(near_dir/'_near_sqlite')
                print(f"[dedup 3/4 near] SBP_NEAR_RESUME=1: keeping {len(list(near_dir.glob('p*.mpk')))} completed partitions",flush=True)
            else:
                shutil.rmtree(near_dir,ignore_errors=True);near_dir.mkdir(parents=True,exist_ok=True)
            nearw=caps['near'];near_fn=_near_partition_incremental if base_dd is not None else _near_partition
            ntasks=[]
            for p in lsh_files:
                part=_partition_from_run_name(p);dst=near_dir/f'p{part:05d}.mpk'
                if near_resume and dst.is_file():continue
                task=(part,[p],dst,thresholds,max_bucket,bands)
                ntasks.append(task if base_dd is not None else task+(str(lsh_meta),))
            t_near=time.perf_counter();last_near_done=[t_near];slow_warn_s=float(os.environ.get('SBP_NEAR_SLOW_WARN_S','300'))
            def _near_consume(res):
                now=time.perf_counter();gap=now-last_near_done[0];last_near_done[0]=now
                if gap>slow_warn_s:
                    print(f'[dedup 3/4 near] WARN: {gap:.0f}s without a partition completion (slow or oversized partition)',flush=True)
                near_stats.append(res);done=len(near_stats)
                if done==len(ntasks) or done%max(1,len(ntasks)//10)==0:
                    el=now-t_near
                    print(f'[dedup 3/4 near] partitions done={done}/{len(ntasks)} elapsed={el:.0f}s rate={done/el:.2f}/s' if el>0 else f'[dedup 3/4 near] partitions done={done}/{len(ntasks)}',flush=True)
            if nearw<=1:
                for t in ntasks:_near_consume(near_fn(t))
            else:
                for _t,res in _bounded_process_results(ntasks,near_fn,nearw,max_in_flight=max(2,nearw),mp_context=process_context()):_near_consume(res)
            near_paths=sorted(near_dir.glob('*.mpk'));_rmtree_any(lsh_root,keep_link=True);route_ckpt.unlink(missing_ok=True);lsh_route_marker.unlink(missing_ok=True)
            if base_dd is None:
                atomic_json(near_marker,{'signature':near_sig,'partitions':int(lsh_partitions),'bands':int(bands),'complete':True,'files':len(near_paths)})
            print(f"[dedup 3/4 near] COMPLETE routed_rows={lsh_rows:,} candidates={sum(x.get('candidates',0) for x in near_stats):,} verified_edges={sum(x.get('edges',0) for x in near_stats):,}",flush=True)
    else:print('[dedup 3/4 near] SKIPPED',flush=True)
    print('[dedup 4/4 finalize] external sort + disk-backed drop resolution',flush=True)
    fin=_finalize_dedup_incremental(out,survivor_paths,near_paths) if base_dd is not None else _finalize_dedup_external(out,survivor_paths,near_paths)
    report={'stage':'dedup','fingerprinted':fps,'exact_dropped':fin['exact_dropped'],'near_dropped':fin['near_dropped'],'survivors':fin['survivors'],
            'near_enabled':near,'near_candidates':sum(x.get('candidates',0) for x in near_stats),'near_edges':fin['near_edges'],
            'oversized_buckets':sum(x.get('oversized_buckets',0) for x in near_stats),'exact_partitions':exact_partitions,'lsh_partitions':lsh_partitions,
            'selection_backend':fin.get('selection_backend','sqlite_external'),'drop_index':fin['drop_index'],'workers_requested':int(workers),'worker_caps':caps,
            'fingerprint_layout':'one_run_per_inventory_shard_then_global_route','incremental_base_wins':base_dd is not None,
            'base_dedup_dir':str(base_dd) if base_dd is not None else None,'exact_base_dropped':int(fin.get('exact_base_dropped',exact_base_dropped)),
            'near_base_dropped':int(fin.get('near_base_dropped',0)),'base_live_exact_compared':int(base_live_compared),
            'near_base_edges':sum(int(x.get('base_edges',0)) for x in near_stats),'lsh_route':lsh_stat if near and base_dd is not None else {},'elapsed_s':time.perf_counter()-t0}
    atomic_json(out/'dedup_report.json',report);success_marker(out,'dedup',report);print(f"[dedup 4/4 finalize] COMPLETE survivors={sum(fin['survivors'].values()):,} total_drop={fin['drop_count']:,} elapsed={report['elapsed_s']:.1f}s",flush=True);return report


# ============================================================================
# BEGIN semantic.py
# ============================================================================
import ast, math, os, re, shutil, time, warnings
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

TREE_SITTER_GRAMMARS={
    'Python':'python','C':'c','C++':'cpp','C#':'csharp','Java':'java','JavaScript':'javascript','TypeScript':'typescript','TSX':'tsx',
    'Go':'go','Rust':'rust','Ruby':'ruby','PHP':'php','Kotlin':'kotlin','Swift':'swift','Scala':'scala','Lua':'lua','Shell':'bash','Dart':'dart',
    'Julia':'julia','R':'r','Haskell':'haskell','Elixir':'elixir','Erlang':'erlang','OCaml':'ocaml','Clojure':'clojure','Objective-C':'objc','Zig':'zig','Solidity':'solidity'
}

@dataclass(frozen=True)
class MathBlock:
    start:int; end:int; text:str; kind:str

LATEX_ENV_RE=re.compile(r"\\begin\{([A-Za-z*]+)\}.*?\\end\{\1\}",re.DOTALL)
DISPLAY_MATH_RE=re.compile(r"\$\$.*?\$\$|\\\[.*?\\\]",re.DOTALL)

def _paragraph_blocks(segment:str,base:int,kind='paragraph')->List[MathBlock]:
    out=[]
    for m in re.finditer(r"\S(?:.*?\S)?(?=(?:\s*\n\s*\n)|\s*\Z)",segment,re.DOTALL):
        txt=segment[m.start():m.end()]
        if txt.strip():out.append(MathBlock(base+m.start(),base+m.end(),txt,kind))
    return out

def split_math_blocks(text:str)->List[MathBlock]:
    structural=[]
    for rx,kind in ((LATEX_ENV_RE,'latex_env'),(DISPLAY_MATH_RE,'display_math')):
        for m in rx.finditer(text):structural.append(MathBlock(m.start(),m.end(),m.group(0),kind))
    structural.sort(key=lambda b:(b.start,-b.end));kept=[];last=-1
    for b in structural:
        if b.start<last:continue
        kept.append(b);last=b.end
    out=[];pos=0
    for b in kept:
        if pos<b.start:out.extend(_paragraph_blocks(text[pos:b.start],pos))
        out.append(b);pos=b.end
    if pos<len(text):out.extend(_paragraph_blocks(text[pos:],pos))
    return sorted(out,key=lambda b:b.start)


def _split_large_paragraph_block(b:MathBlock,max_chars:int)->List[MathBlock]:
    if b.end-b.start<=max_chars or b.kind in {'latex_env','display_math'}: return [b]
    text=b.text;out=[];pos=0
    # Prefer paragraph-internal line/sentence boundaries; raw payload is never changed.
    while pos<len(text):
        end=min(len(text),pos+max_chars)
        if end<len(text):
            cut=max(text.rfind('\n',pos+max_chars//2,end),text.rfind('. ',pos+max_chars//2,end))
            if cut>pos:end=cut+1
        out.append(MathBlock(b.start+pos,b.start+end,text[pos:end],b.kind));pos=end
    return out

def split_natural_blocks(text:str)->List[MathBlock]:
    out=[]
    for b in _paragraph_blocks(text,0):
        s=b.text.strip(); first=s.splitlines()[0] if s else ''
        kind='heading' if (first.startswith('#') or (len(first)<=120 and first.endswith(':')) or (len(first)<=80 and first.isupper() and any(c.isalpha() for c in first))) else 'paragraph'
        out.append(MathBlock(b.start,b.end,b.text,kind))
    return out

@dataclass(frozen=True)
class Span:
    start:int; end:int; kind:str; name:str=''
@dataclass(frozen=True)
class Symbol:
    name:str; qualname:str; kind:str; start:int; end:int; signature:str=''; line:int=0; column:int=0
@dataclass
class AstAnalysis:
    parse_ok:bool; backend:str; outline:str; symbols:List[Symbol]; calls:List[str]; fim_spans:List[Span]; error:str=''

def _line_starts(text):
    starts=[0]
    for m in re.finditer('\n',text):starts.append(m.end())
    return starts

class _Utf8OffsetMapper:
    """Map UTF-8 byte columns to Python character offsets without repeated prefix encode/decode.

    Python AST and tree-sitter both report UTF-8 byte columns. Most source lines are
    ASCII, where mapping is O(1). Non-ASCII lines build one compact cumulative-byte
    table lazily and reuse it for every node on that line.
    """
    __slots__=('text','starts','_cache')
    def __init__(self,text):
        self.text=text;self.starts=_line_starts(text);self._cache={}
    def char_offset(self,lineno,byte_col):
        import bisect
        idx=max(0,int(lineno)-1)
        if idx>=len(self.starts):return len(self.text)
        start=self.starts[idx];end=self.text.find('\n',start);end=len(self.text) if end<0 else end
        col=max(0,int(byte_col));line=self.text[start:end]
        if not line or col<=0:return start
        if line.isascii():return min(end,start+col)
        ends=self._cache.get(idx)
        if ends is None:
            total=0;ends=[]
            for ch in line:
                total+=len(ch.encode('utf-8'));ends.append(total)
            self._cache[idx]=ends
        return min(end,start+bisect.bisect_right(ends,col))

def _py_char_offset(text,starts,lineno,byte_col):
    # Compatibility helper for older callers/tests. New hot paths use _Utf8OffsetMapper.
    mapper=_Utf8OffsetMapper(text);mapper.starts=starts
    return mapper.char_offset(lineno,byte_col)

def _python_analysis(text:str)->AstAnalysis:
    """Recursion-safe Python AST analysis.

    Python's ast.NodeVisitor is recursive and can overflow on valid deep ASTs
    (e.g. long BinOp/attribute chains). This walker carries lexical scope in an
    explicit stack, so traversal depth is bounded by heap memory, not the Python
    recursion limit. SyntaxWarning is analysis noise only and is captured here;
    the original source text is never modified.
    """
    mapper=_Utf8OffsetMapper(text)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SyntaxWarning)
            tree=ast.parse(text, filename='<source>', mode='exec')
    except (SyntaxError, RecursionError, MemoryError) as e:
        return AstAnalysis(False,'python_ast','',[],[],[],f'{type(e).__name__}: {e}')

    def node_span(node):
        a=mapper.char_offset(getattr(node,'lineno',1),getattr(node,'col_offset',0))
        b=mapper.char_offset(getattr(node,'end_lineno',getattr(node,'lineno',1)),getattr(node,'end_col_offset',0))
        return a,max(a+1,b)

    symbols=[];calls=[];spans=[];outline=[]
    work=[(tree,())]
    while work:
        node,scope=work.pop(); child_scope=scope
        if isinstance(node,ast.ClassDef):
            a,b=node_span(node);q='.'.join((*scope,node.name));le=text.find('\n',a);sig=text[a:(le if le>=0 else b)].strip()
            symbols.append(Symbol(node.name,q,'class',a,b,sig,getattr(node,'lineno',0),getattr(node,'col_offset',0)))
            spans.append(Span(a,b,'class',node.name));outline.append('  '*len(scope)+'class '+node.name);child_scope=(*scope,node.name)
        elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            a,b=node_span(node);q='.'.join((*scope,node.name));le=text.find('\n',a);sig=text[a:(le if le>=0 else b)].strip();kind='method' if scope else 'function'
            symbols.append(Symbol(node.name,q,kind,a,b,sig,getattr(node,'lineno',0),getattr(node,'col_offset',0)))
            spans.append(Span(a,b,kind,node.name));outline.append('  '*len(scope)+'def '+node.name)
            for st in node.body:
                sa,sb=node_span(st)
                if sb-sa>=8:spans.append(Span(sa,sb,'statement',node.name))
            child_scope=(*scope,node.name)
        if isinstance(node,ast.Call):
            fn=node.func;name=fn.id if isinstance(fn,ast.Name) else fn.attr if isinstance(fn,ast.Attribute) else None
            if name:calls.append(name)
        children=list(ast.iter_child_nodes(node))
        for child in reversed(children):work.append((child,child_scope))
    return AstAnalysis(True,'python_ast','\n'.join(outline[:256]),symbols,sorted(set(calls)),spans,'')

GENERIC_DEF_PATTERNS=[
 ('class',re.compile(r"(?m)^\s*(?:public\s+|private\s+|protected\s+|internal\s+|export\s+|abstract\s+|sealed\s+|open\s+)*(?:class|struct|interface|trait|enum|record|module)\s+([A-Za-z_][A-Za-z0-9_]*)")),
 ('function',re.compile(r"(?m)^\s*(?:pub\s+|public\s+|private\s+|protected\s+|static\s+|async\s+|export\s+|local\s+)*(?:fn|func|function|def|fun|sub|proc)\s+([A-Za-z_][A-Za-z0-9_!?]*)")),
 ('function',re.compile(r"(?m)^[ \t]*(?:[A-Za-z_~][A-Za-z0-9_:<>,\[\]*&?]*[ \t]+){1,8}([A-Za-z_~][A-Za-z0-9_]*)[ \t]*\([^;\n{}]{0,500}\)[ \t]*(?:\{|=>)")),
]
CALL_RE=re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(");CALL_STOP={'if','for','while','switch','catch','return','sizeof','typeof','new','fn','func','function','def'}

def _brace_end(text,start):
    b=text.find('{',start)
    if b<0:
        le=text.find('\n',start);return len(text) if le<0 else le+1
    depth=0;quote=None;esc=False
    for i in range(b,len(text)):
        c=text[i]
        if quote:
            if esc:esc=False
            elif c=='\\':esc=True
            elif c==quote:quote=None
            continue
        if c in {'"',"'"}:quote=c;continue
        if c=='{':depth+=1
        elif c=='}':
            depth-=1
            if depth==0:return i+1
    return len(text)

def _regex_analysis(language,text,error=''):
    symbols=[];spans=[];outline=[];seen=set()
    for kind,rx in GENERIC_DEF_PATTERNS:
        for m in rx.finditer(text):
            name=m.group(1)
            if (m.start(),name) in seen:continue
            seen.add((m.start(),name));end=_brace_end(text,m.start());le=text.find('\n',m.start());sig=text[m.start():(le if le>=0 else end)].strip();line=text.count('\n',0,m.start())+1
            symbols.append(Symbol(name,name,kind,m.start(),end,sig,line,0));spans.append(Span(m.start(),end,kind,name));outline.append(f'{kind} {name}')
    calls=sorted({m.group(1) for m in CALL_RE.finditer(text) if m.group(1) not in CALL_STOP})
    return AstAnalysis(False,'regex','\n'.join(outline[:256]),symbols,calls,spans,error)

TS_DEF_TYPES={
    'function_definition':'function','function_declaration':'function','method_definition':'method','method_declaration':'method',
    'function_item':'function','class_definition':'class','class_declaration':'class','class_specifier':'class',
    'struct_item':'struct','struct_specifier':'struct','struct_declaration':'struct','interface_declaration':'interface',
    'trait_item':'trait','trait_declaration':'trait','enum_item':'enum','enum_declaration':'enum','enum_specifier':'enum',
    'constructor_declaration':'constructor','constructor_definition':'constructor','record_declaration':'record',
    'protocol_declaration':'interface','contract_declaration':'contract','library_declaration':'module',
    'implementation_definition':'implementation','impl_item':'implementation','module_definition':'module','module_declaration':'module',
    'object_declaration':'object','object_definition':'object','singleton_method':'method','method':'method',
    'function_clause':'function','function_signature':'function','short_function_definition':'function',
    'struct_definition':'struct','type_declaration':'type','type_definition':'type','let_binding':'function',
    'value_definition':'function','modifier_definition':'modifier','event_definition':'event',
}

TS_DEF_TYPES_BY_LANGUAGE={
    'C#':{'constructor_declaration':'constructor','property_declaration':'property','record_declaration':'record'},
    'Java':{'constructor_declaration':'constructor','record_declaration':'record','annotation_type_declaration':'interface'},
    'Go':{'method_declaration':'method','function_declaration':'function','type_declaration':'type','type_spec':'type'},
    'Rust':{'impl_item':'implementation','function_item':'function','struct_item':'struct','trait_item':'trait','enum_item':'enum','mod_item':'module','type_item':'type'},
    'Ruby':{'method':'method','singleton_method':'method','class':'class','module':'module'},
    'PHP':{'method_declaration':'method','function_definition':'function','class_declaration':'class','interface_declaration':'interface','trait_declaration':'trait','enum_declaration':'enum'},
    'Kotlin':{'function_declaration':'function','class_declaration':'class','object_declaration':'object','secondary_constructor':'constructor'},
    'Swift':{'function_declaration':'function','class_declaration':'class','struct_declaration':'struct','protocol_declaration':'interface','enum_declaration':'enum','extension_declaration':'extension','init_declaration':'constructor'},
    'Scala':{'function_definition':'function','function_declaration':'function','class_definition':'class','object_definition':'object','trait_definition':'trait','enum_definition':'enum'},
    'Lua':{'function_declaration':'function','function_definition':'function'},
    'Shell':{'function_definition':'function'},
    'Dart':{'function_signature':'function','method_signature':'method','class_definition':'class','mixin_declaration':'trait','extension_declaration':'extension','constructor_signature':'constructor'},
    'Julia':{'function_definition':'function','short_function_definition':'function','struct_definition':'struct','module_definition':'module','macro_definition':'macro'},
    'Haskell':{'function':'function','signature':'function','data_type':'type','newtype':'type','type_synomym':'type','class':'class','instance':'implementation'},
    'Erlang':{'function_clause':'function','function':'function','type_spec':'type'},
    'OCaml':{'let_binding':'function','value_definition':'function','module_definition':'module','type_definition':'type','class_definition':'class'},
    'Objective-C':{'method_definition':'method','method_declaration':'method','interface_declaration':'interface','implementation_definition':'implementation'},
    'Zig':{'function_declaration':'function','struct_declaration':'struct','test_declaration':'test'},
    'Solidity':{'contract_declaration':'contract','function_definition':'function','modifier_definition':'modifier','struct_declaration':'struct','enum_declaration':'enum','event_definition':'event'},
}

TS_CALL_TYPES={'call_expression','call','invocation_expression','function_call_expression','method_invocation','command','macro_invocation'}
TS_FIM_TYPES={
    'expression_statement','return_statement','assignment','assignment_expression','augmented_assignment_expression',
    'if_statement','for_statement','for_in_statement','while_statement','switch_statement','match_expression','match_statement',
    'try_statement','throw_statement','yield_statement','lexical_declaration','variable_declaration','declaration',
    'compound_statement','statement_block','block','do_block','case_clause','when_clause'
}
TS_CLASSLIKE_KINDS={'class','struct','interface','trait','record','contract','module','object','implementation','extension'}


def _ts_def_kind(language,typ):
    lang=canonical_language(language) or language
    return TS_DEF_TYPES_BY_LANGUAGE.get(lang,{}).get(str(typ),TS_DEF_TYPES.get(str(typ)))


def _ts_special_definition(language,node,raw):
    lang=canonical_language(language) or language;typ=str(getattr(node,'type',''))
    try:snippet=raw[int(node.start_byte):min(int(node.end_byte),int(node.start_byte)+1024)].decode('utf-8','ignore')
    except Exception:return None,''
    if lang=='Elixir' and typ in {'call','stab_clause'}:
        m=re.match(r'\s*defmodule\s+([A-Z][A-Za-z0-9_.]*)',snippet)
        if m:return 'module',m.group(1)
        m=re.match(r'\s*defmacro(?:p)?\s+([A-Za-z_][A-Za-z0-9_!?]*)',snippet)
        if m:return 'macro',m.group(1)
        m=re.match(r'\s*def(?:p)?\s+([A-Za-z_][A-Za-z0-9_!?]*)',snippet)
        if m:return 'function',m.group(1)
    if lang=='Clojure' and typ in {'list_lit','list','list_literal'}:
        m=re.match(r'\s*\(\s*(defn-|defn|defmacro|defrecord|deftype)\s+([^\s\)\[]+)',snippet)
        if m:
            kind={'defn':'function','defn-':'function','defmacro':'macro','defrecord':'record','deftype':'type'}[m.group(1)]
            return kind,m.group(2)
    if lang=='R' and typ in {'binary_operator','left_assignment','assignment'}:
        m=re.match(r'\s*([A-Za-z.][A-Za-z0-9._]*)\s*(?:<-|=)\s*function\s*\(',snippet)
        if m:return 'function',m.group(1)
    if lang in {'JavaScript','TypeScript','TSX'} and typ in {'variable_declarator','variable_declaration'}:
        try:name_node=node.child_by_field_name('name');value_node=node.child_by_field_name('value')
        except Exception:name_node=value_node=None
        if name_node is not None and value_node is not None and str(getattr(value_node,'type','')) in {'arrow_function','function','function_expression'}:
            try:name=raw[int(name_node.start_byte):int(name_node.end_byte)].decode('utf-8','ignore').strip()
            except Exception:name=''
            if name:return 'function',name
    return None,''


def _point_parts(point):
    if isinstance(point,(tuple,list)):
        return int(point[0]),int(point[1])
    return int(getattr(point,'row',0)),int(getattr(point,'column',0))


def _ts_node_name(language,node,raw):
    nn=None
    try: nn=node.child_by_field_name('name')
    except Exception: nn=None
    if nn is not None:
        try:
            val=raw[int(nn.start_byte):int(nn.end_byte)].decode('utf-8','ignore').strip()
            if val:return val
        except Exception:pass
    try: snippet=raw[int(node.start_byte):min(int(node.end_byte),int(node.start_byte)+512)].decode('utf-8','ignore')
    except Exception:return ''
    lang=canonical_language(language) or language
    pats=[]
    if lang in {'C','C++','C#','Java','JavaScript','TypeScript','TSX','Go','Rust','PHP','Kotlin','Swift','Scala','Dart','Objective-C','Zig','Solidity'}:
        pats=[r'\b(?:class|struct|interface|trait|enum|record|contract|protocol|object|fn|func|function|def|fun)\s+([A-Za-z_][A-Za-z0-9_]*)',
              r'\b([A-Za-z_~][A-Za-z0-9_]*)\s*\(']
    elif lang in {'Ruby','Elixir','Julia','Lua'}:pats=[r'\b(?:def|defp|defmodule|function|macro|module|struct)\s+([A-Za-z_][A-Za-z0-9_!?\.]*)']
    elif lang=='Erlang':pats=[r'^\s*([a-z][A-Za-z0-9_]*)\s*\(']
    elif lang in {'OCaml','Haskell'}:pats=[r'\b(?:let|module|type|class|data|newtype)\s+([A-Za-z_][A-Za-z0-9_\']*)']
    elif lang=='Clojure':pats=[r'\((?:defn|defn-|defmacro|defrecord|deftype)\s+([^\s\)\[]+)']
    for pat in pats:
        m=re.search(pat,snippet,re.MULTILINE)
        if m:return m.group(1)
    return ''


def _analyze_ts_root(language,text,raw,root,max_nodes=250000):
    if not hasattr(root,'type') or not hasattr(root,'children'):
        return AstAnalysis(False,'tree_sitter','',[],[],[],f'incompatible tree_sitter Node API: {type(root)!r}')
    symbols=[];calls=[];spans=[];outline=[];mapper=_Utf8OffsetMapper(text);stack=[(root,())];visited=0;partial=False
    max_fim=max(8,int(len(text)*0.70))
    while stack:
        node,scope=stack.pop();visited+=1
        if max_nodes and visited>max_nodes:
            partial=True;break
        typ=str(node.type);kind=_ts_def_kind(language,typ);special_kind,special_name=(None,'') if kind else _ts_special_definition(language,node,raw);kind=kind or special_kind;is_fim=typ in TS_FIM_TYPES;is_call=typ in TS_CALL_TYPES
        child_scope=scope
        if kind or is_fim:
            sr,sc=_point_parts(getattr(node,'start_point',(0,0)));er,ec=_point_parts(getattr(node,'end_point',(0,0)))
            a=mapper.char_offset(sr+1,sc);b=mapper.char_offset(er+1,ec);b=max(a+1,b)
            if kind:
                name=special_name or _ts_node_name(language,node,raw) or f'<anonymous@{sr+1}>'
                q='.'.join((*scope,name)) if name else '.'.join(scope);le=text.find('\n',a);sig=text[a:(le if le>=0 else min(b,a+512))].strip()
                symbols.append(Symbol(name,q,kind,a,b,sig,sr+1,sc));spans.append(Span(a,b,kind,name));outline.append('  '*len(scope)+f'{kind} {name}')
                if kind in TS_CLASSLIKE_KINDS or kind in {'function','method','constructor'}:child_scope=(*scope,name)
            if is_fim and 8<=b-a<=max_fim:spans.append(Span(a,b,typ,''))
        if is_call:
            fn=None
            for field in ('function','name','method'):
                try: fn=node.child_by_field_name(field)
                except Exception: fn=None
                if fn is not None:break
            if fn is not None:
                try:val=raw[int(fn.start_byte):int(fn.end_byte)].decode('utf-8','ignore').split('.')[-1].split('::')[-1].strip()
                except Exception:val=''
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',val):calls.append(val)
        try:children=getattr(node,'children',None) or []
        except Exception:children=[]
        for child in reversed(children):stack.append((child,child_scope))
    try:root_error=bool(getattr(root,'has_error',False))
    except Exception:root_error=True
    ok=not root_error and not partial
    err=[]
    if root_error:err.append('tree_has_error')
    if partial:err.append(f'node_cap:{max_nodes}')
    return AstAnalysis(ok,'tree_sitter','\n'.join(outline[:256]),symbols,sorted(set(calls)),spans,';'.join(err))


class StructuralAnalyzer:
    """Direct parser used only inside an isolated native-parser subprocess."""
    def __init__(self,disable_tree_sitter=False,max_parse_bytes=2<<20,parser_cache_size=8,max_nodes=250000):
        self.get_language=None;self.Parser=None;self.parsers=OrderedDict();self.ts_backend='off';self.ts_error='';self.max_parse_bytes=max(0,int(max_parse_bytes or 0));self.parser_cache_size=max(1,int(parser_cache_size));self.max_nodes=max(1000,int(max_nodes))
        if not disable_tree_sitter:
            try:
                import tree_sitter
                from tree_sitter_language_pack import get_language
                self.Parser=tree_sitter.Parser;self.get_language=get_language;self.ts_backend='standalone'
            except Exception as e:self.ts_error=f'{type(e).__name__}: {e}'
    def _parser(self,grammar):
        p=self.parsers.pop(grammar,None)
        if p is not None:
            self.parsers[grammar]=p;return p
        lang=self.get_language(grammar)
        try:p=self.Parser(lang)
        except TypeError:
            p=self.Parser()
            if hasattr(p,'language'):p.language=lang
            elif hasattr(p,'set_language'):p.set_language(lang)
            else:raise RuntimeError('installed tree_sitter.Parser has no language setter')
        self.parsers[grammar]=p
        while len(self.parsers)>self.parser_cache_size:self.parsers.popitem(last=False)
        return p
    def _ts(self,language,text):
        if not self.get_language or not self.Parser:return None
        grammar=TREE_SITTER_GRAMMARS.get(language)
        if not grammar:return None
        try:
            parser=self._parser(grammar);raw=text.encode('utf-8');tree=parser.parse(raw);root=tree.root_node
            if callable(root):root=root()
            return _analyze_ts_root(language,text,raw,root,max_nodes=self.max_nodes)
        except Exception as e:return AstAnalysis(False,'tree_sitter','',[],[],[],f'{type(e).__name__}: {e}')
    def analyze(self,language,text,source_bytes=None):
        language=canonical_language(language) or language;nbytes=int(source_bytes) if source_bytes is not None else len(text.encode('utf-8'))
        if self.max_parse_bytes and nbytes>self.max_parse_bytes:return _regex_analysis(language,text,f'parse_size_cap:{nbytes}>{self.max_parse_bytes}')
        py_error=''
        if language=='Python':
            r=_python_analysis(text)
            if r.parse_ok:return r
            py_error=r.error or ''
        ts=self._ts(language,text)
        if ts is not None and (ts.parse_ok or ts.symbols or ts.fim_spans):return ts
        err=(ts.error if ts else '') or py_error or self.ts_error or 'tree_sitter_unavailable'
        return _regex_analysis(language,text,err)


def _ts_server_main(conn,max_parse_bytes,parser_cache_size,max_nodes):
    # Native grammar code lives only in this child. A SIGSEGV kills this child, not the code-index worker.
    analyzer=StructuralAnalyzer(False,max_parse_bytes=max_parse_bytes,parser_cache_size=parser_cache_size,max_nodes=max_nodes)
    try:
        while True:
            try:req=conn.recv()
            except EOFError:break
            if req is None:break
            rid,language,text,source_bytes=req
            try:ans=analyzer.analyze(language,text,source_bytes=source_bytes)
            except BaseException as e:ans=AstAnalysis(False,'tree_sitter','',[],[],[],f'{type(e).__name__}: {e}')
            conn.send((rid,ans))
    finally:
        try:conn.close()
        except Exception:pass


def _lang_slug(language):
    raw=str(language);base=re.sub(r'[^A-Za-z0-9_.-]+','_',raw).strip('_') or 'unknown';tag=hashlib.blake2b(raw.encode('utf-8'),digest_size=4).hexdigest();return f'{base}-{tag}'


class IsolatedTSClient:
    """Crash-isolated, persistent Tree-sitter RPC client.

    Python AST remains in the worker. Non-Python Tree-sitter runs in one persistent spawned child per code worker.
    Native crashes/timeouts therefore degrade only the current language/file instead of breaking ProcessPoolExecutor.
    """
    def __init__(self,quarantine_dir,max_parse_bytes=2<<20,timeout_s=15.0,crash_threshold=2,parser_cache_size=8,max_nodes=250000,disable_tree_sitter=False):
        self.qdir=Path(quarantine_dir);self.qdir.mkdir(parents=True,exist_ok=True);self.max_parse_bytes=int(max_parse_bytes);self.timeout_s=float(timeout_s);self.crash_threshold=max(1,int(crash_threshold));self.parser_cache_size=max(1,int(parser_cache_size));self.max_nodes=max(1000,int(max_nodes));self.disable_tree_sitter=bool(disable_tree_sitter);self.proc=None;self.conn=None;self.seq=0
        self.ctx=mp.get_context(os.environ.get('SBP_TS_START_METHOD','spawn'))
    def _flag(self,language):return self.qdir/f'{_lang_slug(language)}.disabled.json'
    def _log(self,language):return self.qdir/f'{_lang_slug(language)}.crashes.log'
    def is_quarantined(self,language):return self._flag(language).exists()
    def _record_failure(self,language,reason):
        lp=self._log(language);line=json.dumps({'pid':os.getpid(),'time':time.time(),'reason':reason},ensure_ascii=False)+'\n'
        fd=os.open(lp,os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o644)
        try:os.write(fd,line.encode('utf-8'))
        finally:os.close(fd)
        try:n=sum(1 for _ in lp.open('r',encoding='utf-8',errors='ignore'))
        except Exception:n=1
        if n>=self.crash_threshold and not self._flag(language).exists():
            atomic_json(self._flag(language),{'language':language,'failures':n,'reason':reason,'disabled_at':time.time()})
        return n
    def _stop_server(self,kill=False):
        conn,proc=self.conn,self.proc;self.conn=None;self.proc=None
        if conn is not None:
            if not kill:
                try:conn.send(None)
                except Exception:pass
            try:conn.close()
            except Exception:pass
        if proc is not None:
            try:
                proc.join(timeout=.5)
                if proc.is_alive():proc.terminate();proc.join(timeout=.5)
                if proc.is_alive():proc.kill();proc.join(timeout=.5)
            except Exception:pass
    def _ensure_server(self):
        if self.proc is not None and self.proc.is_alive() and self.conn is not None:return
        self._stop_server(kill=True);parent,child=self.ctx.Pipe(duplex=True)
        proc=self.ctx.Process(target=_ts_server_main,args=(child,self.max_parse_bytes,self.parser_cache_size,self.max_nodes),daemon=True)
        proc.start();child.close();self.proc=proc;self.conn=parent
    def analyze(self,language,text,source_bytes=None):
        language=canonical_language(language) or language;nbytes=int(source_bytes) if source_bytes is not None else len(text.encode('utf-8'))
        if self.max_parse_bytes and nbytes>self.max_parse_bytes:return _regex_analysis(language,text,f'parse_size_cap:{nbytes}>{self.max_parse_bytes}')
        py_error=''
        if language=='Python':
            r=_python_analysis(text)
            if r.parse_ok:return r
            py_error=r.error or 'python_ast_failed'
        if self.disable_tree_sitter or self.is_quarantined(language):return _regex_analysis(language,text,('tree_sitter_quarantined' if self.is_quarantined(language) else 'tree_sitter_disabled')+(';' + py_error if py_error else ''))
        self.seq+=1;rid=(os.getpid()<<32)^self.seq
        try:
            self._ensure_server();self.conn.send((rid,language,text,nbytes))
            if not self.conn.poll(self.timeout_s):
                self._record_failure(language,f'timeout>{self.timeout_s}s');self._stop_server(kill=True);return _regex_analysis(language,text,'tree_sitter_timeout')
            got,ans=self.conn.recv()
            if got!=rid:raise RuntimeError(f'tree_sitter rpc id mismatch {got}!={rid}')
            if ans is not None and (ans.parse_ok or ans.symbols or ans.fim_spans):return ans
            err=ans.error if ans is not None else 'tree_sitter_unavailable'
            if py_error:err=py_error+';'+err
            return _regex_analysis(language,text,err)
        except (EOFError,BrokenPipeError,ConnectionResetError,OSError) as e:
            exitcode=self.proc.exitcode if self.proc is not None else None;reason=f'native_exit:{exitcode}:{type(e).__name__}'
            self._record_failure(language,reason);self._stop_server(kill=True);return _regex_analysis(language,text,reason)
        except Exception as e:
            self._stop_server(kill=True);return _regex_analysis(language,text,f'tree_sitter_rpc:{type(e).__name__}:{e}')
    def close(self):self._stop_server(kill=False)


_WORKER_TS_CLIENTS={}
def _get_worker_ts_client(out_root,disable_tree_sitter,max_parse_bytes,timeout_s,crash_threshold,parser_cache_size,max_nodes):
    key=(str(Path(out_root).resolve()),bool(disable_tree_sitter),int(max_parse_bytes),float(timeout_s),int(crash_threshold),int(parser_cache_size),int(max_nodes))
    c=_WORKER_TS_CLIENTS.get(key)
    if c is None:
        c=IsolatedTSClient(Path(out_root)/'ts_quarantine',max_parse_bytes=max_parse_bytes,timeout_s=timeout_s,crash_threshold=crash_threshold,parser_cache_size=parser_cache_size,max_nodes=max_nodes,disable_tree_sitter=disable_tree_sitter);_WORKER_TS_CLIENTS[key]=c
    return c


_WORKER_DIRECT_ANALYZERS={}
def _get_worker_direct_analyzer(disable_tree_sitter,max_parse_bytes,parser_cache_size,max_nodes):
    key=(bool(disable_tree_sitter),int(max_parse_bytes),int(parser_cache_size),int(max_nodes))
    a=_WORKER_DIRECT_ANALYZERS.get(key)
    if a is None:
        a=StructuralAnalyzer(disable_tree_sitter,max_parse_bytes=max_parse_bytes,parser_cache_size=parser_cache_size,max_nodes=max_nodes);_WORKER_DIRECT_ANALYZERS[key]=a
    return a

def _code_primary_context(disable_tree_sitter=False):
    # Native parser workers must be spawned fresh; the original crash trace came from popen_fork.py.
    return process_context() if disable_tree_sitter else mp.get_context('spawn')


def _structure_confidence(a):
    if a is None:return 0.0
    if a.backend=='python_ast' and a.parse_ok:return 1.0
    if a.backend=='tree_sitter':
        if a.parse_ok:return 0.95
        if a.symbols or a.fim_spans:return 0.78
        return 0.45
    if a.backend=='regex':return 0.35 if (a.symbols or a.fim_spans) else 0.20
    return 0.25

def extract_import_hints(text,language):
    lang=canonical_language(language) or language;patterns=[]
    if lang=='Python':patterns=[re.compile(r'(?m)^\s*from\s+([\.\w]+)\s+import\b'),re.compile(r'(?m)^\s*import\s+([\w\.]+)')]
    elif lang in {'JavaScript','TypeScript','TSX'}:patterns=[re.compile(r"\b(?:import|export)\b[^\n]*?\bfrom\s*['\"]([^'\"]+)['\"]"),re.compile(r"\brequire\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),re.compile(r"\bimport\s*\(\s*['\"]([^'\"]+)['\"]\s*\)")]
    elif lang in {'C','C++','Objective-C'}:patterns=[re.compile(r'(?m)^\s*#\s*(?:include|import)\s*[<\"]([^>\"]+)[>\"]')]
    elif lang in {'Java','Kotlin','Scala'}:patterns=[re.compile(r'(?m)^\s*import\s+([A-Za-z_][\w\.]*)(?:\.\*)?\s*;?')]
    elif lang=='C#':patterns=[re.compile(r'(?m)^\s*using\s+(?:static\s+)?([A-Za-z_][\w\.]*)\s*;')]
    elif lang=='Go':patterns=[re.compile(r'(?m)^\s*(?:[A-Za-z_][\w]*\s+)?"([^"]+)"')]
    elif lang=='Rust':patterns=[re.compile(r'(?m)^\s*use\s+([^;]+);'),re.compile(r'(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][\w]*)\s*;')]
    elif lang=='Ruby':patterns=[re.compile(r"(?m)^\s*require_relative\s*['\"]([^'\"]+)['\"]"),re.compile(r"(?m)^\s*require\s*['\"]([^'\"]+)['\"]")]
    elif lang=='PHP':patterns=[re.compile(r"(?m)^\s*(?:require|require_once|include|include_once)\s*\(?\s*['\"]([^'\"]+)['\"]"),re.compile(r'(?m)^\s*use\s+([A-Za-z_\\][\w\\]+)')]
    elif lang=='Shell':patterns=[re.compile(r"(?m)^\s*(?:source|\.)\s+['\"]?([^\s'\"]+)")]
    elif lang=='Dart':patterns=[re.compile(r"(?m)^\s*(?:import|export|part)\s+['\"]([^'\"]+)['\"]")]
    elif lang=='Swift':patterns=[re.compile(r'(?m)^\s*(?:@testable\s+)?import\s+(?:(?:class|struct|enum|protocol|func|var|let|typealias)\s+)?([A-Za-z_][\w\.]*)')]
    elif lang=='Julia':patterns=[re.compile(r"\binclude\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),re.compile(r'(?m)^\s*(?:using|import)\s+([A-Za-z_][\w\.]*)')]
    elif lang=='R':patterns=[re.compile(r"\bsource\s*\(\s*['\"]([^'\"]+)['\"]")]
    elif lang=='Haskell':patterns=[re.compile(r'(?m)^\s*import\s+(?:qualified\s+)?([A-Z][\w\.]*)')]
    elif lang=='Elixir':patterns=[re.compile(r'(?m)^\s*(?:alias|import|require|use)\s+([A-Z][\w\.]*)')]
    elif lang=='Erlang':patterns=[re.compile(r'(?m)^\s*-include(?:_lib)?\(\s*"([^"]+)"\s*\)'),re.compile(r'(?m)^\s*-behaviou?r\(\s*([a-z][\w]*)\s*\)')]
    elif lang=='OCaml':patterns=[re.compile(r'(?m)^\s*(?:open|include)\s+([A-Z][\w\.]*)')]
    elif lang=='Clojure':patterns=[re.compile(r'\[([a-zA-Z_][\w\.\-]+)\s+(?::as|:refer|:rename)')]
    elif lang=='Lua':patterns=[re.compile(r"\brequire\s*\(?\s*['\"]([^'\"]+)['\"]\s*\)?")]
    elif lang=='Zig':patterns=[re.compile(r'@import\s*\(\s*"([^"]+)"\s*\)')]
    elif lang=='Solidity':patterns=[re.compile(r"(?m)^\s*import\s+(?:[^;]*?\s+from\s+)?['\"]([^'\"]+)['\"]\s*;")]
    out=[]
    for rx in patterns:
        for m in rx.finditer(text):
            v=m.group(1).strip();v=v.split('::{',1)[0].strip() if lang=='Rust' else v
            if v and v not in out:out.append(v)
    return out

def _norm_path(p):return str(PurePosixPath(str(p).replace('\\','/').lstrip('/')))
def _exts(lang):return {'Python':('.py',),'JavaScript':('.js','.jsx','.mjs','.cjs'),'TypeScript':('.ts','.tsx'),'TSX':('.tsx','.ts'),'C':('.h','.c'),'C++':('.h','.hpp','.hh','.cc','.cpp','.cxx'),'Objective-C':('.h','.m','.mm'),'Java':('.java',),'Kotlin':('.kt','.kts'),'Scala':('.scala',),'C#':('.cs',),'Go':('.go',),'Rust':('.rs',),'Ruby':('.rb',),'PHP':('.php',),'Shell':('.sh',),'Dart':('.dart',),'Julia':('.jl',),'R':('.R','.r'),'Haskell':('.hs','.lhs'),'Elixir':('.ex','.exs'),'Erlang':('.erl','.hrl'),'OCaml':('.ml','.mli'),'Clojure':('.clj','.cljs','.cljc'),'Lua':('.lua',),'Zig':('.zig',),'Solidity':('.sol',),'Swift':('.swift',)}.get(lang,())

class RepoPathIndex:
    def __init__(self,paths):
        self.paths=tuple(_norm_path(x) for x in paths if x);self.pathset=set(self.paths);self.by_name=defaultdict(list);self.by_stem=defaultdict(list)
        for p in self.paths:
            pp=PurePosixPath(p);self.by_name[pp.name].append(p);self.by_stem[pp.stem].append(p)
    def resolve(self,current,lang,hint):
        cur=_norm_path(current);parent=str(PurePosixPath(cur).parent);hint=hint.strip().strip('\"\'');cands=[]
        def add(base):
            base=_norm_path(base);cands.append(base)
            for e in _exts(lang):cands.append(base if base.endswith(e) else base+e)
            if lang=='Python':cands.append(base+'/__init__.py')
            if lang in {'JavaScript','TypeScript','TSX'}:cands.extend([base+'/index.ts',base+'/index.tsx',base+'/index.js',base+'/index.jsx'])
            if lang=='Rust':cands.append(base+'/mod.rs')
        if lang=='Python' and hint.startswith('.'):
            dots=len(hint)-len(hint.lstrip('.'));mod=hint[dots:];base=PurePosixPath(parent)
            for _ in range(max(0,dots-1)):base=base.parent
            add(str(base/mod.replace('.','/')))
        elif hint.startswith('.'):add(str(PurePosixPath(parent)/hint))
        elif lang in {'Java','Kotlin','Scala','C#','Haskell','Go'}:add(hint.replace('.','/'))
        elif lang=='Rust':add(str(PurePosixPath(parent)/re.sub(r'^(?:crate|self)::','',hint).replace('::','/')))
        elif lang in {'Lua','Clojure'}:add(hint.replace('.','/').replace('-','_'))
        else:add(str(PurePosixPath(parent)/hint));add(hint)
        for c in cands:
            c=_norm_path(c)
            if c in self.pathset and c!=cur:return c
        base=PurePosixPath(hint.replace('.','/')).name;matches=set(self.by_name.get(base,[]))|set(self.by_stem.get(base,[]));matches.discard(cur)
        return next(iter(matches)) if len(matches)==1 else None

def is_test_path(path):
    p='/'+str(path).replace('\\','/').lower().lstrip('/');base=PurePosixPath(p).name
    return any(x in p for x in ('/test/','/tests/','/__tests__/','/spec/','/specs/')) or bool(re.search(r'(?:^|[._-])(test|spec)(?:[._-]|$)',base)) or base.startswith('test_')

def _file_quality(text,a,is_test=False,is_meta=False):
    if not text.strip():return 0.0
    q=0.35+0.25*_structure_confidence(a)+0.15*min(1,len(text)/2000)+0.1*min(1,len(a.symbols)/2 if a else 0)+0.1*(1 if is_test else 0)+0.05*(1 if is_meta else 0)
    if '\x00' in text:q-=0.5
    return max(0,min(1,q))

def _semantic_score(a,imports,resolved,incoming,outgoing,is_test):
    return max(0,min(1,0.25*_structure_confidence(a)+0.2*min(1,len(a.symbols)/5)+0.15*(resolved/max(1,imports) if imports else .3)+0.2*min(1,(incoming+outgoing)/6)+0.2*(1 if is_test else 0)))

def _semantic_stem(path):
    name=PurePosixPath(str(path)).name.lower()
    stem=PurePosixPath(name).stem
    for suff in ('.test','.spec','_test','_tests','-test','-tests','_spec','-spec'):
        if stem.endswith(suff):stem=stem[:-len(suff)]
    for pref in ('test_','tests_','spec_'):
        if stem.startswith(pref):stem=stem[len(pref):]
    return stem

def _same_stem_relation_edges(by_path,block_by_file):
    groups=defaultdict(list)
    for fp in by_path:
        if block_by_file.get(fp):groups[_semantic_stem(fp)].append(fp)
    edges=[]
    headers={'.h','.hh','.hpp','.hxx'};sources={'.c','.cc','.cpp','.cxx','.m','.mm'}
    for stem,paths in groups.items():
        if not stem or len(paths)<2:continue
        tests=[p for p in paths if by_path[p].get('is_test') or is_test_path(p)];impl=[p for p in paths if p not in tests]
        for tp in tests:
            for dst in sorted(impl,key=lambda x:(0 if PurePosixPath(x).suffix.lower() in sources else 1,len(x)))[:2]:
                edges.append({'src_file':tp,'dst_file':dst,'kind':'test','confidence':0.95,'detail':'same_stem'})
        hs=[p for p in impl if PurePosixPath(p).suffix.lower() in headers];ss=[p for p in impl if PurePosixPath(p).suffix.lower() in sources]
        for h in hs[:2]:
            for src in ss[:2]:edges.append({'src_file':src,'dst_file':h,'kind':'companion','confidence':0.92,'detail':'same_stem'})
    return edges

def extract_base_hints(text,language):
    lang=canonical_language(language) or language;out=[];pats=[]
    if lang=='Python':pats=[r'(?m)^\s*class\s+\w+\s*\(([^)]*)\)']
    elif lang in {'Java','JavaScript','TypeScript','TSX'}:pats=[r'\bclass\s+\w+\s+extends\s+([A-Za-z_][\w.]*)',r'\b(?:class|interface)\s+\w+[^\n{]*?implements\s+([^\n{]+)']
    elif lang in {'C#','Kotlin','Swift'}:pats=[r'\b(?:class|struct|interface|protocol)\s+\w+[^\n{]*?:\s*([^\n{]+)']
    elif lang=='C++':pats=[r'\b(?:class|struct)\s+\w+\s*:\s*([^\n{]+)']
    elif lang=='Rust':pats=[r'\bimpl\s+([A-Za-z_][\w:]*)\s+for\s+[A-Za-z_][\w:]*']
    for pat in pats:
        for m in re.finditer(pat,text):
            val=m.group(1)
            for part in re.split(r'[, ]+',val):
                part=re.sub(r'^(?:public|private|protected|virtual|final|open|internal|static)$','',part.strip()).strip('&*?:<>')
                name=part.split('.')[-1].split('::')[-1]
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',name) and name not in out:out.append(name)
    return out[:32]

def _block_ref(f,start,end,symbol='',kind='file'):
    loc=f.get('locator') or {}; bpt=float(f.get('bytes_per_token',4.0)); text=str(f.get('text') or '')
    return {'format':loc.get('format') or f.get('format'),'pack_path':loc.get('pack_path') or f.get('pack_path'),'offset':int(loc.get('offset',f.get('offset',0))),
            'start':int(start),'end':int(end),'file_path':f.get('path') or f.get('file_path',''),'language':f.get('language'),'symbol':symbol,'kind':kind,
            'approx_tokens':_estimate_tokens(text[int(start):int(end)],bpt) if text else max(1,int((int(end)-int(start))/max(bpt,1e-6)))}

def analyze_code_repo(repo:Mapping[str,Any],disable_tree_sitter:bool=False)->Dict[str,Any]:
    analyzer=StructuralAnalyzer(disable_tree_sitter);files=list(repo.get('files') or []);paths=[f['path'] for f in files if f.get('path')];path_idx=RepoPathIndex(paths)
    analyses={};imports={};by_path={f['path']:f for f in files};blocks=[];block_by_file=defaultdict(list);symbol_files=defaultdict(set)
    for f in files:
        if not f.get('language') or f.get('is_meta'):continue
        a=analyzer.analyze(f['language'],f['text']);h=extract_import_hints(f['text'],f['language']);analyses[f['path']]=a;imports[f['path']]=h
        syms=a.symbols or [Symbol(PurePosixPath(f['path']).name,PurePosixPath(f['path']).name,'file',0,len(f['text']),'',1,0)]
        max_block_chars=48000
        for si,sym in enumerate(syms):
            if sym.end<=sym.start:continue
            segments=[]
            pos=sym.start
            while pos<sym.end:
                end=min(sym.end,pos+max_block_chars)
                if end<sym.end:
                    nl=f['text'].rfind('\n',pos+max_block_chars//2,end)
                    if nl>pos:end=nl+1
                segments.append((pos,end));pos=end
            for ci,(sa,se) in enumerate(segments):
                name=sym.name if len(segments)==1 else f'{sym.name}#chunk{ci+1}'
                kind=sym.kind if len(segments)==1 else 'symbol_chunk'
                bid=stable_u64(f"{repo.get('repo_key')}|{f['path']}|{sa}|{se}|{name}")
                fim=[]
                for sp in a.fim_spans:
                    aa=max(sa,sp.start);bb=min(se,sp.end)
                    if bb-aa>=8 and not (aa==sa and bb==se):fim.append({'start':aa-sa,'end':bb-sa,'kind':sp.kind})
                b={'block_id':bid,'source':'code','primary_language':f['language'],'repo_key':repo.get('repo_key'),'file_path':f['path'],'symbol':name,'kind':kind,
                   'ref':_block_ref(f,sa,se,name,kind),'approx_tokens':_estimate_tokens(f['text'][sa:se],float(f.get('bytes_per_token',4.0))),'fim_spans':fim,'outline':a.outline,
                   'quality':_file_quality(f['text'],a,is_test=f.get('is_test',False),is_meta=f.get('is_meta',False)),'parse_ok':a.parse_ok,'backend':a.backend}
                blocks.append(b);block_by_file[f['path']].append(b);symbol_files[sym.name].add(f['path'])
    edges=[];incoming=Counter();outgoing=Counter();resolved=Counter()
    for fp,hints in imports.items():
        srcblocks=block_by_file.get(fp,[])
        if not srcblocks:continue
        for hint in hints:
            dst=path_idx.resolve(fp,by_path[fp].get('language'),hint)
            if dst and block_by_file.get(dst):
                edges.append({'src_file':fp,'dst_file':dst,'kind':'import','confidence':1.0,'detail':hint});outgoing[fp]+=1;incoming[dst]+=1;resolved[fp]+=1
                if by_path[fp].get('is_test') or is_test_path(fp):edges.append({'src_file':fp,'dst_file':dst,'kind':'test','confidence':0.9,'detail':hint})
    for fp,a in analyses.items():
        for name in a.calls:
            fs=symbol_files.get(name,set())
            if len(fs)==1:
                dst=next(iter(fs))
                if dst!=fp:
                    edges.append({'src_file':fp,'dst_file':dst,'kind':'call','confidence':0.7,'detail':name});outgoing[fp]+=1;incoming[dst]+=1
                    if by_path[fp].get('is_test') or is_test_path(fp):edges.append({'src_file':fp,'dst_file':dst,'kind':'test','confidence':0.85,'detail':name})
    parsed=sum(1 for a in analyses.values() if a.parse_ok); tests=sum(1 for f in files if f.get('is_test') or is_test_path(f.get('path',''))); meta=sum(1 for f in files if f.get('is_meta'))
    import_total=sum(len(v) for v in imports.values()); resolved_total=sum(resolved.values()); symbol_total=sum(len(a.symbols) for a in analyses.values())
    cf=max(1,len(analyses)); repo_quality=max(0.0,min(1.0,0.28*(parsed/cf)+0.16*min(1,tests/max(1,cf*.2))+0.10*min(1,meta/2)+0.22*(resolved_total/max(1,import_total) if import_total else .35)+0.14*min(1,symbol_total/max(1,cf*5))+0.10*min(1,math.log2(max(1,cf))/4)))
    # attach file-level semantic score to blocks
    for b in blocks:
        fp=b['file_path'];a=analyses.get(fp);b['semantic']=_semantic_score(a,len(imports.get(fp,[])),resolved[fp],incoming[fp],outgoing[fp],is_test_path(fp)) if a else 0
        b['quality']=max(0,min(1,.72*float(b['quality'])+.28*repo_quality))
    return {'blocks':blocks,'edges':edges,'analyses':analyses,'imports':imports,'repo_quality':repo_quality}

def _estimate_tokens(text:str,bytes_per_token:float)->int:
    return max(1,int(round(len(text.encode('utf-8'))/max(float(bytes_per_token),1e-6))))

def _text_quality(text,source):
    if not text.strip():return 0.0
    n=len(text);printable=sum(c.isprintable() or c in '\n\t' for c in text)/max(1,n);alpha=sum(c.isalpha() for c in text)/max(1,n)
    q=0.45+0.25*min(1,n/4000)+0.2*printable+0.1*min(1,alpha/.5)
    if source.startswith('math_') and ('\\' in text or '$' in text):q+=.05
    return max(0,min(1,q))

def _generic_semantic_shard(args):
    shard,out_root,chunk_tokens=args;shard=Path(shard);out_root=Path(out_root);source=shard.parent.name
    bp=out_root/'blocks'/source/shard.name;cp=out_root/'candidates'/source/shard.name
    with FramedWriter(bp) as bw,FramedWriter(cp) as cw:
        count=0
        for rec in iter_frames(shard):
            text=read_text(rec);bpt=float(rec.get('bytes_per_token',4.0));raw=split_math_blocks(text) if source.startswith('math_') else split_natural_blocks(text)
            max_chars=max(2048,int(chunk_tokens*bpt)); raw=[x for b in raw for x in _split_large_paragraph_block(b,max_chars)]
            if not raw:continue
            refs=[]
            for i,b in enumerate(raw):
                bid=stable_u64(f"{rec['record_id']}|{b.start}|{b.end}|{b.kind}");ref={'format':rec['format'],'pack_path':rec['pack_path'],'offset':int(rec['offset']),'start':b.start,'end':b.end,'kind':b.kind}
                block={'block_id':bid,'source':source,'document_id':int(rec['record_id']),'kind':b.kind,'ref':ref,'approx_tokens':_estimate_tokens(b.text,bpt),'quality':_text_quality(b.text,source)};bw.write(block);refs.append(block)
            # contiguous semantic windows, no tokenization here
            cur=[];tok=0;window=0
            def emit(parts):
                nonlocal count,window
                if not parts:return
                approx=sum(x['approx_tokens'] for x in parts);qid=stable_u64(f"cand|{rec['record_id']}|{window}");window+=1
                fim_indices=[]
                if source.startswith('math_') and len(parts)>=3:
                    for j,x in enumerate(parts[1:-1],1):
                        if x['kind'] in {'latex_env','display_math','paragraph'} and x['approx_tokens']>=4:fim_indices.append(j)
                cand={'candidate_id':qid,'source':source,'primary_language':None,'group_id':str(rec['record_id']),'approx_tokens':approx,'quality':sum(x['quality'] for x in parts)/len(parts),
                      'semantic':0.2 if source.startswith('math_') else 0.0,'tasks':['causal']+(['math_fim'] if fim_indices else []),'parts':[x['ref'] for x in parts],'fim_indices':fim_indices}
                cw.write(cand);count+=1
            for b in refs:
                if cur and tok+b['approx_tokens']>chunk_tokens:emit(cur);cur=[];tok=0
                cur.append(b);tok+=b['approx_tokens']
            emit(cur)
    return source,count


# Pack-ordered generic pass. Hash-partitioned survivor shards scatter every
# document read across all packs (~1.3 survivors per 4.7MB general pack), which
# collapses to device random-read latency. Bucketing locators by pack turns the
# same scan into one forward read per pack (general packs are ~54% live text).
_GENERIC_INDEX_VERSION = 1


def _generic_index_dir(out: Path) -> Path:
    env=os.environ.get('SBP_SEMANTIC_INDEX_DIR')
    return Path(env) if env else Path(out)/'_pack_index'


def _generic_index_signature(shards, buckets: int) -> str:
    return _stable_hash({'version': _GENERIC_INDEX_VERSION, 'buckets': int(buckets),
                         'shards': [f'{p.parent.name}/{p.name}' for p in sorted(map(Path, shards))]})


def _ensure_generic_locator_index(shards, buckets: int, index_dir: Path) -> str:
    index_dir=Path(index_dir);index_dir.mkdir(parents=True,exist_ok=True)
    sig=_generic_index_signature(shards,buckets);done=index_dir/'index_done.json'
    if done.is_file():
        try:
            m=json.loads(done.read_text(encoding='utf-8'))
            if m.get('signature')==sig and all((index_dir/f'bucket_{i:05d}.mpk').is_file() for i in range(buckets)):
                print(f'[semantic 1/3 index] RESUME buckets={buckets} shards={len(shards)}',flush=True);return sig
        except Exception:pass
    for q in list(index_dir.glob('bucket_*.mpk'))+list(index_dir.glob('bucket_*.mpk.tmp.*')):
        try:q.unlink()
        except FileNotFoundError:pass
    writers={i:FramedWriter(index_dir/f'bucket_{i:05d}.mpk') for i in range(int(buckets))}
    rows=0
    try:
        for k,p in enumerate(sorted(map(Path,shards)),1):
            for rec in iter_frames(p):
                writers[stable_u64(str(rec['pack_path']))%int(buckets)].write(
                    (str(rec['pack_path']),str(rec.get('format') or ''),int(rec['offset']),int(rec['record_id']),
                     float(rec.get('bytes_per_token',4.0)),str(rec.get('source') or p.parent.name)))
                rows+=1
            if k%200==0:print(f'[semantic 1/3 index] shards={k}/{len(shards)} rows={rows:,}',flush=True)
    finally:
        for w in writers.values():w.close()
    atomic_json(done,{'signature':sig,'buckets':int(buckets),'shards':len(shards),'rows':int(rows)})
    print(f'[semantic 1/3 index] COMPLETE rows={rows:,} buckets={buckets}',flush=True)
    return sig


def _generic_bucket_cleanup(out: Path, bucket: int) -> None:
    for root in ('blocks','candidates'):
        base=Path(out)/root
        if not base.is_dir():continue
        for target in base.glob(f'*/b{bucket:05d}.mpk'):
            try:target.unlink()
            except FileNotFoundError:pass
        for target in base.glob(f'*/b{bucket:05d}.mpk.tmp.*'):
            try:target.unlink()
            except FileNotFoundError:pass


def _emit_generic_document(source, record_id, fmt, pack_path, offset, text, bpt, chunk_tokens, bw, cw):
    raw=split_math_blocks(text) if source.startswith('math_') else split_natural_blocks(text)
    max_chars=max(2048,int(chunk_tokens*bpt));raw=[x for b in raw for x in _split_large_paragraph_block(b,max_chars)]
    if not raw:return 0
    refs=[]
    for b in raw:
        bid=stable_u64(f"{record_id}|{b.start}|{b.end}|{b.kind}")
        ref={'format':fmt,'pack_path':pack_path,'offset':int(offset),'start':b.start,'end':b.end,'kind':b.kind}
        block={'block_id':bid,'source':source,'document_id':int(record_id),'kind':b.kind,'ref':ref,
               'approx_tokens':_estimate_tokens(b.text,bpt),'quality':_text_quality(b.text,source)}
        if bw is not None:bw.write(block)
        refs.append(block)
    cur=[];tok=0;window=0;n=0
    def emit(parts):
        nonlocal n,window
        if not parts:return
        approx=sum(x['approx_tokens'] for x in parts);qid=stable_u64(f"cand|{record_id}|{window}");window+=1
        fim=[]
        if source.startswith('math_') and len(parts)>=3:
            for j,x in enumerate(parts[1:-1],1):
                if x['kind'] in {'latex_env','display_math','paragraph'} and x['approx_tokens']>=4:fim.append(j)
        cand={'candidate_id':qid,'source':source,'primary_language':None,'group_id':str(record_id),'approx_tokens':approx,
              'quality':sum(x['quality'] for x in parts)/len(parts),'semantic':0.2 if source.startswith('math_') else 0.0,
              'tasks':['causal']+(['math_fim'] if fim else []),'parts':[x['ref'] for x in parts],'fim_indices':fim}
        cw.write(cand);n+=1
    for b in refs:
        if cur and tok+b['approx_tokens']>chunk_tokens:emit(cur);cur=[];tok=0
        cur.append(b);tok+=b['approx_tokens']
    emit(cur)
    return n


def _generic_pack_bucket(args):
    bucket_file,out_root,chunk_tokens,bucket=args
    out_root=Path(out_root);bucket=int(bucket);_generic_bucket_cleanup(out_root,bucket)
    rows=list(iter_frames(bucket_file));groups={}
    for pack_path,fmt,off,rid,bpt,source in rows:
        groups.setdefault(str(pack_path),[]).append((int(off),int(rid),str(fmt),float(bpt),str(source)))
    writers={};counts=Counter();nbytes=0
    def pick(source):
        w=writers.get(source)
        if w is None:
            w=FramedWriter(out_root/'candidates'/source/f'b{bucket:05d}.mpk')
            writers[source]=w
        return w
    ok=False
    try:
        hs=PACK_RECORD.size
        for pack in sorted(groups):
            locs=sorted(groups[pack])
            with open(pack,'rb') as f:
                fd=f.fileno()
                try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_SEQUENTIAL)
                except OSError:pass
                for off,rid,fmt,bpt,source in locs:
                    h=os.pread(fd,hs,off)
                    if len(h)!=hs:raise RuntimeError(f'truncated pack header {pack}@{off}')
                    _,n=PACK_RECORD.unpack(h);data=os.pread(fd,n,off+hs)
                    if len(data)!=n:raise RuntimeError(f'truncated pack payload {pack}@{off}')
                    nbytes+=len(data)
                    cw=pick(source)
                    counts[source]+=_emit_generic_document(source,rid,fmt,pack,off,data.decode('utf-8','strict'),bpt,chunk_tokens,None,cw)
        for cw in writers.values():cw.close()
        ok=True
    finally:
        if not ok:
            for cw in writers.values():
                if not cw.closed:
                    try:cw.f.close()
                    except Exception:pass
                    cw.closed=True
                    try:cw.tmp.unlink()
                    except FileNotFoundError:pass
    atomic_json(out_root/'_bucket_done'/f'b{bucket:05d}.json',{'bucket':bucket,'locators':len(rows),'text_bytes':int(nbytes),'candidates':dict(counts)})
    return {'candidates':dict(counts),'locators':len(rows),'text_bytes':int(nbytes)}

_DROP_IDS_WORKER_CACHE={}

def _load_drop_ids(dedup_dir,source=None):
    p=Path(dedup_dir)/'drop_ids.mpk'
    if not p.exists():return set()
    return {int(r['record_id']) for r in iter_frames(p) if source is None or str(r.get('source'))==str(source)}

def _load_drop_ids_cached(dedup_dir,source=None):
    key=(str(Path(dedup_dir).resolve()),None if source is None else str(source))
    cached=_DROP_IDS_WORKER_CACHE.get(key)
    if cached is None:
        cached=_load_drop_ids(dedup_dir,source=source);_DROP_IDS_WORKER_CACHE[key]=cached
    return cached

def _route_code_global(shards:Sequence[Path],run_root:Path|str,parts:int)->int:
    """Route code inventory into only `parts` global repo files with bounded RAM/inodes."""
    from collections import OrderedDict
    run_root=Path(run_root)
    if run_root.exists():shutil.rmtree(run_root)
    run_root.mkdir(parents=True,exist_ok=True)
    max_open=max(8,int(os.environ.get('SBP_CODE_ROUTE_OPEN_WRITERS','64')));writers=OrderedDict();count=0
    def writer(part:int):
        w=writers.pop(part,None)
        if w is None:w=_AppendFramedWriter(run_root/f'p{part:05d}.mpk')
        writers[part]=w
        while len(writers)>max_open:
            _k,old=writers.popitem(last=False);old.close()
        return w
    try:
        for shard in shards:
            for r in iter_frames(shard):
                key=r.get('repo_key')
                if not key:continue
                part=stable_u64(str(key))%int(parts);writer(part).write(r);count+=1
    finally:
        for w in writers.values():w.close()
    return count


def _cleanup_code_partition_outputs(out_root,part):
    out_root=Path(out_root)
    for p in [out_root/'blocks'/'code'/f'p{part:05d}.mpk',out_root/'repos'/'code'/f'p{part:05d}.mpk',out_root/'code_done'/f'p{part:05d}.json']:
        try:p.unlink()
        except FileNotFoundError:pass
    cand=out_root/'candidates'/'code'
    if cand.exists():
        for p in cand.glob(f'*/p{part:05d}.mpk'):
            try:p.unlink()
            except FileNotFoundError:pass
    # A SIGKILL/SIGSEGV cannot run context-manager cleanup; remove orphan atomics/scratch on retry.
    for root in (out_root/'blocks'/'code',out_root/'repos'/'code',cand):
        if root.exists():
            for q in root.rglob(f'p{part:05d}.mpk.tmp.*'):
                try:q.unlink()
                except FileNotFoundError:pass
    scratch=out_root/'repo_scratch'/f'p{part:05d}'
    if scratch.exists():shutil.rmtree(scratch,ignore_errors=True)
    try:(out_root/'code_active'/f'p{part:05d}.json').unlink()
    except FileNotFoundError:pass


def _tree_path_value(x):
    if isinstance(x,Mapping):
        raw=x.get('file_path') or x.get('path') or ''
    else:raw=x
    return str(raw or '').replace('\\','/').strip('/')


def hierarchical_tree_summary(tree_entries,max_lines=500,detail_dirs=8,detail_children=6):
    """Bound a huge repository tree while preserving global hierarchy/counts.

    Small trees stay exact. Large trees become a deterministic two-level summary;
    no arbitrary ``tree[:N]`` prefix bias is used.
    """
    max_lines=max(1,int(max_lines));paths=sorted(set(p for p in (_tree_path_value(x) for x in (tree_entries or [])) if p))
    if len(paths)<=max_lines:return paths
    top=Counter();root=[]
    for path in paths:
        parts=[x for x in path.split('/') if x]
        if len(parts)==1:root.append(parts[0])
        elif parts:top[parts[0]]+=1
    ranked=sorted(top.items(),key=lambda kv:(-int(kv[1]),kv[0]))
    detail_names={name for name,_ in ranked[:max(0,int(detail_dirs))]}
    second={name:Counter() for name in detail_names}
    for path in paths:
        parts=[x for x in path.split('/') if x]
        if len(parts)>=2 and parts[0] in second:
            second[parts[0]][parts[1]]+=1
    lines=[f'[repo-tree-summary] total_files={len(paths)} root_files={len(root)} top_dirs={len(top)}']
    # Root metadata/build files are disproportionately informative and cheap.
    for name in sorted(root)[:min(16,max_lines-len(lines))]:lines.append(name)
    for name,count in ranked:
        if len(lines)>=max_lines:break
        lines.append(f'{name}/ [{int(count)} files]')
        if name not in detail_names:continue
        children=sorted(second[name].items(),key=lambda kv:(-int(kv[1]),kv[0]))
        emitted=0;covered=0
        for child,n in children:
            if len(lines)>=max_lines or emitted>=max(0,int(detail_children)):break
            lines.append(f'  {name}/{child}/ [{int(n)} files]');emitted+=1;covered+=int(n)
        rest=int(count)-covered
        if rest>0 and len(lines)<max_lines and len(children)>emitted:
            lines.append(f'  {name}/... [{rest} other files]')
    return lines[:max_lines]


def _repo_row_rank(repo_key,row):
    return stable_u64(f'giant-row|{repo_key}|{row.get("record_id",0)}|{row.get("file_path","")}')


class _GiantRepoBuffer:
    """Per-repository metadata buffer with deterministic bottom-k giant-repo cap."""
    def __init__(self,repo_key,giant_threshold=20000,row_cap=12000,tree_summary_lines=500):
        self.repo_key=str(repo_key);self.giant_threshold=max(1,int(giant_threshold));self.row_cap=max(1,int(row_cap));self.tree_summary_lines=max(1,int(tree_summary_lines))
        self.manifest=None;self.small=[];self.data_heap=[];self.meta_heap=[];self.files_seen=0;self.giant=False;self.tree_original_files=0
    def _heap_add(self,heap,row,cap):
        import heapq
        if cap<=0:return
        rank=_repo_row_rank(self.repo_key,row);path=str(row.get('file_path') or '');rid=int(row.get('record_id') or 0);item=(-rank,path,rid,row)
        if len(heap)<cap:heapq.heappush(heap,item)
        elif rank < -heap[0][0]:heapq.heapreplace(heap,item)
    def _activate(self):
        if self.giant:return
        self.giant=True;old=self.small;self.small=[]
        meta_cap=min(64,max(1,self.row_cap//8));data_cap=max(0,self.row_cap-meta_cap)
        for row in old:self._heap_add(self.meta_heap if row.get('is_meta') else self.data_heap,row,meta_cap if row.get('is_meta') else data_cap)
    def _shrink_manifest(self,row):
        out=dict(row);raw=list(out.get('tree') or []);self.tree_original_files=len(raw)
        if len(raw)>self.tree_summary_lines:
            out['tree']=hierarchical_tree_summary(raw,self.tree_summary_lines);out['tree_is_summary']=True
        else:
            out['tree']=[p for p in (_tree_path_value(x) for x in raw) if p];out['tree_is_summary']=False
        original=int((out.get('repo_metadata') or {}).get('num_files') or out.get('num_files') or self.tree_original_files or 0)
        if original>self.giant_threshold:self._activate()
        return out
    def add(self,row):
        if row.get('kind')=='repo':
            self.manifest=self._shrink_manifest(row);return
        if row.get('kind')!='code_file':return
        self.files_seen+=1
        if not self.giant and self.files_seen>min(self.giant_threshold,self.row_cap):self._activate()
        if not self.giant:self.small.append(row);return
        meta_cap=min(64,max(1,self.row_cap//8));data_cap=max(0,self.row_cap-meta_cap)
        self._heap_add(self.meta_heap if row.get('is_meta') else self.data_heap,row,meta_cap if row.get('is_meta') else data_cap)
    def finish(self):
        if self.giant:
            kept=[x[3] for x in self.meta_heap]+[x[3] for x in self.data_heap]
            kept.sort(key=lambda r:(_repo_row_rank(self.repo_key,r),str(r.get('file_path') or ''),int(r.get('record_id') or 0)))
        else:kept=list(self.small)
        manifest=dict(self.manifest or {'kind':'repo','repo_key':self.repo_key,'repo_path':self.repo_key,'tree':[]})
        manifest['giant_repo']=bool(self.giant);manifest['repo_rows_seen']=int(self.files_seen);manifest['repo_rows_kept']=len(kept);manifest['tree_original_files']=int(self.tree_original_files)
        return [manifest]+kept


def _select_giant_repo_targets(targets,repo_key,cap):
    cap=max(1,int(cap));rows=list(targets)
    if len(rows)<=cap:return sorted(rows,key=lambda f:(stable_u64(f'giant-target|{repo_key}|{f.get("path","")}'),str(f.get('path') or '')))
    by_lang=defaultdict(list)
    for f in rows:by_lang[str(f.get('language') or '<unknown>')].append(f)
    for lang in by_lang:by_lang[lang].sort(key=lambda f:(stable_u64(f'giant-target|{repo_key}|{f.get("path","")}'),str(f.get('path') or '')))
    selected=[];seen=set()
    # One representative per language first; the remaining slots stay natural-frequency-ish.
    if cap>=len(by_lang):
        for lang in sorted(by_lang):
            f=by_lang[lang][0];selected.append(f);seen.add(str(f.get('path') or ''))
    pool=[f for f in rows if str(f.get('path') or '') not in seen]
    pool.sort(key=lambda f:(stable_u64(f'giant-target|{repo_key}|{f.get("path","")}'),str(f.get('path') or '')))
    selected.extend(pool[:max(0,cap-len(selected))])
    return sorted(selected,key=lambda f:(stable_u64(f'giant-target|{repo_key}|{f.get("path","")}'),str(f.get('path') or '')))


def _target_neighborhoods(paths,target_paths,max_paths=24):
    max_paths=max(1,int(max_paths));target_paths=set(str(x) for x in target_paths);parents={str(PurePosixPath(x).parent) for x in target_paths};members=defaultdict(list)
    for p in sorted(set(str(x) for x in paths if x)):
        parent=str(PurePosixPath(p).parent)
        if parent in parents and len(members[parent])<max_paths:members[parent].append(p)
    return {t:list(members.get(str(PurePosixPath(t).parent),[])) for t in target_paths}


def _compose_tree_context(global_tree,local_tree,limit=200):
    limit=max(0,int(limit))
    if limit<=0:return []
    g=[str(x) for x in (global_tree or []) if str(x)];local=[str(x) for x in (local_tree or []) if str(x)]
    if not local:return g[:limit]
    out=[]
    if g and g[0].startswith('[repo-tree-summary]'):out.append(g[0]);g=g[1:]
    if len(out)<limit:out.append('[target-neighborhood]')
    for x in local:
        if len(out)>=limit:break
        if x not in out:out.append(x)
    for x in g:
        if len(out)>=limit:break
        if x not in out:out.append(x)
    return out[:limit]


def _iter_repo_groups_bounded(files,out_root,part,repo_buckets=32,giant_threshold=20000,row_cap=12000,tree_summary_lines=500):
    """Externally group one routed code partition by repository with bounded RAM.

    v9 first re-bucketed a partition with ``stable_u64(repo_key) % repo_buckets``.
    Because the route partition itself is ``stable_u64(repo_key) % 256`` and
    ``repo_buckets`` defaults to 32 (a divisor of 256), every row in p00174, for
    example, necessarily landed in bucket 14.  The subsequent ``groups`` dict then
    materialized the *entire* ~200 MiB routed partition as Python objects before the
    first repository was yielded.  On real data this expanded to >82 GiB anon RSS.

    This implementation uses an on-disk SQLite B-tree keyed by (repo_key, sequence)
    and holds only one ``_GiantRepoBuffer`` at a time.  ``repo_buckets`` remains in
    the signature for config/backward compatibility but no longer controls memory.
    """
    out_root=Path(out_root);scratch=out_root/'repo_scratch'/f'p{part:05d}'
    if scratch.exists():shutil.rmtree(scratch,ignore_errors=True)
    scratch.mkdir(parents=True,exist_ok=True)
    db_path=scratch/'repo_groups.sqlite'
    conn=None
    try:
        conn=_open_scale_sqlite(db_path)
        # Keep SQLite itself bounded; payload pages spill to disk rather than mmap/RAM.
        conn.execute('PRAGMA cache_size=-65536')  # ~64 MiB
        conn.execute('PRAGMA mmap_size=0')
        conn.execute('CREATE TABLE rows(repo_key TEXT NOT NULL, seq INTEGER NOT NULL, payload BLOB NOT NULL, PRIMARY KEY(repo_key,seq)) WITHOUT ROWID')
        batch=[];seq=0;batch_cap=max(64,int(os.environ.get('SBP_REPO_GROUP_INSERT_BATCH','2048')))
        conn.execute('BEGIN')
        for src in files:
            for r in iter_frames(src):
                key=str(r.get('repo_key') or '')
                if not key:continue
                payload=msgpack.packb(r,use_bin_type=True)
                batch.append((key,seq,payload));seq+=1
                if len(batch)>=batch_cap:
                    conn.executemany('INSERT INTO rows(repo_key,seq,payload) VALUES(?,?,?)',batch);batch.clear()
                    conn.commit();conn.execute('BEGIN')
        if batch:
            conn.executemany('INSERT INTO rows(repo_key,seq,payload) VALUES(?,?,?)',batch);batch.clear()
        conn.commit()

        current_key=None;buf=None
        # PRIMARY KEY order is repo_key,seq, so no in-memory grouping/sort is needed.
        for key,payload in conn.execute('SELECT repo_key,payload FROM rows ORDER BY repo_key,seq'):
            key=str(key)
            if current_key is None:
                current_key=key;buf=_GiantRepoBuffer(key,giant_threshold,row_cap,tree_summary_lines)
            elif key!=current_key:
                rows=buf.finish()
                yield current_key,rows
                del rows,buf
                current_key=key;buf=_GiantRepoBuffer(key,giant_threshold,row_cap,tree_summary_lines)
            row=msgpack.unpackb(payload,raw=False,strict_map_key=False)
            buf.add(row)
            del row,payload
        if buf is not None:
            rows=buf.finish()
            yield current_key,rows
            del rows,buf
    finally:
        if conn is not None:
            try:conn.close()
            except Exception:pass
        shutil.rmtree(scratch,ignore_errors=True)

def _file_desc_from_row(r,drops):
    fm=r.get('file_metadata') or {};path=str(r.get('file_path') or fm.get('file_path') or '');low='/'+path.lower().lstrip('/')
    generated_hint=bool(fm.get('is_vendor')) or any(x in low for x in ('/node_modules/','/vendor/','/dist/','/build/','/generated/'))
    return {'row':r,'path':path,'language':canonical_language(r.get('language') or fm.get('language')),'is_test':is_test_path(path),'is_meta':bool(r.get('is_meta')),
            'countable':bool(r.get('countable')),'generated_hint':generated_hint,'duplicate':int(r.get('record_id',0)) in drops,'record_id':int(r.get('record_id',0)),
            'bytes_per_token':float(r.get('bytes_per_token',4.0)),'payload_len':int(r.get('payload_len',0)),
            'locator':{'format':r.get('format'),'pack_path':r.get('pack_path'),'offset':int(r.get('offset',0))},'file_metadata':fm}


def _cap_analysis_for_file(a, symbol_cap=4096, fim_span_cap=8192, call_cap=4096):
    """Deterministically bound per-file structural metadata retained by code-index.

    The parser may legitimately discover hundreds of thousands of nodes in a <=2 MiB
    file.  Downstream candidate generation never needs an unbounded number of symbols,
    FIM spans, or call names from one file, so trim these collections before they can
    participate in repo-level products.
    """
    symbol_cap=max(1,int(symbol_cap));fim_span_cap=max(1,int(fim_span_cap));call_cap=max(1,int(call_cap))
    syms=sorted(a.symbols or [],key=lambda x:(int(x.start),int(x.end),str(x.kind),str(x.name)))
    spans=sorted(a.fim_spans or [],key=lambda x:(int(x.start),int(x.end),str(x.kind),str(x.name)))
    calls=sorted(set(str(x) for x in (a.calls or []) if x))
    a.symbols=syms[:symbol_cap];a.fim_spans=spans[:fim_span_cap];a.calls=calls[:call_cap]
    return a


def _bounded_fim_for_block(spans,start,end,cap=64):
    """Return at most ``cap`` FIM spans overlapping [start,end), relative to block.

    Spans are expected in source order (the cap helper above guarantees that).  The
    output is deterministic and cannot grow quadratically with nested AST structure.
    """
    cap=max(0,int(cap));out=[];start=int(start);end=int(end)
    if cap<=0 or end<=start:return out
    for sp in spans or []:
        sa=int(sp.start);sb=int(sp.end)
        if sa>=end:break
        if sb<=start:continue
        aa=max(start,sa);bb=min(end,sb)
        if bb-aa>=8 and not (aa==start and bb==end):
            out.append({'start':aa-start,'end':bb-start,'kind':sp.kind})
            if len(out)>=cap:break
    return out


def _bounded_relation_insert(table,file_path,item,cap):
    cap=max(1,int(cap));arr=table[file_path]
    # Avoid duplicate file-level relations before they are expanded to block refs.
    key=(str(item.get('role')),str(item.get('other_file')),str(item.get('detail','')))
    for old in arr:
        if (str(old.get('role')),str(old.get('other_file')),str(old.get('detail','')))==key:
            if float(item.get('score',0))>float(old.get('score',0)):old.update(item)
            return
    if len(arr)<cap:
        arr.append(item);return
    # Keep the highest-confidence relations deterministically.
    worst=min(range(len(arr)),key=lambda i:(float(arr[i].get('score',0)),str(arr[i].get('role')),str(arr[i].get('other_file'))))
    if (float(item.get('score',0)),str(item.get('role')),str(item.get('other_file'))) > (float(arr[worst].get('score',0)),str(arr[worst].get('role')),str(arr[worst].get('other_file'))):
        arr[worst]=item


def _build_bounded_file_relations(edges,cap_per_file=32):
    """Collapse repo edges to a bounded per-file relation index.

    v10 expanded every edge to every source/destination block and stored a relation
    list for each block.  A single repo with many blocks and dense relations therefore
    produced O(blocks*edges) Python dictionaries and could exceed 80 GiB RSS.  Final
    candidates only consume a handful of relations, so retain a small file-level index
    and expand it lazily while each candidate is emitted.
    """
    rel=defaultdict(list)
    reverse={'call':'called_by','import':'imported_by','test':'tested_by','companion':'companion','inherits':'inherited_by'}
    for e in edges or []:
        src=str(e.get('src_file') or '');dst=str(e.get('dst_file') or '')
        if not src or not dst:continue
        role=str(e.get('kind') or 'related');score=float(e.get('confidence',0));detail=str(e.get('detail') or '')
        _bounded_relation_insert(rel,src,{'role':role,'score':score,'other_file':dst,'detail':detail},cap_per_file)
        _bounded_relation_insert(rel,dst,{'role':reverse.get(role,'related_by'),'score':score,'other_file':src,'detail':detail},cap_per_file)
    return rel


def _bounded_unique(seq,cap):
    cap=max(0,int(cap));out=[];seen=set()
    if cap<=0:return out
    for x in seq or []:
        x=str(x)
        if not x or x in seen:continue
        seen.add(x);out.append(x)
        if len(out)>=cap:break
    return out

def _process_code_repo(repo_key,rows,manifest,drop_lookup,analyzer,max_related,max_parse_bytes,max_code_file_bytes,bw,rw,writers,out_root,part,giant_threshold=20000,giant_target_cap=4096,giant_neighbor_cap=512,target_tree_paths=24):
    file_rows=[r for r in rows if r.get('kind')=='code_file']
    drops=drop_lookup.contains_many((int(r.get('record_id',0)) for r in file_rows),source='code') if hasattr(drop_lookup,'contains_many') else drop_lookup
    desc=[_file_desc_from_row(r,drops) for r in file_rows]
    if not desc:return Counter(),{'oversize':0,'generated':0,'read_error':0,'parsed':0,'regex_fallback':0}
    by_path={f['path']:f for f in desc if f['path']};path_idx=RepoPathIndex(by_path.keys())
    targets=[f for f in desc if f['language'] in CODE_LANGUAGES and f['countable'] and not f['is_meta'] and not f['duplicate'] and not f['generated_hint']]
    if not targets:return Counter(),{'oversize':0,'generated':0,'read_error':0,'parsed':0,'regex_fallback':0}
    meta_files=[f for f in desc if f['is_meta']]
    original_files=int(manifest.get('tree_original_files') or (manifest.get('repo_metadata') or {}).get('num_files') or manifest.get('repo_rows_seen') or len(desc))
    giant=bool(manifest.get('giant_repo')) or original_files>int(giant_threshold) or len(targets)>int(giant_target_cap)
    diag=Counter()
    if giant:
        diag['giant_repos']+=1;diag['giant_original_files']+=original_files;diag['giant_rows_kept']+=len(desc)
        before=len(targets);targets=_select_giant_repo_targets(targets,repo_key,giant_target_cap);diag['giant_targets_dropped']+=max(0,before-len(targets))
    neighborhoods=_target_neighborhoods(by_path.keys(),[f['path'] for f in targets],target_tree_paths) if giant else {}

    repo_block_cap=max(1024,int(os.environ.get('SBP_REPO_BLOCK_CAP','131072')))
    file_block_cap=max(64,int(os.environ.get('SBP_FILE_BLOCK_CAP','4096')))
    file_symbol_cap=max(64,int(os.environ.get('SBP_FILE_SYMBOL_CAP','4096')))
    file_fim_span_cap=max(64,int(os.environ.get('SBP_FILE_FIM_SPAN_CAP','8192')))
    block_fim_span_cap=max(1,int(os.environ.get('SBP_BLOCK_FIM_SPAN_CAP','64')))
    file_call_cap=max(64,int(os.environ.get('SBP_FILE_CALL_CAP','4096')))
    file_import_cap=max(32,int(os.environ.get('SBP_FILE_IMPORT_CAP','2048')))
    file_base_cap=max(8,int(os.environ.get('SBP_FILE_BASE_CAP','256')))
    repo_edge_cap=max(1024,int(os.environ.get('SBP_REPO_EDGE_CAP','262144')))
    file_relation_cap=max(int(max_related),int(os.environ.get('SBP_FILE_RELATION_CAP',str(max(32,int(max_related)*4)))))
    outline_cap=max(0,int(os.environ.get('SBP_BLOCK_OUTLINE_CHARS','4096')))
    active_path=Path(out_root)/'code_active'/f'p{part:05d}.json'
    def active_phase(phase,**extra):
        if not active_path.exists():return
        payload={'partition':int(part),'repo_key':str(repo_key),'rows':len(rows),'pid':os.getpid(),'phase':str(phase),'updated_at_unix':time.time()};payload.update(extra)
        atomic_json(active_path,payload)

    blocks=[];block_heads=defaultdict(list);symbol_owner={};imports={};bases={};summary={}
    AMBIG=object()
    def note_symbol(name,path):
        name=str(name or '')
        if not name:return
        old=symbol_owner.get(name,AMBIG)
        if old is AMBIG:symbol_owner[name]=path
        elif old!=path:symbol_owner[name]=None
    block_cap_hit=False
    for f in targets:
        if len(blocks)>=repo_block_cap:
            diag['repo_block_cap_hits']+=1;block_cap_hit=True;break
        if max_code_file_bytes and f['payload_len']>max_code_file_bytes:
            diag['oversize']+=1;continue
        try:text=read_text(f['row'])
        except Exception:
            diag['read_error']+=1;continue
        if 'generated file' in text[:1000].lower():
            diag['generated']+=1;del text;continue
        a=analyzer.analyze(f['language'],text,source_bytes=f['payload_len'])
        before_symbols=len(a.symbols or []);before_spans=len(a.fim_spans or []);before_calls=len(a.calls or [])
        _cap_analysis_for_file(a,file_symbol_cap,file_fim_span_cap,file_call_cap)
        if before_symbols>len(a.symbols):diag['file_symbol_cap_hits']+=1
        if before_spans>len(a.fim_spans):diag['file_fim_span_cap_hits']+=1
        if before_calls>len(a.calls):diag['file_call_cap_hits']+=1
        h=_bounded_unique(extract_import_hints(text,f['language']),file_import_cap);imports[f['path']]=h
        bh=_bounded_unique(extract_base_hints(text,f['language']),file_base_cap);bases[f['path']]=bh
        if a.parse_ok:diag['parsed']+=1
        if a.backend=='regex':diag['regex_fallback']+=1
        if a.backend=='tree_sitter':diag['tree_sitter_structured']+=1
        if a.backend=='python_ast':diag['python_ast']+=1
        if a.backend!='regex' and (a.symbols or a.fim_spans):diag['structured']+=1
        diag[f'backend::{f["language"]}::{a.backend}']+=1
        syms=a.symbols or [Symbol(PurePosixPath(f['path']).name,PurePosixPath(f['path']).name,'file',0,len(text),'',1,0)]
        # tree-sitter query patterns can match the same node twice (identical
        # span+name); duplicates would produce duplicate block_ids and crash the
        # planner's unique candidate check.
        _seen_syms=set();_uniq_syms=[]
        for _s in syms:
            _k=(_s.start,_s.end,_s.name)
            if _k in _seen_syms:continue
            _seen_syms.add(_k);_uniq_syms.append(_s)
        if len(_uniq_syms)!=len(syms):diag['duplicate_symbols_dropped']+=len(syms)-len(_uniq_syms)
        syms=_uniq_syms
        q=_file_quality(text,a,is_test=f['is_test'],is_meta=f['is_meta'])
        summary[f['path']]={'parse_ok':bool(a.parse_ok),'structure_confidence':_structure_confidence(a),'symbol_count':len(a.symbols),'calls':tuple(a.calls),'quality':q}
        max_block_chars=48000;file_blocks=0;file_seen_bids=set()
        for sym in syms:
            if sym.end<=sym.start:continue
            if file_blocks>=file_block_cap:
                diag['file_block_cap_hits']+=1;break
            pos=sym.start;ci=0
            while pos<sym.end:
                if len(blocks)>=repo_block_cap:
                    diag['repo_block_cap_hits']+=1;block_cap_hit=True;break
                if file_blocks>=file_block_cap:
                    diag['file_block_cap_hits']+=1;break
                end=min(sym.end,pos+max_block_chars)
                if end<sym.end:
                    nl=text.rfind('\n',pos+max_block_chars//2,end)
                    if nl>pos:end=nl+1
                name=sym.name if sym.end-sym.start<=max_block_chars else f'{sym.name}#chunk{ci+1}';kind=sym.kind if sym.end-sym.start<=max_block_chars else 'symbol_chunk'
                bid=stable_u64(f'{repo_key}|{f["path"]}|{pos}|{end}|{name}')
                fim=_bounded_fim_for_block(a.fim_spans,pos,end,block_fim_span_cap)
                if len(fim)>=block_fim_span_cap:diag['block_fim_span_cap_hits']+=1
                approx=_estimate_tokens(text[pos:end],f['bytes_per_token'])
                ref={'format':f['locator']['format'],'pack_path':f['locator']['pack_path'],'offset':f['locator']['offset'],'start':int(pos),'end':int(end),'file_path':f['path'],'language':f['language'],'symbol':name,'kind':kind,'approx_tokens':approx}
                # Outline is useful for audit, but duplicating a long outline into every
                # block is pure amplification. Keep it only on the first block per file.
                b={'block_id':bid,'source':'code','primary_language':f['language'],'repo_key':repo_key,'file_path':f['path'],'symbol':name,'kind':kind,'ref':ref,
                   'approx_tokens':approx,'fim_spans':fim,'outline':(a.outline[:outline_cap] if outline_cap and file_blocks==0 else ''),'quality':q,'parse_ok':a.parse_ok,'backend':a.backend}
                # Nested symbols can share a name and produce identical chunk spans
                # (same block_id); keep the first write only.
                if bid in file_seen_bids:
                    diag['duplicate_block_writes_skipped']+=1
                else:
                    file_seen_bids.add(bid);blocks.append(b)
                    if len(block_heads[f['path']])<2:block_heads[f['path']].append({'block_id':bid,'ref':ref,'symbol':name})
                    note_symbol(sym.name,f['path']);file_blocks+=1
                pos=end;ci+=1
            if block_cap_hit or file_blocks>=file_block_cap:break
        del text,a,syms
        if block_cap_hit:break
    if not blocks:return Counter(),dict(diag)
    active_phase('graph',blocks=len(blocks),files=len(summary))

    edges=[];incoming=Counter();outgoing=Counter();resolved=Counter();edge_cap_hit=False
    def add_edge(src,dst,kind,confidence,detail=''):
        nonlocal edge_cap_hit
        if not src or not dst or src==dst:return False
        if len(edges)>=repo_edge_cap:
            if not edge_cap_hit:diag['repo_edge_cap_hits']+=1;edge_cap_hit=True
            return False
        edges.append({'src_file':src,'dst_file':dst,'kind':kind,'confidence':float(confidence),'detail':detail})
        outgoing[src]+=1;incoming[dst]+=1
        return True

    for fp,hints in imports.items():
        if not block_heads.get(fp):continue
        src=by_path.get(fp)
        for hint in hints:
            dst=path_idx.resolve(fp,src.get('language') if src else None,hint)
            if dst and giant and not block_heads.get(dst) and diag['giant_neighbor_contexts']<int(giant_neighbor_cap):
                df=by_path.get(dst)
                if df and not df.get('duplicate') and not df.get('generated_hint') and not df.get('is_meta'):
                    try:
                        dt=read_text(df['row']);de=min(len(dt),32000)
                        if de>0:
                            ref={'format':df['locator']['format'],'pack_path':df['locator']['pack_path'],'offset':df['locator']['offset'],'start':0,'end':de,'file_path':df['path'],'language':df['language'],'symbol':PurePosixPath(df['path']).name,'kind':'file_context','approx_tokens':_estimate_tokens(dt[:de],df['bytes_per_token'])}
                            block_heads[dst].append({'block_id':stable_u64(f'ctx|{repo_key}|{dst}'),'ref':ref,'symbol':PurePosixPath(df['path']).name});diag['giant_neighbor_contexts']+=1
                        del dt
                    except Exception:diag['giant_neighbor_read_error']+=1
            if dst and block_heads.get(dst):
                if add_edge(fp,dst,'import',1.0,hint):resolved[fp]+=1
                if src and src.get('is_test'):add_edge(fp,dst,'test',0.9,hint)
            if edge_cap_hit:break
        if edge_cap_hit:break

    if not edge_cap_hit:
        for fp,info in summary.items():
            for name in info['calls']:
                dst=symbol_owner.get(name)
                if dst and dst!=fp:add_edge(fp,dst,'call',0.7,name)
                if edge_cap_hit:break
            if edge_cap_hit:break
    if not edge_cap_hit:
        for e in _same_stem_relation_edges(by_path,block_heads):
            add_edge(e['src_file'],e['dst_file'],e['kind'],e['confidence'],e.get('detail',''))
            if edge_cap_hit:break
    if not edge_cap_hit:
        for fp,names in bases.items():
            for name in names:
                dst=symbol_owner.get(name)
                if dst and dst!=fp:add_edge(fp,dst,'inherits',0.88,name)
                if edge_cap_hit:break
            if edge_cap_hit:break

    active_phase('score_and_write_blocks',blocks=len(blocks),edges=len(edges))
    parsed=sum(float(x.get('structure_confidence',0)) for x in summary.values());cf=max(1,len(summary));tests=sum(1 for f in targets if f['is_test']);meta=len(meta_files)
    import_total=sum(len(v) for v in imports.values());resolved_total=sum(resolved.values());symbol_total=sum(x['symbol_count'] for x in summary.values())
    repo_quality=max(0.0,min(1.0,0.28*(parsed/cf)+0.16*min(1,tests/max(1,cf*.2))+0.10*min(1,meta/2)+0.22*(resolved_total/max(1,import_total) if import_total else .35)+0.14*min(1,symbol_total/max(1,cf*5))+0.10*min(1,math.log2(max(1,cf))/4)))
    for b in blocks:
        fp=b['file_path'];info=summary.get(fp,{});imports_n=len(imports.get(fp,[]))
        sem=max(0,min(1,0.25*(1 if info.get('parse_ok') else .3)+0.2*min(1,float(info.get('symbol_count',0))/5)+0.15*(resolved[fp]/max(1,imports_n) if imports_n else .3)+0.2*min(1,(incoming[fp]+outgoing[fp])/6)+0.2*(1 if is_test_path(fp) else 0)))
        b['semantic']=sem;b['quality']=max(0,min(1,.72*float(b['quality'])+.28*repo_quality));bw.write(b)

    active_phase('build_file_relations',blocks=len(blocks),edges=len(edges))
    file_rel=_build_bounded_file_relations(edges,file_relation_cap)
    tree=[str(x.get('file_path') or x.get('path') or '') if isinstance(x,dict) else str(x) for x in (manifest.get('tree') or [])]
    def meta_priority(f):
        n=PurePosixPath(f['path']).name.lower()
        if n.startswith('readme'):return 0
        if n in {'pyproject.toml','package.json','cargo.toml','go.mod','pom.xml','build.gradle','build.gradle.kts','cmakelists.txt'}:return 1
        if n.startswith('license'):return 2
        return 3
    meta_refs=[]
    for mf in sorted(meta_files,key=meta_priority)[:3]:
        try:
            mt=read_text(mf['row']);me=min(len(mt),32768);del mt
            meta_refs.append({'format':mf['locator']['format'],'pack_path':mf['locator']['pack_path'],'offset':mf['locator']['offset'],'start':0,'end':me,'file_path':mf['path'],'language':mf.get('language'),'symbol':'','kind':'meta'})
        except Exception:pass
    repo_off=rw.write({'repo_key':repo_key,'repo_path':manifest.get('repo_path',repo_key),'tree':tree[:500],'meta':meta_refs,'giant_repo':giant,'original_file_count':original_files,'retained_file_rows':len(desc)});repo_ref={'path':str(Path(rw.path).resolve()),'offset':repo_off}

    active_phase('write_candidates',blocks=len(blocks),edges=len(edges),relation_files=len(file_rel))
    counts=Counter()
    for b in blocks:
        lang=b['primary_language'];fp=b['file_path'];rels=[];seen_rel=set()
        for fr in file_rel.get(fp,[]):
            for other in block_heads.get(fr['other_file'],[])[:2]:
                ref=other['ref'];key=(fr['role'],ref.get('pack_path'),int(ref.get('offset',0)),int(ref.get('start',0)),int(ref.get('end',0)))
                if key in seen_rel:continue
                seen_rel.add(key);rels.append({'role':fr['role'],'score':fr['score'],'ref':ref,'symbol':other.get('symbol','')})
        rels=sorted(rels,key=lambda x:(-float(x['score']),x['role'],str(x['ref'].get('file_path',''))))[:max_related]
        tasks=['single_causal','repo_causal']
        if b.get('fim_spans'):tasks+=['ast_fim','semantic_fim']
        if b.get('fim_spans') and any(x['role'] in {'test','tested_by'} for x in rels):tasks.append('test_aware')
        cand={'candidate_id':stable_u64(f"cand|{b['block_id']}"),'source':'code','primary_language':lang,'group_id':repo_key,
              'approx_tokens':int(b['approx_tokens']+sum(int(r['ref'].get('approx_tokens',max(1,(r['ref']['end']-r['ref']['start'])//4))) for r in rels[:4])),'quality':float(b['quality']),'semantic':float(b['semantic']),
              'tasks':list(dict.fromkeys(tasks)),'target':b['ref'],'fim_spans':b.get('fim_spans') or [],'related':rels,'repo_ref':repo_ref,'tree_local':neighborhoods.get(fp,[]) if giant else [],'symbol':b.get('symbol',''),'backend':b.get('backend'),'parse_ok':bool(b.get('parse_ok'))}
        cp=Path(out_root)/'candidates'/'code'/lang/f'p{part:05d}.mpk'
        if lang not in writers:writers[lang]=FramedWriter(cp)
        writers[lang].write(cand);counts[lang]+=1
    return counts,dict(diag)

def _semantic_code_partition(args):
    import faulthandler
    if os.environ.get('SBP_NATIVE_FAULTHANDLER')=='1':
        try:faulthandler.enable(all_threads=True)
        except Exception:pass
    part,files,out_root,drop_file,disable_tree_sitter,max_related,repo_buckets,max_parse_bytes,max_code_file_bytes,ts_timeout_s,ts_crash_threshold,ts_parser_cache,ts_max_nodes,giant_threshold,giant_row_cap,giant_target_cap,giant_neighbor_cap,tree_summary_lines,target_tree_paths,parser_mode=args
    out_root=Path(out_root);_cleanup_code_partition_outputs(out_root,part)
    drop_lookup=_DropLookup(Path(drop_file).parent,legacy_source='code');writers={};counts=Counter();diag=Counter();repos=0
    if parser_mode=='direct':analyzer=_get_worker_direct_analyzer(False,max_parse_bytes,ts_parser_cache,ts_max_nodes)
    elif parser_mode=='isolated':analyzer=_get_worker_ts_client(out_root,False,max_parse_bytes,ts_timeout_s,ts_crash_threshold,ts_parser_cache,ts_max_nodes)
    else:analyzer=_get_worker_direct_analyzer(True,max_parse_bytes,ts_parser_cache,ts_max_nodes)
    block_path=out_root/'blocks'/'code'/f'p{part:05d}.mpk';repo_store_path=out_root/'repos'/'code'/f'p{part:05d}.mpk'
    active=out_root/'code_active'/f'p{part:05d}.json'
    with FramedWriter(block_path) as bw,FramedWriter(repo_store_path) as rw:
        try:
            for repo_key,rows in _iter_repo_groups_bounded(files,out_root,part,repo_buckets,giant_threshold,giant_row_cap,tree_summary_lines):
                mark_active=len(rows)>=max(1,int(os.environ.get('SBP_CODE_ACTIVE_MIN_ROWS','256')))
                if mark_active:
                    atomic_json(active,{'partition':int(part),'repo_key':str(repo_key),'rows':len(rows),'pid':os.getpid(),'started_at_unix':time.time()})
                manifest=next((r for r in rows if r.get('kind')=='repo'),{'repo_key':repo_key,'repo_path':repo_key,'tree':[]})
                c,d=_process_code_repo(repo_key,rows,manifest,drop_lookup,analyzer,max_related,max_parse_bytes,max_code_file_bytes,bw,rw,writers,out_root,part,giant_threshold,giant_target_cap,giant_neighbor_cap,target_tree_paths)
                if c:
                    counts.update(c);repos+=1
                diag.update(d)
                rows.clear()
                if mark_active:
                    try:active.unlink()
                    except FileNotFoundError:pass
        finally:
            for w in writers.values():w.close()
            drop_lookup.close()
            try:active.unlink()
            except FileNotFoundError:pass
    result={'partition':part,'repos':repos,'candidates':dict(counts),'blocks':sum(counts.values()),'diagnostics':dict(diag),'tree_sitter':parser_mode!='regex','parser_mode':parser_mode}
    atomic_json(out_root/'code_done'/f'p{part:05d}.json',result)
    return result

def _valid_framed_file(path:Path|str,magic:bytes=MAGIC)->bool:
    path=Path(path)
    try:
        if not path.is_file() or path.stat().st_size < len(magic):return False
        with path.open('rb') as f:return f.read(len(magic))==magic
    except OSError:return False


def _generic_task_paths(task):
    shard,out_root,_chunk_tokens=task;shard=Path(shard);out_root=Path(out_root);source=shard.parent.name
    return out_root/'blocks'/source/shard.name,out_root/'candidates'/source/shard.name


def _generic_task_complete(task)->bool:
    bp,cp=_generic_task_paths(task)
    return _valid_framed_file(bp) and _valid_framed_file(cp)


def _cleanup_generic_task_outputs(task)->None:
    bp,cp=_generic_task_paths(task)
    for target in (bp,cp):
        try:target.unlink()
        except FileNotFoundError:pass
        if target.parent.exists():
            for q in target.parent.glob(target.name+'.tmp.*'):
                try:q.unlink()
                except FileNotFoundError:pass


def _validate_code_partition_checkpoint(out_root:Path|str,part:int):
    out_root=Path(out_root);part=int(part);done=out_root/'code_done'/f'p{part:05d}.json'
    try:
        rec=json.loads(done.read_text(encoding='utf-8'))
    except Exception:return False,None
    if int(rec.get('partition',-1))!=part:return False,None
    if not _valid_framed_file(out_root/'blocks'/'code'/f'p{part:05d}.mpk'):return False,None
    if not _valid_framed_file(out_root/'repos'/'code'/f'p{part:05d}.mpk'):return False,None
    for lang,n in (rec.get('candidates') or {}).items():
        if int(n or 0)>0 and not _valid_framed_file(out_root/'candidates'/'code'/str(lang)/f'p{part:05d}.mpk'):
            return False,None
    return True,rec


def _valid_code_checkpoints(out_root:Path|str,parts=None,cleanup_invalid:bool=False):
    out_root=Path(out_root);allowed=None if parts is None else {int(x) for x in parts};vals={}
    done_dir=out_root/'code_done'
    if not done_dir.exists():return vals
    for q in sorted(done_dir.glob('p*.json')):
        try:part=int(q.stem[1:])
        except Exception:continue
        if allowed is not None and part not in allowed:continue
        ok,rec=_validate_code_partition_checkpoint(out_root,part)
        if ok:vals[part]=rec
        elif cleanup_invalid:_cleanup_code_partition_outputs(out_root,part)
    return vals


def _route_partition_files(run_root:Path|str):
    run_root=Path(run_root);out={}
    if not run_root.exists():return out
    for p in sorted(run_root.glob('p*.mpk')):
        try:part=_partition_from_run_name(p)
        except Exception:continue
        if _valid_framed_file(p):out[int(part)]=p
    return out


def _can_adopt_legacy_code_route(run_root:Path|str,parts:int,completed_code_parts:int)->bool:
    # Legacy v8 had no route-done marker. A valid code_done partition proves that
    # generic semantic work and the global code router both returned successfully
    # before code-index started. At 250B all configured route partitions are populated.
    if int(completed_code_parts)<=0:return False
    files=_route_partition_files(run_root)
    return set(files)==set(range(int(parts)))


def _semantic_resume_guard(out:Path,resume_signature)->None:
    marker=out/'semantic_resume_state.json'
    sig=None if resume_signature is None else str(resume_signature)
    if marker.exists():
        try:old=json.loads(marker.read_text(encoding='utf-8')).get('signature')
        except Exception as e:raise RuntimeError(f'cannot read semantic resume state {marker}: {e}') from e
        if old!=sig:
            raise RuntimeError(f'semantic partial state belongs to a different input/config signature: {old!r} != {sig!r}; use --rebuild or a new work-dir')
        return
    atomic_json(marker,{'signature':sig,'created_at_unix':time.time(),'legacy_adoption':any(out.iterdir())})


def run_semantic(inventory_dir:Path|str,dedup_dir:Path|str,out_dir:Path|str,workers:int=16,code_workers:int=12,repo_partitions:int=256,repo_buckets:int=32,
                 chunk_tokens:int=8192,max_related:int=8,disable_tree_sitter:bool=False,max_parse_bytes:int=2<<20,max_code_file_bytes:int=8<<20,
                 ts_timeout_s:float=15.0,ts_crash_threshold:int=2,ts_parser_cache:int=8,ts_max_nodes:int=250000,giant_threshold:int=20000,giant_row_cap:int=12000,giant_target_cap:int=4096,giant_neighbor_cap:int=512,tree_summary_lines:int=500,target_tree_paths:int=24,rebuild:bool=False,resume_signature=None)->Dict[str,Any]:
    from concurrent.futures.process import BrokenProcessPool
    inv=Path(inventory_dir).resolve();dd=Path(dedup_dir).resolve();out=Path(out_dir).resolve()
    if rebuild and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);_semantic_resume_guard(out,resume_signature);t0=time.perf_counter()

    # 1/3 generic documents: survivor locators are repacked into pack-ordered
    # buckets. Hash-partitioned shards would otherwise touch almost every pack
    # in the corpus with ~20KB reads, which the device serves at random-read
    # latency (~1ms) instead of streaming forwards.
    shards=sorted((dd/'survivors').rglob('*.mpk'));generic_tasks=[(p,out,chunk_tokens) for p in shards if p.parent.name!='code']
    generic_marker=out/'generic_done.json';source_counts=Counter();generic_counts_known=False
    buckets=max(1,int(os.environ.get('SBP_GENERIC_BUCKETS') or 1024))
    prev={}
    if generic_marker.exists():
        try:prev=json.loads(generic_marker.read_text(encoding='utf-8')) or {}
        except Exception:prev={}
    done_dir=out/'_bucket_done'
    def _buckets_complete(n:int)->bool:
        return done_dir.is_dir() and all((done_dir/f'b{i:05d}.json').is_file() for i in range(int(n)))
    marker_complete=(prev.get('mode')=='pack_buckets' and int(prev.get('tasks',-1))==len(generic_tasks)
                     and int(prev.get('buckets') or 0)==buckets and _buckets_complete(buckets))
    adopted_generic=[t for t in generic_tasks if _generic_task_complete(t)]
    pending_generic=[] if marker_complete else [t for t in generic_tasks if not _generic_task_complete(t)]
    if marker_complete:
        if isinstance(prev.get('source_counts'),dict):source_counts.update({str(k):int(v) for k,v in prev['source_counts'].items()})
        generic_counts_known=bool(prev.get('counts_known'))
        print(f'[semantic 1/3 documents] RESUME pack-bucket outputs complete buckets={buckets}',flush=True)
    elif not adopted_generic:
        generic_counts_known=True
    for t in pending_generic:_cleanup_generic_task_outputs(t)
    _raw_gw=os.environ.get('SBP_GENERIC_WORKERS')
    _gw=max(1,int(_raw_gw)) if _raw_gw not in (None,'') else int(workers)
    generic_workers=max(1,min(_gw,len(pending_generic) or 1,max(1,os.cpu_count() or 8)))
    print(f'[semantic 1/3 documents] shards={len(generic_tasks)} adopted={len(adopted_generic)} pending={len(pending_generic)} mode=pack_buckets buckets={buckets} workers={generic_workers}',flush=True)
    if pending_generic:
        index_dir=_generic_index_dir(out);idx_marker=index_dir/'index_done.json'
        idx_sig=_generic_index_signature([t[0] for t in pending_generic],buckets)
        stale=False
        if idx_marker.is_file():
            try:stale=str(json.loads(idx_marker.read_text(encoding='utf-8')).get('signature'))!=idx_sig
            except Exception:stale=True
        if stale:
            print('[semantic 1/3 index] INVALIDATE stale pack index and bucket outputs',flush=True)
            for root in ('blocks','candidates'):
                for q in list((out/root).glob('*/b*.mpk'))+list((out/root).glob('*/b*.mpk.tmp.*')):
                    try:q.unlink()
                    except FileNotFoundError:pass
            shutil.rmtree(done_dir,ignore_errors=True)
        _ensure_generic_locator_index([t[0] for t in pending_generic],buckets,index_dir)
        bt=[(index_dir/f'bucket_{i:05d}.mpk',out,chunk_tokens,i) for i in range(buckets) if not (done_dir/f'b{i:05d}.json').is_file()]
        print(f'[semantic 1/3 pack-scan] pending_buckets={len(bt)}/{buckets}',flush=True)
        if bt:
            if generic_workers<=1:
                for t in bt:
                    res=_generic_pack_bucket(t);source_counts.update(res['candidates'])
            else:
                for _t,res in _bounded_process_results(bt,_generic_pack_bucket,generic_workers,max_in_flight=max(2,generic_workers*2),mp_context=process_context()):
                    source_counts.update(res['candidates'])
        for t in adopted_generic:
            if not _generic_task_complete(t):raise RuntimeError('generic semantic resume validation failed: adopted shard outputs are incomplete')
        for i in range(buckets):
            if not (done_dir/f'b{i:05d}.json').is_file():raise RuntimeError(f'generic pack-bucket semantic incomplete: bucket {i}')
        shutil.rmtree(index_dir,ignore_errors=True)
        print('[semantic 1/3 pack-scan] COMPLETE locator index removed',flush=True)
    atomic_json(generic_marker,{'mode':'pack_buckets','tasks':len(generic_tasks),'buckets':buckets,'adopted':len(adopted_generic),
                                'source_counts':dict(source_counts) if generic_counts_known else None,'counts_known':generic_counts_known,'completed_at_unix':time.time()})
    shown=sum(source_counts.values()) if generic_counts_known else None
    print(f'[semantic 1/3 documents] COMPLETE candidates={shown if shown is not None else "legacy-adopted (count unavailable without rescanning)"}',flush=True)

    # Load valid code checkpoints before route adoption. Legacy v8 can be safely
    # adopted when at least one valid code_done exists: code-index only starts after
    # the global route function returned.
    done_dir=out/'code_done';done_dir.mkdir(parents=True,exist_ok=True)
    preexisting_done=_valid_code_checkpoints(out,cleanup_invalid=True)
    code_shards=sorted((inv/'code').glob('*.mpk'));run_root=out/'repo_runs';route_marker=out/'code_route_done.json'
    part_files={};routed_code_rows=None;route_adopted=False
    if route_marker.exists():
        try:
            rm=json.loads(route_marker.read_text(encoding='utf-8'));expected={int(x) for x in (rm.get('populated_parts') or [])};files=_route_partition_files(run_root)
            if int(rm.get('repo_partitions',-1))==int(repo_partitions) and expected and set(files)==expected:
                part_files={i:[files[i]] for i in sorted(expected)};routed_code_rows=rm.get('rows');route_adopted=True
        except Exception:pass
    if not route_adopted and _can_adopt_legacy_code_route(run_root,repo_partitions,len(preexisting_done)):
        files=_route_partition_files(run_root);part_files={i:[files[i]] for i in sorted(files)};route_adopted=True
        atomic_json(route_marker,{'repo_partitions':int(repo_partitions),'populated_parts':sorted(files),'rows':None,'legacy_adopted':True,'completed_at_unix':time.time()})
    if route_adopted:
        print(f'[semantic 2/3 code-route] RESUME adopted populated_partitions={len(part_files)} rows={routed_code_rows if routed_code_rows is not None else "legacy-unknown"}',flush=True)
    else:
        print(f'[semantic 2/3 code-route] inventory_shards={len(code_shards)} repo_partitions={repo_partitions} global_streaming=1',flush=True)
        routed_code_rows=_route_code_global(code_shards,run_root,repo_partitions)
        files=_route_partition_files(run_root);part_files={i:[files[i]] for i in sorted(files)}
        atomic_json(route_marker,{'repo_partitions':int(repo_partitions),'populated_parts':sorted(files),'rows':int(routed_code_rows),'legacy_adopted':False,'completed_at_unix':time.time()})
        print(f'[semantic 2/3 code-route] COMPLETE rows={routed_code_rows:,} populated_partitions={len(part_files)}',flush=True)
    if not part_files:raise RuntimeError('code route produced no populated partitions')

    primary_mode='regex' if disable_tree_sitter else 'direct';drop_file=dd/'drop_ids.mpk'
    ctasks=[(i,part_files[i],out,drop_file,disable_tree_sitter,max_related,repo_buckets,max_parse_bytes,max_code_file_bytes,ts_timeout_s,ts_crash_threshold,ts_parser_cache,ts_max_nodes,giant_threshold,giant_row_cap,giant_target_cap,giant_neighbor_cap,tree_summary_lines,target_tree_paths,primary_mode) for i in sorted(part_files)]
    priority=[]
    for tok in os.environ.get('SBP_CODE_PRIORITY_PARTITIONS','').replace(',', ' ').split():
        try:priority.append(int(tok))
        except ValueError:raise RuntimeError(f'invalid SBP_CODE_PRIORITY_PARTITIONS token: {tok!r}')
    pr={p:i for i,p in enumerate(priority)}
    ctasks.sort(key=lambda t:((0,pr[int(t[0])]) if int(t[0]) in pr else (1,-sum(Path(p).stat().st_size for p in t[1]))))
    valid_done=_valid_code_checkpoints(out,parts=part_files,cleanup_invalid=True);pending=[t for t in ctasks if int(t[0]) not in valid_done]
    _raw_cw=os.environ.get('SBP_CODE_INDEX_WORKERS')
    if _raw_cw not in (None,''):
        _cw_cap=max(1,int(_raw_cw))
    else:
        _cw_cap=max(1,min(os.cpu_count() or 8,int(_mem_available_gib()*0.5)//2))
    code_workers=max(1,min(max(int(code_workers),int(_raw_cw or 0)),_cw_cap,len(pending) or 1))
    print(f'[semantic 3/3 code-index] partitions={len(ctasks)} resume_ok={len(valid_done)} pending={len(pending)} workers={code_workers} parser={primary_mode}_spawn tree_sitter={not disable_tree_sitter} repo_grouping=sqlite_external repo_block_cap={int(os.environ.get('SBP_REPO_BLOCK_CAP','131072'))} file_block_cap={int(os.environ.get('SBP_FILE_BLOCK_CAP','4096'))} relation_mode=file_bounded priority={priority or 'none'} parse_cap={max_parse_bytes/(1<<20):.1f}MiB file_cap={max_code_file_bytes/(1<<20):.1f}MiB',flush=True)
    progress_last=len(valid_done)
    def done_parts():return _valid_code_checkpoints(out,parts=part_files,cleanup_invalid=False)
    def run_batch(batch,nworkers,ctx):
        nonlocal progress_last
        if not batch:return
        with ProcessPoolExecutor(max_workers=max(1,min(nworkers,len(batch))),mp_context=ctx) as ex:
            futs={ex.submit(_semantic_code_partition,t):t for t in batch}
            for fut in as_completed(futs):
                fut.result();d=len(done_parts())
                if d>progress_last and (d==len(ctasks) or d-progress_last>=max(1,len(ctasks)//20)):
                    progress_last=d;print(f'[semantic 3/3 code-index] progress={d}/{len(ctasks)}',flush=True)
    if pending:
        if code_workers<=1:
            for t in pending:_semantic_code_partition(t)
        else:
            try:run_batch(pending,code_workers,_code_primary_context(disable_tree_sitter))
            except BrokenProcessPool:
                complete=done_parts();pending2=[t for t in pending if int(t[0]) not in complete]
                print(f'[semantic 3/3 code-index] WARNING direct native parser pool crashed; completed={len(complete)} pending={len(pending2)}. Retrying only pending partitions with crash-isolated Tree-sitter.',flush=True)
                isolated=[(*t[:-1],'isolated') for t in pending2]
                try:run_batch(isolated,min(code_workers,len(isolated) or 1),process_context())
                except BrokenProcessPool:
                    complete=done_parts();pending3=[t for t in isolated if int(t[0]) not in complete]
                    print(f'[semantic 3/3 code-index] WARNING isolated pool terminated; retrying {len(pending3)} pending partitions with regex/Python-AST only.',flush=True)
                    regex=[(*t[:-1],'regex') for t in pending3];run_batch(regex,min(4,code_workers),mp.get_context('spawn'))
    completed=done_parts();code_stats=[completed[i] for i in sorted(completed) if i in part_files]
    missing=sorted(set(part_files)-set(completed))
    if missing:raise RuntimeError(f'code-index incomplete partitions: {missing[:20]}'+(' ...' if len(missing)>20 else ''))
    lang=Counter();diag=Counter()
    for r in code_stats:lang.update(r['candidates']);diag.update(r.get('diagnostics') or {})
    print(f'[semantic 3/3 code-index] COMPLETE repos={sum(r["repos"] for r in code_stats):,} code_candidates={sum(lang.values()):,} diagnostics={dict(diag)}',flush=True)
    quarantined=[];qdir=out/'ts_quarantine'
    if qdir.exists():
        for qp in sorted(qdir.glob('*.disabled.json')):
            try:quarantined.append(json.loads(qp.read_text(encoding='utf-8')))
            except Exception:pass
    report={'stage':'semantic','generic_candidates':dict(source_counts) if generic_counts_known else None,'generic_candidates_known':generic_counts_known,'code_candidates':dict(lang),'code_repos':sum(r['repos'] for r in code_stats),'workers':workers,'code_workers':code_workers,
            'repo_partitions':repo_partitions,'repo_buckets':repo_buckets,'max_parse_bytes':max_parse_bytes,'max_code_file_bytes':max_code_file_bytes,'giant_repo_files':giant_threshold,'giant_repo_row_cap':giant_row_cap,'giant_repo_target_cap':giant_target_cap,'giant_repo_neighbor_cap':giant_neighbor_cap,'tree_summary_lines':tree_summary_lines,'target_tree_paths':target_tree_paths,'ts_timeout_s':ts_timeout_s,'ts_crash_threshold':ts_crash_threshold,'ts_parser_cache':ts_parser_cache,'ts_max_nodes':ts_max_nodes,'ts_quarantined_languages':quarantined,'diagnostics':dict(diag),'resume':{'generic_adopted':len(adopted_generic),'code_route_adopted':route_adopted,'code_partitions_adopted':len(valid_done)},'elapsed_s':time.perf_counter()-t0}
    atomic_json(out/'semantic_report.json',report);success_marker(out,'semantic',report);return report


# ============================================================================
# BEGIN planner.py
# ============================================================================
from collections import defaultdict, Counter
import hashlib, math, shutil, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


def largest_remainder(total:int,shares:Mapping[str,float])->Dict[str,int]:
    if total<0:raise ValueError('total must be >=0')
    if not shares:return {}
    s=sum(max(0,float(v)) for v in shares.values())
    if s<=0:raise ValueError('shares must have positive mass')
    raw={k:total*max(0,float(v))/s for k,v in shares.items()};out={k:int(math.floor(v)) for k,v in raw.items()};remain=total-sum(out.values())
    order=sorted(raw,key=lambda k:(-(raw[k]-out[k]),str(k)))
    for k in order[:remain]:out[k]+=1
    return out

def _stable_fraction(seed:int,key:str)->float:
    b=hashlib.blake2b(f'{seed}|{key}'.encode(),digest_size=8).digest();return int.from_bytes(b,'little')/2**64

def _candidate_score(c,seed):
    return .55*float(c.get('quality',0))+.35*float(c.get('semantic',0))+.10*_stable_fraction(seed,str(c.get('candidate_id')))

def _candidate_capabilities(c:Mapping[str,Any])->List[str]:
    source=str(c.get('source') or '')
    tasks=set(c.get('tasks') or [])
    if source=='code':
        caps=['causal','ordinary_fim']
        if 'ast_fim' in tasks and c.get('fim_spans'):caps.append('ast_fim')
        if 'semantic_fim' in tasks and c.get('fim_spans') and c.get('related'):caps.append('semantic_fim')
        if 'test_aware' in tasks and c.get('fim_spans') and any(str(x.get('role')) in {'test','tested_by'} for x in (c.get('related') or [])):caps.append('test_aware_fim')
        return caps
    caps=['causal']
    if source.startswith('math_') and 'math_fim' in tasks and c.get('fim_indices'):caps.append('math_fim')
    return caps


def plan_candidates(candidates:Sequence[Mapping[str,Any]],*,target_tokens:int,source_shares:Mapping[str,float],language_ratios:Mapping[str,float],
                    code_sample_ratios:Optional[Mapping[str,float]]=None,overplan:float=1.06,max_repo_fraction:float=0.01,seed:int=20260830,math_fim_rate:float=0.25)->List[Dict[str,Any]]:
    """Select content only. FIM/causal mode is deliberately NOT frozen in preprocessing."""
    source_targets=largest_remainder(target_tokens,source_shares);plan=[]
    by_source=defaultdict(list)
    for c in candidates:by_source[str(c['source'])].append(dict(c))
    for source,target in source_targets.items():
        want=int(math.ceil(target*overplan))
        if source!='code':
            rows=sorted(by_source.get(source,[]),key=lambda c:(-_candidate_score(c,seed),int(c['candidate_id'])))
            used=0;group_used=Counter();group_cap=max(1,int(want*max_repo_fraction)) if max_repo_fraction>0 else 2**63-1
            for c in rows:
                n=max(1,int(c.get('approx_tokens',1)));gid=str(c.get('group_id',''))
                if group_used[gid]+n>max(group_cap,n):continue
                x=dict(c);x.pop('sample_type',None);x['capabilities']=_candidate_capabilities(c);x['plan_score']=_candidate_score(c,seed);plan.append(x)
                used+=n;group_used[gid]+=n
                if used>=want:break
        else:
            code_target=target;lang_targets=largest_remainder(code_target,language_ratios);by_lang=defaultdict(list)
            for c in by_source.get('code',[]):by_lang[str(c.get('primary_language'))].append(c)
            selected_ids=set();remaining_pool=[];code_used=0;repo_cap=max(1,int(code_target*max_repo_fraction)) if max_repo_fraction>0 else 2**63-1;repo_used=Counter()
            for lang,lt in lang_targets.items():
                want_lang=int(math.ceil(lt*overplan));rows=sorted(by_lang.get(lang,[]),key=lambda c:(-_candidate_score(c,seed),int(c['candidate_id'])));used=0
                for c in rows:
                    cid=int(c['candidate_id']);n=max(1,int(c.get('approx_tokens',1)));gid=str(c.get('group_id',''))
                    if repo_used[gid]+n>max(repo_cap,n):remaining_pool.append(c);continue
                    if used>=want_lang:remaining_pool.append(c);continue
                    x=dict(c);x.pop('sample_type',None);x['capabilities']=_candidate_capabilities(c);x['plan_score']=_candidate_score(c,seed);plan.append(x);selected_ids.add(cid)
                    used+=n;code_used+=n;repo_used[gid]+=n
                for c in rows:
                    if int(c['candidate_id']) not in selected_ids:remaining_pool.append(c)
            total_want=int(math.ceil(code_target*overplan))
            if code_used<total_want:
                uniq={int(c['candidate_id']):c for c in remaining_pool if int(c['candidate_id']) not in selected_ids}
                for c in sorted(uniq.values(),key=lambda c:(-_candidate_score(c,seed),int(c['candidate_id']))):
                    n=max(1,int(c.get('approx_tokens',1)));gid=str(c.get('group_id',''))
                    if repo_used[gid]+n>max(repo_cap,n):continue
                    x=dict(c);x.pop('sample_type',None);x['capabilities']=_candidate_capabilities(c);x['plan_score']=_candidate_score(c,seed);plan.append(x);selected_ids.add(int(c['candidate_id']))
                    code_used+=n;repo_used[gid]+=n
                    if code_used>=total_want:break
    return plan


def _selection_bucket(row:Mapping[str,Any])->str:
    return 'code/'+str(row.get('primary_language')) if str(row.get('source'))=='code' else str(row.get('source'))


def _selection_key(bucket:str,identifier:Any)->str:
    return str(bucket)+'\x1f'+_u64_key(identifier)


def _selected_score_map(conn,keys:Sequence[str])->Dict[str,float]:
    out={};keys=list(dict.fromkeys(keys))
    for i in range(0,len(keys),800):
        part=keys[i:i+800]
        if not part:continue
        q=','.join('?' for _ in part)
        for k,score in conn.execute(f'SELECT selection_key,score FROM selected WHERE selection_key IN ({q})',part):out[str(k)]=float(score)
    return out


def _plan_partition_key(c:Mapping[str,Any])->int:
    refs=c.get('parts') or []
    if refs:return stable_u64(str(refs[0].get('pack_path')))
    target=c.get('target') or {}
    if target.get('pack_path'):return stable_u64(str(target.get('pack_path')))
    return stable_u64(str(c.get('candidate_id')))


def _plan_db_path(out:Path)->Path:
    # HDD/NTFS random B-tree inserts are pathologically slow at 200M rows; use
    # the NVMe scratch dir when present (same failure mode as dedup sqlite).
    d=os.environ.get('SBP_PLAN_SQLITE_DIR')
    if d:return Path(d)/'.plan_selection.sqlite'
    alt=Path('/home/zjr1236/data/rebuild250b/plan_ssd')
    return (alt if alt.is_dir() else out)/'.plan_selection.sqlite'


def _trim_db_path(out:Path)->Path:
    d=os.environ.get('SBP_TRIM_SQLITE_DIR')
    if d:return Path(d)/'.trim_selection.sqlite'
    alt=Path('/home/zjr1236/data/rebuild250b/plan_ssd')
    return (alt if alt.is_dir() else out)/'.trim_selection.sqlite'


def _plan_write_worker(task):
    """Filter one file slice against the selection hashes and write part shards."""
    import gc
    import numpy as np
    from collections import Counter as _Counter
    wi,files,out_s,hash_path,min_quality,seed,plan_shards,progress_dir=task
    out=Path(out_s);prog=Path(progress_dir)/f'w{wi:02d}.json'
    hashes=np.load(hash_path,mmap_mode='r');nh=int(hashes.shape[0])
    # msgpack payloads are acyclic; disabling the collector keeps a long write
    # pass from rescanning every live container on each gen2 collection.
    gc.disable()
    writers={};planned=_Counter();planned_tokens=_Counter();caps=_Counter();written=0;seen=0;read_bytes=0
    def flush(batch):
        nonlocal written
        if not batch:return
        arr=np.fromiter((b[0] for b in batch),dtype=np.uint64,count=len(batch))
        pos=np.searchsorted(hashes,arr);np.minimum(pos,nh-1,out=pos)
        hit=hashes[pos]==arr
        for j in np.nonzero(hit)[0].tolist():
            _h,origin,c=batch[j]
            x=dict(c);x.pop('sample_type',None);x['capabilities']=_candidate_capabilities(c);x['plan_score']=_candidate_score(c,seed);x['provenance']=origin
            part=_plan_partition_key(x)%plan_shards
            w=writers.get(part)
            if w is None:w=writers[part]=FramedWriter(out/f'part-{part:05d}.w{wi:02d}.mpk')
            w.write(x)
            bucket=_selection_bucket(x);n=int(x.get('approx_tokens',0));planned[bucket]+=1;planned_tokens[bucket]+=n;written+=1
            for cap in x.get('capabilities') or []:caps[cap]+=n
        batch.clear()
    batch=[]
    for fi,(origin,pth) in enumerate(files,1):
        try:read_bytes+=int(Path(pth).stat().st_size)
        except OSError:pass
        for c in iter_frames(pth):
            seen+=1
            if float(c.get('quality',0))<min_quality:continue
            key=str(_selection_bucket(c))+'\x1f'+str(origin)+'\x1f'+_u64_key(int(c['candidate_id']))
            batch.append((stable_u64(key),str(origin),c))
            if len(batch)>=4096:flush(batch)
        flush(batch)
        if fi%20==0 or fi==len(files):
            atomic_json(prog,{'worker':wi,'files_done':fi,'files_total':len(files),'rows_seen':seen,'written':written,'bytes':read_bytes})
    for w in writers.values():w.close()
    gc.enable()
    return {'worker':wi,'files':len(files),'rows_seen':seen,'written':written,'planned':dict(planned),'planned_tokens':dict(planned_tokens),'caps':dict(caps),'bytes':read_bytes}


def run_plan(semantic_dir:Path|str|Sequence[Tuple[str,Path|str]],out_dir:Path|str,*,target_tokens:int,source_shares:Mapping[str,float],language_ratios:Mapping[str,float],code_sample_ratios:Optional[Mapping[str,float]]=None,
             overplan:float=1.06,max_repo_fraction:float=0.01,plan_shards:int=128,seed:int=20260830,math_fim_rate:float=0.25,min_quality:float=0.0,rebuild:bool=False,
             base_capacity_hint:Optional[Mapping[str,int]]=None,resume_signature:Optional[str]=None,resume_select_signature:Optional[str]=None)->Dict[str,Any]:
    """External-memory deterministic planner for 250B-scale corpora.

    Only compact ranking metadata enters SQLite. Candidate payloads stay in their
    immutable semantic shards and are scanned a second time after selection.
    """
    if isinstance(semantic_dir,(str,Path)):
        semantic_roots=[('single',Path(semantic_dir).resolve())]
    else:
        semantic_roots=[(str(name),Path(root).resolve()) for name,root in semantic_dir]
    if not semantic_roots:raise RuntimeError('planner requires at least one semantic root')
    for name,root in semantic_roots:
        if not (root/'candidates').is_dir():raise RuntimeError(f'planner semantic candidates missing for {name}: {root}')
    out=Path(out_dir).resolve()
    if rebuild and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter()
    plan_meta=out/'.plan_resume.json';dbp=_plan_db_path(out)
    scan_sig=None if resume_signature is None else str(resume_signature)
    sel_sig=str(resume_select_signature) if resume_select_signature is not None else scan_sig
    plan_db_preserved=False;resume_select=False
    if scan_sig is not None:
        pm={}
        if plan_meta.is_file():
            try:pm=json.loads(plan_meta.read_text(encoding='utf-8'))
            except Exception:pm={}
        if pm.get('signature')==scan_sig and dbp.is_file():
            plan_db_preserved=True
            resume_select=bool(pm.get('select_done')) and str(pm.get('select_signature') or '')==str(sel_sig)
            print(f'[plan] RESUME scan_signature={scan_sig} select_done={resume_select}',flush=True)
            if pm.get('select_done') and not resume_select:
                print('[plan] selection parameters changed; reusing scanned candidates and re-selecting',flush=True)
        else:
            if pm or dbp.exists() or any(out.glob('part-*.mpk')):
                print('[plan] INVALIDATE stale partial plan (scan signature changed)',flush=True)
            for q in out.glob('part-*.mpk'):q.unlink()
            for q in out.glob('part-*.mpk.tmp.*'):q.unlink()
            for suffix in ('','-wal','-shm'):
                try:(out/f'.plan_selection.sqlite{suffix}').unlink()
                except FileNotFoundError:pass
            atomic_json(plan_meta,{'signature':scan_sig,'select_signature':sel_sig,'scan_done':False,'select_done':False})
    else:
        try:dbp.unlink()
        except FileNotFoundError:pass

    # Fast necessary-condition check for incremental runs. A validated base hint
    # contains exact capacities from a previous full base planner scan. We only
    # rescan the small augmentation semantic tree here. If any known base
    # shortfall remains negative after augmentation, failing now is exact and
    # avoids rebuilding the 61+ GiB base ranking DB. If all known shortfalls are
    # repaired, the normal full combined scan below still runs and is authoritative.
    if base_capacity_hint and any(name=='aug' for name,_ in semantic_roots):
        source_targets_pre=largest_remainder(target_tokens,source_shares);bucket_targets_pre={}
        for source,target in source_targets_pre.items():
            if source!='code':bucket_targets_pre[source]=int(math.ceil(target*overplan))
        code_target_pre=int(source_targets_pre.get('code',0));lang_targets_pre=largest_remainder(code_target_pre,language_ratios) if code_target_pre else {}
        for lang,target in lang_targets_pre.items():bucket_targets_pre['code/'+lang]=int(math.ceil(target*overplan))
        aug_root=next(root for name,root in semantic_roots if name=='aug')
        aug_avail=defaultdict(lambda:{'candidates':0,'approx_tokens':0})
        for pth in sorted((aug_root/'candidates').rglob('*.mpk')):
            for c in iter_frames(pth):
                if float(c.get('quality',0))<min_quality:continue
                bucket=_selection_bucket(c);n=max(1,int(c.get('approx_tokens',1)))
                aug_avail[bucket]['candidates']+=1;aug_avail[bucket]['approx_tokens']+=n
        pre={}
        for bucket,base_cap_raw in base_capacity_hint.items():
            if bucket not in bucket_targets_pre:continue
            base_cap=int(base_cap_raw);aug_cap=int(aug_avail.get(bucket,{}).get('approx_tokens',0));required=int(bucket_targets_pre[bucket]);combined=base_cap+aug_cap
            pre[bucket]={'base_capacity':base_cap,'aug_capacity':aug_cap,'combined':combined,'required':required,'margin':combined-required}
        atomic_json(out/'capacity_report.json',{'stage':'plan_capacity_precheck','target_tokens':int(target_tokens),'overplan':float(overplan),
            'base_capacity_hint_used':True,'partial_known_base_shortfalls':True,'buckets':pre})
        short=[f'{bucket}: need~{x["required"]:,}, combined~{x["combined"]:,}, margin={x["margin"]:,}' for bucket,x in pre.items() if int(x['margin'])<0]
        if short:
            raise RuntimeError('plan capacity shortfall from validated base capacity hint: '+'; '.join(short))
        print('[plan] validated base shortfall hint fully repaired; proceeding to authoritative full combined scan',flush=True)

    all_paths=[(origin,p) for origin,root in semantic_roots for p in sorted((root/'candidates').rglob('*.mpk'))]
    conn=_open_plan_sqlite(dbp)
    conn.execute('CREATE TABLE IF NOT EXISTS candidates(bucket TEXT NOT NULL,score REAL NOT NULL,selection_key TEXT PRIMARY KEY,group_id TEXT NOT NULL,tokens INTEGER NOT NULL,origin TEXT NOT NULL) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS selected(selection_key TEXT PRIMARY KEY,score REAL NOT NULL) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS group_usage(scope TEXT NOT NULL,group_id TEXT NOT NULL,tokens INTEGER NOT NULL,PRIMARY KEY(scope,group_id)) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS scanned(file TEXT PRIMARY KEY) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS availability(bucket TEXT NOT NULL,origin TEXT NOT NULL,candidates INTEGER NOT NULL,tokens INTEGER NOT NULL,PRIMARY KEY(bucket,origin)) WITHOUT ROWID')
    conn.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT) WITHOUT ROWID')
    conn.execute("INSERT OR REPLACE INTO meta VALUES('signature',?)",(str(resume_signature or ''),))
    scanned={str(r[0]) for r in conn.execute('SELECT file FROM scanned')} if plan_db_preserved else set()
    pending_paths=[(origin,p) for origin,p in all_paths if str(p) not in scanned]
    print(f'[plan] scan files_total={len(all_paths)} resume_files={len(scanned)} pending_files={len(pending_paths)}',flush=True)
    new_candidates=0
    for fi,(origin,pth) in enumerate(pending_paths,1):
        batch=[];file_avail=defaultdict(lambda:[0,0])
        conn.execute('BEGIN')
        for c in iter_frames(pth):
            if float(c.get('quality',0))<min_quality:continue
            bucket=_selection_bucket(c);cid=int(c['candidate_id']);n=max(1,int(c.get('approx_tokens',1)));score=_candidate_score(c,seed);gid=str(c.get('group_id',''))
            skey=str(bucket)+'\x1f'+str(origin)+'\x1f'+_u64_key(cid)
            batch.append((bucket,float(score),skey,gid,int(n),str(origin)))
            file_avail[(bucket,str(origin))][0]+=1;file_avail[(bucket,str(origin))][1]+=n
            if len(batch)>=20000:
                try:conn.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?)',batch)
                except Exception as e:conn.rollback();conn.close();raise RuntimeError(f'duplicate candidate identity in planner origin={origin} file={pth}') from e
                batch.clear()
        if batch:
            try:conn.executemany('INSERT INTO candidates VALUES(?,?,?,?,?,?)',batch)
            except Exception as e:conn.rollback();conn.close();raise RuntimeError(f'duplicate candidate identity in planner origin={origin} file={pth}') from e
        for (bucket,orig),(cnt,tok) in file_avail.items():
            conn.execute('INSERT INTO availability VALUES(?,?,?,?) ON CONFLICT(bucket,origin) DO UPDATE SET candidates=candidates+excluded.candidates,tokens=tokens+excluded.tokens',(bucket,orig,int(cnt),int(tok)))
        conn.execute('INSERT OR IGNORE INTO scanned VALUES(?)',(str(pth),))
        conn.commit();new_candidates+=sum(v[0] for v in file_avail.values())
        if fi%50==0 or fi==len(pending_paths):print(f'[plan] scan files={fi}/{len(pending_paths)} candidates+={new_candidates:,}',flush=True)
    conn.execute('CREATE INDEX IF NOT EXISTS candidate_rank ON candidates(bucket,score DESC,selection_key ASC)')
    conn.execute("INSERT OR REPLACE INTO meta VALUES('scan_done','1')");conn.commit()
    if scan_sig is not None:atomic_json(plan_meta,{'signature':scan_sig,'select_signature':sel_sig,'scan_done':True,'select_done':bool(resume_select)})
    availability=defaultdict(lambda:{'candidates':0,'approx_tokens':0});availability_by_origin=defaultdict(lambda:defaultdict(lambda:{'candidates':0,'approx_tokens':0}));candidate_count=0
    for bucket,orig,cnt,tok in conn.execute('SELECT bucket,origin,candidates,tokens FROM availability'):
        availability[bucket]['candidates']+=int(cnt);availability[bucket]['approx_tokens']+=int(tok)
        availability_by_origin[orig][bucket]['candidates']+=int(cnt);availability_by_origin[orig][bucket]['approx_tokens']+=int(tok)
        candidate_count+=int(cnt)
    print(f'[plan] scan COMPLETE candidates={candidate_count:,} buckets={len(availability)}',flush=True)

    source_targets=largest_remainder(target_tokens,source_shares);bucket_targets={}
    for source,target in source_targets.items():
        if source!='code':bucket_targets[source]=int(math.ceil(target*overplan))
    code_target=int(source_targets.get('code',0));lang_targets=largest_remainder(code_target,language_ratios) if code_target else {}
    for lang,target in lang_targets.items():bucket_targets['code/'+lang]=int(math.ceil(target*overplan))
    capacity_report={}
    origins=[name for name,_root in semantic_roots]
    for bucket,want in bucket_targets.items():
        base_cap=int(availability_by_origin.get('base',{}).get(bucket,{}).get('approx_tokens',0))
        aug_cap=int(availability_by_origin.get('aug',{}).get(bucket,{}).get('approx_tokens',0))
        if origins==['single']:
            base_cap=int(availability.get(bucket,{}).get('approx_tokens',0));aug_cap=0
        combined=int(availability.get(bucket,{}).get('approx_tokens',0))
        capacity_report[bucket]={'base_capacity':base_cap,'aug_capacity':aug_cap,'combined':combined,'required':int(want),'margin':combined-int(want)}
    atomic_json(out/'capacity_report.json',{'stage':'plan_capacity','target_tokens':int(target_tokens),'overplan':float(overplan),'buckets':capacity_report})
    short=[f'{bucket}: need~{int(x["required"]):,}, available~{int(x["combined"]):,}' for bucket,x in capacity_report.items() if int(x['margin'])<0]
    if short:
        conn.close();raise RuntimeError('plan capacity shortfall before ranking: '+'; '.join(short))

    if resume_select:
        selected_count=int(conn.execute('SELECT COUNT(*) FROM selected').fetchone()[0])
        print(f'[plan] selection RESUME selected={selected_count:,}',flush=True)
    else:
        conn.execute('DELETE FROM selected');conn.execute('DELETE FROM group_usage');conn.commit()
        selected_count=0
        conn.execute('BEGIN')
        # repo usage is tracked in memory: per-row sqlite lookups+upserts over
        # ~100M rows cost hours on this DB. doc-level gids are unique and can
        # never reach group_cap, so only code buckets need the shared usage map.
        code_usage={};t_sel=time.perf_counter()
        for bi,(bucket,want) in enumerate(bucket_targets.items(),1):
            scope='code' if bucket.startswith('code/') else bucket
            base_target=code_target if scope=='code' else int(source_targets.get(bucket,want))
            group_cap=max(1,int(base_target*max_repo_fraction)) if (max_repo_fraction>0 and scope=='code') else 2**63-1
            usage=code_usage if scope=='code' else None
            used=0;seen=0;batch=[];t0=time.perf_counter()
            for score,key,gid,n in conn.execute('SELECT score,selection_key,group_id,tokens FROM candidates WHERE bucket=? ORDER BY score DESC,selection_key ASC',(bucket,)):
                n=int(n);seen+=1
                if usage is not None:
                    gu=usage.get(gid,0)
                    if gu+n>max(group_cap,n):continue
                    usage[gid]=gu+n
                batch.append((key,float(score)));used+=n;selected_count+=1
                if len(batch)>=20000:
                    conn.executemany('INSERT OR IGNORE INTO selected VALUES(?,?)',batch);batch.clear()
                if used>=want:break
            if batch:conn.executemany('INSERT OR IGNORE INTO selected VALUES(?,?)',batch)
            if used<want:
                conn.rollback();conn.close();raise RuntimeError(f'plan capacity shortfall after repo cap: {bucket}: need~{want:,}, selected~{used:,}')
            el=time.perf_counter()-t0
            print(f'[plan] select {bi}/{len(bucket_targets)} {bucket} seen={seen:,} tokens={used:,}/{want:,} repos={len(code_usage) if usage is not None else "-"} {el:.1f}s',flush=True)
        conn.commit()
        conn.execute("INSERT OR REPLACE INTO meta VALUES('select_done','1')");conn.commit()
        if scan_sig is not None:atomic_json(plan_meta,{'signature':scan_sig,'select_signature':sel_sig,'scan_done':True,'select_done':True})
        print(f'[plan] selection COMPLETE selected={selected_count:,}',flush=True)

    for q in out.glob('part-*.mpk'):q.unlink()
    for q in out.glob('part-*.mpk.tmp.*'):q.unlink()
    # 184M probes into `selected` cost hours of random B-tree reads, and a 25GB
    # Python string set makes every membership test a cache miss plus periodic GC
    # scans of 184M objects (measured ~2.7k rows/s). One sorted uint64 hash array
    # is 1.5GB, is never visited by the GC, and is shared by all write workers.
    import numpy as _np, heapq
    t_load=time.perf_counter()
    hash_path=_plan_db_path(out).parent/'plan_sel_hashes.npy'
    reuse=False
    if hash_path.is_file():
        try:hashes=_np.load(hash_path,mmap_mode='r');reuse=int(hashes.shape[0])==selected_count
        except Exception:reuse=False
    if reuse:
        print(f'[plan] write reusing selection hashes={hash_path}',flush=True)
    else:
        hashes=_np.fromiter((stable_u64(str(r[0])) for r in conn.execute('SELECT selection_key FROM selected')),dtype=_np.uint64,count=selected_count)
        if int(hashes.shape[0])!=selected_count:
            conn.close();raise RuntimeError(f'external planner selected table mismatch: rows={hashes.shape[0]:,} expected={selected_count:,}')
        hashes.sort();_np.save(hash_path,hashes)
    del hashes
    print(f'[plan] write selection hashes={selected_count:,} {time.perf_counter()-t_load:.1f}s',flush=True)
    write_workers=max(1,min(int(os.environ.get('SBP_PLAN_WRITE_WORKERS') or 32),len(all_paths)))
    sized=[]
    for origin,p in all_paths:
        try:sz=int(Path(p).stat().st_size)
        except OSError:sz=0
        sized.append((sz,str(origin),str(p)))
    sized.sort(key=lambda t:-t[0])
    heap=[(0,i) for i in range(write_workers)];heapq.heapify(heap)
    file_lists=[[] for _ in range(write_workers)]
    for sz,origin,p in sized:
        load,i=heapq.heappop(heap);file_lists[i].append((origin,p));heapq.heappush(heap,(load+sz,i))
    total_bytes=sum(t[0] for t in sized)
    progress_dir=out/'write_progress'
    if progress_dir.exists():shutil.rmtree(progress_dir)
    progress_dir.mkdir(parents=True,exist_ok=True)
    tasks=[(i,file_lists[i],str(out),str(hash_path),float(min_quality),int(seed),int(plan_shards),str(progress_dir)) for i in range(write_workers) if file_lists[i]]
    print(f'[plan] write START selected={selected_count:,} files={len(all_paths)} shards={plan_shards} workers={len(tasks)} read={total_bytes/1e9:.0f}GB',flush=True)
    planned=Counter();planned_tokens=Counter();caps=Counter();written=0
    stop=threading.Event();t_w=time.perf_counter()
    def _watch():
        while not stop.wait(60):
            fd=wr=by=0
            for q in progress_dir.glob('w*.json'):
                try:m=json.loads(q.read_text(encoding='utf-8'))
                except Exception:continue
                fd+=int(m.get('files_done',0));wr+=int(m.get('written',0));by+=int(m.get('bytes',0))
            el=time.perf_counter()-t_w
            print(f'[plan] write progress files={fd}/{len(all_paths)} written={wr:,}/{selected_count:,} read={by/1e9:.0f}/{total_bytes/1e9:.0f}GB {el:.0f}s ({by/max(el,1e-9)/1e6:.1f}MB/s)',flush=True)
    watcher=threading.Thread(target=_watch,daemon=True);watcher.start()
    try:
        for _t,res in _bounded_process_results(tasks,_plan_write_worker,len(tasks),max_in_flight=len(tasks)):
            written+=int(res['written']);planned.update(res['planned']);planned_tokens.update(res['planned_tokens']);caps.update(res['caps'])
            print(f'[plan] write worker {int(res["worker"])+1}/{len(tasks)} done files={res["files"]} written={res["written"]:,} read={res["bytes"]/1e9:.1f}GB {time.perf_counter()-t_w:.0f}s',flush=True)
    finally:
        stop.set();watcher.join(timeout=5)
        shutil.rmtree(progress_dir,ignore_errors=True)
    try:hash_path.unlink()
    except FileNotFoundError:pass
    if written!=selected_count:
        conn.close();raise RuntimeError(f'external planner selected/written mismatch: selected={selected_count} written={written}')
    print(f'[plan] write COMPLETE written={written:,}',flush=True)
    conn.close()
    for suffix in ('','-wal','-shm'):
        try:(out/f'.plan_selection.sqlite{suffix}').unlink()
        except FileNotFoundError:pass
    report={'stage':'plan','target_tokens':target_tokens,'candidate_count':candidate_count,'planned_samples':written,'availability':dict(availability),
            'availability_by_origin':{o:dict(v) for o,v in availability_by_origin.items()},'capacity_report':capacity_report,'semantic_roots':{o:str(r) for o,r in semantic_roots},
            'planned_counts':dict(planned),'planned_approx_tokens':dict(planned_tokens),'capability_approx_tokens':dict(caps),'fim_policy':'online_training','plan_shards':plan_shards,
            'selection_backend':'sqlite_external_combined','strict_code_language_targets':True,'no_oversampling':True,'elapsed_s':time.perf_counter()-t0}
    atomic_json(out/'plan_report.json',report);success_marker(out,'plan',report);return report



# ============================================================================
# BEGIN structured_render.py
# ============================================================================
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json, math, os, shutil, time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fim import (
    SCHEMA as STRUCT_SCHEMA,
    STRUCT_TRAIN_MAGIC,
    SPECIAL_TOKENS as STRUCT_SPECIAL_TOKENS,
    canonical_causal_text,
    iter_structured_trainpack,
    load_hf_tokenizer,
    validate_structured_record,
)

RENDER_MAGIC=b'SBREND2\n'


_REF_TEXT_CACHE={}
_REF_TEXT_CACHE_CHARS=0

def _cached_doc_text(rec:Mapping[str,Any])->str:
    global _REF_TEXT_CACHE_CHARS
    key=(str(rec['pack_path']),int(rec['offset']))
    text=_REF_TEXT_CACHE.get(key)
    if text is None:
        text=read_text(rec)
        cap=int(os.environ.get('SBP_REF_CACHE_CHARS','67108864'))
        if _REF_TEXT_CACHE_CHARS+len(text)>cap or len(_REF_TEXT_CACHE)>=4096:
            _REF_TEXT_CACHE.clear();_REF_TEXT_CACHE_CHARS=0
        _REF_TEXT_CACHE[key]=text;_REF_TEXT_CACHE_CHARS+=len(text)
    return text


def _ref_text(ref:Mapping[str,Any])->str:
    whole=_cached_doc_text({'format':ref['format'],'pack_path':ref['pack_path'],'offset':int(ref['offset'])});n=len(whole)
    a=int(ref.get('start',0));b=int(ref.get('end',n))
    # Stored spans can overrun after text truncation; clamp into the stored text so a stale
    # span yields an empty string (dropped by callers) instead of killing a multi-hour stage.
    a=max(0,min(a,n));b=max(a,min(b,n))
    return whole[a:b]


def _validated_fim_candidates(text:str,spans:Sequence[Mapping[str,Any]])->List[Dict[str,Any]]:
    out=[];seen=set()
    n=len(text)
    for i,sp in enumerate(spans or []):
        a=max(0,min(int(sp['start']),n));b=max(a,min(int(sp['end']),n));kind=str(sp.get('kind') or 'span')
        if a>=b:continue
        key=(a,b,kind)
        if key in seen:continue
        seen.add(key);length=b-a
        # Keep the score sidecar-only. It guides online selection, never enters training text.
        score=float(sp.get('score',1.0));score=max(0.0,min(1.0,score))
        out.append({'start':a,'end':b,'kind':kind,'score':score,'chars':length})
    return out


def _context_from_relation(rel:Mapping[str,Any])->Dict[str,Any]:
    ref=rel['ref'];return {'role':str(rel.get('role') or 'related'),'score':float(rel.get('score',0.0)),'path':str(ref.get('file_path') or ''),'symbol':str(ref.get('symbol') or rel.get('symbol') or ''),'language':ref.get('language'),'text':_ref_text(ref)}


def _structured_candidate(c:Mapping[str,Any],cfg:Mapping[str,Any],compact:bool=False)->Dict[str,Any]:
    source=str(c['source']);sample_id=stable_u64(f"structured|{c['candidate_id']}")
    if source!='code':
        parts=[]
        for r in c.get('parts') or []:
            parts.append({'kind':str(r.get('kind') or 'part'),'text':_ref_text(r)})
        rec={'schema':STRUCT_SCHEMA,'sample_id':sample_id,'candidate_id':int(c['candidate_id']),'source':source,'primary_language':None,'score':float(c.get('plan_score',0.0)),
             'capabilities':list(c.get('capabilities') or _candidate_capabilities(c)),
             'structured':{'kind':'document','parts':parts,'fim_candidate_indices':[int(x) for x in (c.get('fim_indices') or [])]}}
        validate_structured_record(rec);return rec
    target_text=_ref_text(c['target']);repo={}
    if c.get('repo_ref'):repo=read_frame_at(c['repo_ref']['path'],int(c['repo_ref']['offset']))
    related=list(c.get('related') or [])
    related.sort(key=lambda r:(-float(r.get('score',0.0)),str(r.get('role','')),str((r.get('ref') or {}).get('file_path',''))))
    context=[]
    if not compact:
        for mr in list(repo.get('meta') or [])[:2]:
            try:text=_ref_text(mr)
            except Exception:continue
            if text:context.append({'role':'repo_meta','score':0.55,'path':str(mr.get('file_path') or ''),'symbol':'','language':mr.get('language'),'text':text})
    rel_limit=1 if compact else int(cfg.get('max_related',8))
    for rel in related[:rel_limit]:
        item=_context_from_relation(rel)
        if item['text']:context.append(item)
    tree=[] if compact else _compose_tree_context(repo.get('tree') or [],c.get('tree_local') or [],int(cfg.get('tree_limit',200)))
    fim=_validated_fim_candidates(target_text,c.get('fim_spans') or [])
    caps=list(c.get('capabilities') or _candidate_capabilities(c))
    # Re-evaluate capabilities after context compaction so metadata never claims an impossible online mode.
    if not fim:caps=[x for x in caps if x not in {'ast_fim','semantic_fim','test_aware_fim'}]
    if not context:caps=[x for x in caps if x not in {'semantic_fim','test_aware_fim'}]
    if not any(x['role'] in {'test','tested_by'} for x in context):caps=[x for x in caps if x!='test_aware_fim']
    if 'causal' not in caps:caps.insert(0,'causal')
    if 'ordinary_fim' not in caps:caps.append('ordinary_fim')
    tref=c['target']
    rec={'schema':STRUCT_SCHEMA,'sample_id':sample_id,'candidate_id':int(c['candidate_id']),'source':'code','primary_language':c.get('primary_language'),'score':float(c.get('plan_score',0.0)),
         'capabilities':list(dict.fromkeys(caps)),
         'structured':{'kind':'code','repo':{'repo_key':repo.get('repo_key',c.get('group_id')),'repo_path':repo.get('repo_path',c.get('group_id','')),'tree':tree},'context':context,
                       'target':{'path':str(tref.get('file_path') or ''),'symbol':str(tref.get('symbol') or c.get('symbol') or ''),'language':c.get('primary_language'),'backend':c.get('backend'),'parse_ok':bool(c.get('parse_ok')),
                                 'text':target_text,'fim_candidates':fim}}}
    validate_structured_record(rec);return rec


class _TestEncoding:
    def __init__(self,ids):self.ids=ids
class _TestTokenizer:
    def __init__(self):self.special={s:i for i,s in enumerate(STRUCT_SPECIAL_TOKENS)}
    def token_to_id(self,s):return self.special.get(s)
    def encode(self,text,add_special_tokens=False):
        # Recognize embedded control tokens exactly like an AddedToken tokenizer, then byte-encode raw text.
        ids=[];i=0;ordered=sorted(self.special,key=len,reverse=True)
        while i<len(text):
            hit=None
            for tok in ordered:
                if text.startswith(tok,i):hit=tok;break
            if hit is not None:ids.append(self.special[hit]);i+=len(hit);continue
            b=text[i].encode('utf-8');ids.extend(1000+x for x in b);i+=1
        return _TestEncoding(ids)
    def encode_batch(self,texts,add_special_tokens=False):return [self.encode(x,add_special_tokens=add_special_tokens) for x in texts]


def _load_render_tokenizer(path,cfg):
    if cfg.get('_test_counter')=='byte':return _TestTokenizer()
    return load_hf_tokenizer(path)


def _render_plan_shard(args):
    plan_path,out_root,tokenizer_path,cfg=args;plan_path=Path(plan_path);out_root=Path(out_root);tok=_load_render_tokenizer(tokenizer_path,cfg);data=out_root/'data'/plan_path.name;idxp=out_root/'index'/plan_path.name
    stats=Counter();batch_size=int(cfg.get('tokenizer_batch_size',128));max_tokens=int(cfg.get('max_sample_tokens',16384));pending=[]
    with FramedWriter(data,magic=RENDER_MAGIC) as dw,FramedWriter(idxp) as iw:
        def flush():
            if not pending:return
            texts=[canonical_causal_text(x[1],max_tree_paths=int(cfg.get('tree_limit',200)),max_contexts=int(cfg.get('max_related',8)),validate=False) for x in pending]
            counts=[len(e.ids) for e in tok.encode_batch(texts,add_special_tokens=False)]
            for (c,rec),n in zip(pending,counts):
                if n>max_tokens:
                    try:rec=_structured_candidate(c,cfg,compact=True)
                    except ValueError as exc:
                        if 'reserved control token' in str(exc):stats['reserved_token_collision']+=1
                        else:
                            stats['invalid_sample']+=1
                            if stats['invalid_sample']<=5:print(f'[render] skip {plan_path.name} cid={c.get("candidate_id")}: {exc}',flush=True)
                        continue
                    except Exception as exc:
                        stats['invalid_sample']+=1
                        if stats['invalid_sample']<=5:print(f'[render] skip {plan_path.name} cid={c.get("candidate_id")}: {type(exc).__name__}: {exc}',flush=True)
                        continue
                    n=len(tok.encode(canonical_causal_text(rec,max_tree_paths=0,max_contexts=1,validate=False),add_special_tokens=False).ids)
                    if n>max_tokens:stats['too_long']+=1;continue
                rec=dict(rec);rec['tokens']=int(n);rec['base_tokens']=int(n)
                off=dw.write(rec);iw.write({'sample_id':int(rec['sample_id']),'sample_path':str(data.resolve()),'sample_offset':off,'source':rec['source'],'primary_language':rec.get('primary_language'),'tokens':int(n),'score':float(rec.get('score',0.0)),'capabilities':rec.get('capabilities') or []})
                stats['samples']+=1;stats['tokens']+=int(n)
                for cap in rec.get('capabilities') or []:stats[f'cap::{cap}']+=1
            pending.clear()
        for c in iter_frames(plan_path):
            try:rec=_structured_candidate(c,cfg,compact=False)
            except ValueError as exc:
                if 'reserved control token' in str(exc):stats['reserved_token_collision']+=1
                else:
                    stats['invalid_sample']+=1
                    if stats['invalid_sample']<=5:print(f'[render] skip {plan_path.name} cid={c.get("candidate_id")}: {exc}',flush=True)
                continue
            except Exception as exc:
                stats['invalid_sample']+=1
                if stats['invalid_sample']<=5:print(f'[render] skip {plan_path.name} cid={c.get("candidate_id")}: {type(exc).__name__}: {exc}',flush=True)
                continue
            pending.append((c,rec))
            if len(pending)>=batch_size:flush()
        flush()
    return dict(stats)


def run_render(plan_dir:Path|str,out_dir:Path|str,*,tokenizer_path:Path|str,workers:int=16,config:Optional[Mapping[str,Any]]=None,rebuild:bool=False,resume_signature:Optional[str]=None)->Dict[str,Any]:
    plan=Path(plan_dir).resolve();out=Path(out_dir).resolve();cfg=dict(config or {})
    if rebuild and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    sigp=out/'.render_resume.json'
    if resume_signature is not None:
        prev=None
        if sigp.is_file():
            try:prev=json.loads(sigp.read_text(encoding='utf-8')).get('signature')
            except Exception:prev=None
        leftovers=[p for p in out.iterdir() if p.name!='.render_resume.json']
        if leftovers and (prev is None or str(prev)!=str(resume_signature)):
            print('[render] INVALIDATE partial render (signature changed or unversioned)',flush=True);shutil.rmtree(out);out.mkdir(parents=True,exist_ok=True)
        atomic_json(sigp,{'signature':str(resume_signature)})
    for q in list((out/'data').glob('*.tmp.*'))+list((out/'index').glob('*.tmp.*')):q.unlink(missing_ok=True)
    plan_paths=sorted(plan.glob('part-*.mpk'));t0=time.perf_counter()
    # Plan parts can be reclaimed as they are rendered (SBP_RENDER_CONSUME_PLAN),
    # so cached render parts may have no plan source left: enumerate both sides.
    names=sorted({p.name for p in plan_paths}|{p.name for p in (out/'data').glob('part-*.mpk')})
    done=[];tasks=[]
    for name in names:
        dp=out/'data'/name;ip=out/'index'/name
        if dp.is_file() and dp.stat().st_size>len(RENDER_MAGIC) and ip.is_file():
            try:
                ns=0;nt=0;caps=Counter()
                for m in iter_frames(ip):
                    ns+=1;nt+=int(m.get('tokens',0))
                    for cap in m.get('capabilities') or []:caps[str(cap)]+=1
                done.append((dp,ns,nt,caps));continue
            except Exception as exc:
                print(f'[render] redo unreadable part {name}: {exc}',flush=True)
        pp=plan/name
        if not pp.is_file():
            raise RuntimeError(f'render part has neither complete output nor plan source: {name}')
        tasks.append((pp,out,str(Path(tokenizer_path).resolve()),cfg))
    stats=Counter()
    for p,ns,nt,caps in done:
        stats['samples']+=ns;stats['tokens']+=nt
        for cap,c in caps.items():stats['cap::'+cap]+=c
    stats['resumed_parts']=len(done)
    if done:print(f'[render] resume parts_cached={len(done)}/{len(names)} pending={len(tasks)}',flush=True)
    _raw_rw=os.environ.get('SBP_RENDER_WORKERS')
    render_workers=max(1,min(int(workers),max(1,int(_raw_rw)) if _raw_rw not in (None,'') else max(1,os.cpu_count() or 8)))
    consume_plan=os.environ.get('SBP_RENDER_CONSUME_PLAN')=='1'
    if render_workers<=1:
        for t in tasks:
            stats.update(_render_plan_shard(t))
            if consume_plan:t[0].unlink(missing_ok=True)
    else:
        for _t,r in _bounded_process_results(tasks,_render_plan_shard,render_workers,max_in_flight=max(2,render_workers*2),mp_context=process_context()):
            stats.update(r)
            # The worker result means the render part is closed and renamed into place.
            if consume_plan:_t[0].unlink(missing_ok=True)
    report={'stage':'render','workers':workers,'format':'SBSTRUCT1','fim_policy':'online_training',**dict(stats),'elapsed_s':time.perf_counter()-t0}
    atomic_json(out/'render_report.json',report);success_marker(out,'render',report);return report


def final_select(rows:Sequence[Mapping[str,Any]],source_targets:Mapping[str,int],language_targets:Mapping[str,int])->Tuple[List[Dict[str,Any]],Dict[str,Any]]:
    rows=[dict(x) for x in rows];selected=[];selected_ids=set();source_actual=Counter();lang_actual=Counter();by_source=defaultdict(list)
    for r in rows:by_source[str(r['source'])].append(r)
    for source,target in source_targets.items():
        if source=='code':continue
        for r in sorted(by_source.get(source,[]),key=lambda r:(-float(r.get('score',0)),int(r['sample_id']))):
            if source_actual[source]>=target:break
            selected.append(r);selected_ids.add(int(r['sample_id']));source_actual[source]+=int(r['tokens'])
    code_rows=by_source.get('code',[]);by_lang=defaultdict(list)
    for r in code_rows:by_lang[str(r.get('primary_language'))].append(r)
    for lang,target in language_targets.items():
        for r in sorted(by_lang.get(lang,[]),key=lambda r:(-float(r.get('score',0)),int(r['sample_id']))):
            if lang_actual[lang]>=target:break
            if int(r['sample_id']) in selected_ids:continue
            selected.append(r);selected_ids.add(int(r['sample_id']));n=int(r['tokens']);lang_actual[lang]+=n;source_actual['code']+=n
    code_target=int(source_targets.get('code',sum(language_targets.values())))
    if source_actual['code']<code_target:
        leftovers=[r for r in code_rows if int(r['sample_id']) not in selected_ids]
        for r in sorted(leftovers,key=lambda r:(-float(r.get('score',0)),int(r['sample_id']))):
            if source_actual['code']>=code_target:break
            selected.append(r);selected_ids.add(int(r['sample_id']));n=int(r['tokens']);source_actual['code']+=n;lang_actual[str(r.get('primary_language'))]+=n
    return selected,{'source_actual':dict(source_actual),'language_actual':dict(lang_actual),'selected_samples':len(selected),'selected_tokens':sum(int(r['tokens']) for r in selected)}


class _StructuredPartitionWriter:
    def __init__(self,out_dir,partition,shard_tokens,fmt):
        self.out_dir=Path(out_dir);self.partition=partition.replace('/','_');self.shard_tokens=int(shard_tokens);self.fmt=fmt;self.index=0;self.tokens=0;self.writer=None;self.file=None
    def _open(self):
        self.out_dir.mkdir(parents=True,exist_ok=True)
        if self.fmt=='structured_trainpack':
            self.file=self.out_dir/f'{self.partition}-{self.index:05d}.trainpack';self.writer=FramedWriter(self.file,magic=STRUCT_TRAIN_MAGIC)
        elif self.fmt=='jsonl':
            self.file=self.out_dir/f'{self.partition}-{self.index:05d}.jsonl';self.writer=self.file.open('w',encoding='utf-8')
        else:raise ValueError(f'unsupported structured output format: {self.fmt}')
    def write(self,row):
        n=int(row['tokens'])
        if self.writer is None:self._open()
        if self.tokens and self.tokens+n>self.shard_tokens:self.close_current();self.index+=1;self.tokens=0;self._open()
        if self.fmt=='structured_trainpack':self.writer.write(row)
        else:self.writer.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
        self.tokens+=n
    def close_current(self):
        if self.writer is None:return
        self.writer.close();self.writer=None
    def close(self):self.close_current()


def _percentile(vals:Sequence[int],q:float)->int:
    if not vals:return 0
    v=sorted(int(x) for x in vals);return v[min(len(v)-1,max(0,int(round((len(v)-1)*q))))]


class _FrameReaderCache:
    def __init__(self,max_open:int=16,magic:bytes=RENDER_MAGIC):self.max_open=max(1,int(max_open));self.magic=magic;self.files=OrderedDict()
    def _get(self,path:Path|str):
        key=str(Path(path).resolve());f=self.files.pop(key,None)
        if f is None:
            # Same 1 MiB refill as iter_frames: render parts are read sequentially
            # and 8 KiB FUSE reads are an order of magnitude slower on NTFS.
            f=open(key,'rb',buffering=1<<20)
            if f.read(len(self.magic))!=self.magic:f.close();raise RuntimeError(f'invalid render frame magic: {key}')
        self.files[key]=f
        while len(self.files)>self.max_open:
            _,old=self.files.popitem(last=False);old.close()
        return f
    def read(self,path:Path|str,offset:int):
        f=self._get(path);f.seek(int(offset));h=f.read(FRAME.size)
        if len(h)!=FRAME.size:raise RuntimeError(f'bad render frame offset {path}@{offset}')
        (n,)=FRAME.unpack(h);payload=f.read(n)
        if len(payload)!=n:raise RuntimeError(f'truncated render frame {path}@{offset}')
        return msgpack.unpackb(payload,raw=False,strict_map_key=False)
    def close(self):
        for f in self.files.values():f.close()
        self.files.clear()


def _hist_add(hist:MutableMapping[int,int],value:int)->None:hist[int(value)]=int(hist.get(int(value),0))+1

def _hist_percentile(hist:Mapping[int,int],q:float)->int:
    total=sum(int(v) for v in hist.values())
    if total<=0:return 0
    rank=min(total-1,max(0,int(round((total-1)*q))));seen=0
    for value,count in sorted((int(k),int(v)) for k,v in hist.items()):
        seen+=count
        if seen>rank:return value
    return max((int(k) for k in hist),default=0)


def run_trim(render_dir:Path|str,output_dir:Path|str,*,source_targets:Mapping[str,int],language_targets:Mapping[str,int],shard_tokens:int=25_000_000,output_format:str='structured_trainpack',rebuild:bool=False)->Dict[str,Any]:
    """Strict external-memory final selector.

    Fixed code-language quotas are fail-closed: one language can no longer borrow
    leftovers from another merely to make the aggregate Code total look complete.
    """
    rend=Path(render_dir).resolve();out=Path(output_dir).resolve()
    if rebuild and out.exists():shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();index_paths=sorted((rend/'index').glob('*.mpk'));dbp=_trim_db_path(out)
    os.environ.setdefault('SQLITE_TMPDIR',str(dbp.parent))
    try:dbp.unlink()
    except FileNotFoundError:pass
    conn=_open_scale_sqlite(dbp)
    conn.execute('PRAGMA cache_size=-1048576')
    conn.execute('CREATE TABLE rows(bucket TEXT NOT NULL,score REAL NOT NULL,selection_key TEXT PRIMARY KEY,tokens INTEGER NOT NULL) WITHOUT ROWID')
    conn.execute('CREATE TABLE selected(selection_key TEXT PRIMARY KEY,score REAL NOT NULL) WITHOUT ROWID')
    availability=defaultdict(lambda:{'samples':0,'tokens':0});row_count=0
    conn.execute('BEGIN');batch=[]
    for pth in index_paths:
        for m in iter_frames(pth):
            bucket=_selection_bucket(m);key=_selection_key(bucket,int(m['sample_id']));n=int(m['tokens']);score=float(m.get('score',0.0))
            batch.append((bucket,score,key,n));availability[bucket]['samples']+=1;availability[bucket]['tokens']+=n;row_count+=1
            if len(batch)>=20000:
                conn.executemany('INSERT OR REPLACE INTO rows VALUES(?,?,?,?)',batch);conn.commit();conn.execute('BEGIN');batch.clear()
            if row_count%2000000==0:print(f'[trim] index rows={row_count:,}',flush=True)
    if batch:conn.executemany('INSERT OR REPLACE INTO rows VALUES(?,?,?,?)',batch)
    conn.commit()
    bucket_targets={str(s):int(v) for s,v in source_targets.items() if str(s)!='code'}
    for lang,target in language_targets.items():bucket_targets['code/'+str(lang)]=int(target)
    short=[];clamped=[]
    for b,t in list(bucket_targets.items()):
        av=int(availability.get(b,{}).get('tokens',0))
        if av>=t:continue
        if t-av<=max(1_000_000,int(t*0.005)):
            # Render-time skips (empty targets, reserved-token collisions) can shave a few per
            # mille off one bucket; accept that instead of failing a multi-hour copy. Strictly
            # no cross-language refill: the target is clamped to what actually exists.
            clamped.append(f'{b}: need={t:,}, available={av:,}, short={t-av:,}');bucket_targets[b]=av;continue
        short.append(f'{b}: need={t:,}, available={av:,}')
    if clamped:print('[trim] accepted <=0.5% bucket shortfall: '+'; '.join(clamped),flush=True)
    if short:
        conn.close()
        try:dbp.unlink()
        except FileNotFoundError:pass
        raise RuntimeError('final capacity shortfall (strict, no cross-language refill): '+'; '.join(short))
    conn.execute('CREATE INDEX row_rank ON rows(bucket,score DESC,selection_key ASC)');conn.commit()
    source_actual=Counter();lang_actual=Counter();selected_samples=0;selected_tokens=0
    conn.execute('BEGIN')
    for bucket,target in bucket_targets.items():
        used=0;batch=[]
        for score,key,n in conn.execute('SELECT score,selection_key,tokens FROM rows WHERE bucket=? ORDER BY score DESC,selection_key ASC',(bucket,)):
            batch.append((key,float(score)));n=int(n);used+=n;selected_samples+=1;selected_tokens+=n
            if len(batch)>=20000:conn.executemany('INSERT OR IGNORE INTO selected VALUES(?,?)',batch);batch.clear()
            if bucket.startswith('code/'):
                lang=bucket[5:];lang_actual[lang]+=n;source_actual['code']+=n
            else:source_actual[bucket]+=n
            if used>=target:break
        if batch:conn.executemany('INSERT OR IGNORE INTO selected VALUES(?,?)',batch)
        if used<target:
            conn.rollback();conn.close();raise RuntimeError(f'final capacity shortfall after ranking: {bucket}: need={target:,}, selected={used:,}')
    conn.commit()

    writers={};cap_samples=Counter();cap_tokens=Counter();backend_samples=Counter();backend_tokens=Counter();relation_samples=Counter();length_hist=Counter();context_hist=Counter();reader=_FrameReaderCache();written=0;pending=[]
    def flush_pending():
        nonlocal written
        if not pending:return
        keys=[_selection_key(_selection_bucket(m),int(m['sample_id'])) for m in pending];chosen=set(_selected_score_map(conn,keys))
        for m,key in zip(pending,keys):
            if key not in chosen:continue
            row=reader.read(m['sample_path'],int(m['sample_offset']));validate_structured_record(row);part='code/'+str(row.get('primary_language')) if row['source']=='code' else row['source']
            if part not in writers:writers[part]=_StructuredPartitionWriter(out/part,part,shard_tokens,output_format)
            writers[part].write(row);n=int(row['tokens']);_hist_add(length_hist,n);written+=1
            for cap in row.get('capabilities') or []:cap_samples[cap]+=1;cap_tokens[cap]+=n
            if row.get('source')=='code':
                target=(row.get('structured') or {}).get('target') or {};backend=str(target.get('backend') or 'unknown');backend_samples[backend]+=1;backend_tokens[backend]+=n
                ctx=list((row.get('structured') or {}).get('context') or []);_hist_add(context_hist,len(ctx))
                for rel in set(str(x.get('role') or 'related') for x in ctx):relation_samples[rel]+=1
        pending.clear()
    try:
        for pth in index_paths:
            for m in iter_frames(pth):
                pending.append(m)
                if len(pending)>=800:flush_pending()
        flush_pending()
    finally:
        reader.close()
        for w in writers.values():w.close()
    if written!=selected_samples:
        conn.close();raise RuntimeError(f'external trim selected/written mismatch: selected={selected_samples} written={written}')
    conn.close()
    try:dbp.unlink()
    except FileNotFoundError:pass
    capability={k:{'samples':cap_samples[k],'base_tokens':cap_tokens[k]} for k in sorted(cap_samples)};backends={k:{'samples':backend_samples[k],'base_tokens':backend_tokens[k]} for k in sorted(backend_samples)}
    stats={'source_actual':dict(source_actual),'language_actual':dict(lang_actual),'selected_samples':selected_samples,'selected_tokens':selected_tokens}
    short=[];accepted={}
    for label,actual_map,targets in (('source',source_actual,source_targets),('language',lang_actual,language_targets)):
        for k,v in targets.items():
            av=int(actual_map.get(k,0));tv=int(v)
            if av>=tv:continue
            if tv-av<=max(1_000_000,int(tv*0.005)):accepted[f'{label}/{k}']={'target':tv,'actual':av,'short':tv-av};continue
            short.append(f'{label}/{k}: need={tv:,}, actual={av:,}')
    if accepted:print('[trim] accepted <=0.5% final quota shortfall: '+json.dumps(accepted,sort_keys=True),flush=True)
    complete=not short
    report={'stage':'trim','source_targets':dict(source_targets),'language_targets':dict(language_targets),**stats,'selected_base_tokens':selected_tokens,'final_shortfall_accepted':accepted,
            'token_accounting':'base_causal_tokens; online FIM actual tokens are counted by the trainer attention_mask','selection_backend':'sqlite_external','strict_code_language_targets':True,
            'output_format':output_format,'schema':STRUCT_SCHEMA,'fim_policy':'online_training','capability_actual':capability,'code_backend_actual':backends,'code_context_relation_samples':dict(relation_samples),
            'sample_length_base_tokens':{'p50':_hist_percentile(length_hist,.50),'p90':_hist_percentile(length_hist,.90),'p99':_hist_percentile(length_hist,.99),'max':max(length_hist,default=0)},
            'code_context_count':{'p50':_hist_percentile(context_hist,.50),'p90':_hist_percentile(context_hist,.90),'p99':_hist_percentile(context_hist,.99),'max':max(context_hist,default=0)},
            'availability_exact':dict(availability),'elapsed_s':time.perf_counter()-t0,'complete':bool(complete)}
    if not complete:raise RuntimeError(f'final strict quota validation failed: {stats}; '+'; '.join(short))
    atomic_json(out/'processing_report.json',report);success_marker(out,'trim',report);return report



def iter_trainpack(path):
    return iter_structured_trainpack(path,validate=True)



# ============================================================================
# BEGIN cli.py
# ============================================================================
import argparse, hashlib, json, shutil, sys, time
from pathlib import Path

REVISION='2026-09-18-250b-v13-incremental-base-wins'
SEMANTIC_REVISION='2026-09-14-semantic-resume-v7'
CACHE_COMPAT_REVISION='2026-08-31-semantic-block-inventory-v2-pack-compat'
DEDUP_REVISION='2026-09-12-dedup-global-route-v8'
PHASES=['inventory','dedup','semantic','plan','render','trim']

def _prune_milestone(work:Path,milestone:str,level:int):
    """Reclaim intermediates that no later phase (or same-signature resume) reads.

    Safe because every pruned path sits behind a completed _SUCCESS stamp:
    same-signature reruns adopt via stamps; a different signature recomputes
    from upstream inputs, not from pruned caches.
    level 0=off, 1=default (trim-time render-data + candidates reclaim),
    2=aggressive (additionally reclaim candidates+repos right after plan —
    changing target_tokens/ratios afterwards then requires a full semantic redo).
    Set SBP_AUTO_PRUNE=0/1/2 to control.
    """
    if not level:return
    try:
        if milestone=='semantic':
            if os.environ.get('SBP_PRUNE_DEDUP')=='1':
                for sub in (work/'01_inventory',work/'02_dedup'):
                    if sub.is_dir():
                        for ch in sub.iterdir():
                            if ch.name=='_SUCCESS.json':continue
                            _rmtree_any(ch)
            sem=work/'03_semantic'
            for name in ('blocks','repo_runs','repo_scratch','code_active','code_done'):
                _rmtree_any(sem/name)
        elif milestone=='plan':
            plan_done=(work/'04_plan'/'_SUCCESS.json').is_file()
            for db in (work/'04_plan'/'.plan_selection.sqlite',_plan_db_path(work/'04_plan')):
                if plan_done and db.exists():db.unlink()
            if level>=2 and plan_done and os.environ.get('SBP_PRUNE_CANDIDATES')=='1':
                sem=work/'03_semantic'
                for name in ('candidates','repos'):
                    p=sem/name
                    if p.exists():shutil.rmtree(p,ignore_errors=True)
        elif milestone=='trim':
            data=work/'05_render'/'data'
            if data.exists():shutil.rmtree(data,ignore_errors=True)
            sem=work/'03_semantic'
            for name in ('candidates','repos'):
                p=sem/name
                if p.exists():shutil.rmtree(p,ignore_errors=True)
        else:return
        print(f'[prune] {milestone} intermediates reclaimed (SBP_AUTO_PRUNE level={level})',flush=True)
    except Exception as e:
        print(f'[prune] {milestone} prune failed (non-fatal): {e}',flush=True)


def load_config(path):
    override={}
    if path: override=json.loads(Path(path).read_text(encoding='utf-8'))
    cfg=deep_merge(DEFAULT_CONFIG,override)
    # Ratio maps are complete distributions: an override replaces, rather than merges, the default map.
    for key in ('source_shares','language_ratios','code_sample_ratios'):
        if key in override: cfg[key]=dict(override[key])
    return cfg

def _stable_hash(obj):
    raw=json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode('utf-8')
    return hashlib.blake2b(raw,digest_size=16).hexdigest()

def _pack_manifest(stage_dir):
    stage=Path(stage_dir)
    rows=[]
    for p in sorted(stage.rglob('*.pack')):
        try: st=p.stat()
        except FileNotFoundError: continue
        try: rel=str(p.relative_to(stage))
        except ValueError: rel=str(p)
        rows.append((rel,int(st.st_size)))
    manifest=stage/'stage_manifest.json'
    if manifest.exists():
        st=manifest.stat(); rows.append(('stage_manifest.json',int(st.st_size),hashlib.sha256(manifest.read_bytes()).hexdigest()))
    return rows

def _file_fingerprint(path):
    p=Path(path); h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return {'path':str(p.resolve()),'size':p.stat().st_size,'sha256':h.hexdigest()}

def _done(p,signature):
    marker=Path(p)/'_SUCCESS.json'
    if not marker.exists():return False
    try:return json.loads(marker.read_text(encoding='utf-8')).get('_signature')==signature
    except Exception:return False

def _stamp(p,signature):
    marker=Path(p)/'_SUCCESS.json'
    payload=json.loads(marker.read_text(encoding='utf-8')) if marker.exists() else {}
    payload['_signature']=signature
    atomic_json(marker,payload)

def build_parser():
    p=argparse.ArgumentParser(description='Partitioned SemanticBlock preprocessing: no global SQLite, no mandatory LSP')
    p.add_argument('--phase',choices=['inventory','dedup','semantic','plan','render','trim','all','report'],default='all',help='run through this phase; all=trim')
    p.add_argument('--stage-dir',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--base-work-dir',type=Path,default=None,help='immutable completed base work tree; enables v13 incremental base-wins dedup and combined planning')
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=None)
    p.add_argument('--tokenizer',type=Path,default=Path('/home/zjr1236/data/tokenizer_64k/tokenizer.json'))
    p.add_argument('--target-tokens',type=int,default=1_000_000_000)
    p.add_argument('--workers',type=int,default=32)
    p.add_argument('--code-workers',type=int,default=12,help='memory-bounded code-index workers; separate from document/dedup workers')
    p.add_argument('--repo-buckets',type=int,default=None,help='temporary sub-buckets per code partition; default from config')
    p.add_argument('--max-parse-mib',type=float,default=None,help='AST/tree-sitter parse cap per file; larger files use regex structural fallback')
    p.add_argument('--max-code-file-mib',type=float,default=None,help='skip semantic indexing for pathological giant code files; source data remains untouched')
    p.add_argument('--ts-timeout-s',type=float,default=None,help='per-file isolated tree-sitter timeout; default from config')
    p.add_argument('--ts-crash-threshold',type=int,default=None,help='native crashes before quarantining only that language')
    p.add_argument('--ts-parser-cache',type=int,default=None,help='native parser LRU size per isolated parser subprocess')
    p.add_argument('--ts-max-nodes',type=int,default=None,help='max tree-sitter nodes walked per file')
    p.add_argument('--render-workers',type=int,default=16)
    p.add_argument('--seed',type=int,default=20260830)
    p.add_argument('--disable-tree-sitter',action='store_true')
    p.add_argument('--rebuild',action='store_true',help='delete work/output and rebuild all requested stages')
    p.add_argument('--near-dedup',choices=['config','on','off'],default='config')
    p.add_argument('--max-sample-tokens',type=int,default=None)
    p.add_argument('--output-format',choices=['structured_trainpack','jsonl'],default=None)
    return p

def _print_header(a,cfg):
    print('=== semantic block pipeline ===',flush=True)
    print(f'revision       : {REVISION}',flush=True)
    print(f'phase          : {a.phase}',flush=True)
    print(f'target tokens  : {a.target_tokens:,}',flush=True)
    print(f'workers        : docs/dedup={a.workers} code-index={a.code_workers} render={a.render_workers}',flush=True)
    print(f'stage/work/out : {a.stage_dir} | {a.work_dir} | {a.output_dir}',flush=True)
    if a.base_work_dir:print(f'base work      : {a.base_work_dir} (IMMUTABLE reference)',flush=True)
    print('LSP            : OFF (architecture: optional enrichment only)',flush=True)
    print('Tree-sitter    : crash-isolated persistent child per code worker; language quarantine only',flush=True)
    print('FIM            : ONLINE in training; processor stores spans/context only',flush=True)
    print('Giant repos    : hierarchical tree + bounded deterministic local graph',flush=True)
    print('Scale safety   : external-memory dedup finalize / planner / strict trim',flush=True)

def _prepare_stage_output(path:Path|str,resume_partial:bool=False)->None:
    path=Path(path)
    if path.exists() and not resume_partial:shutil.rmtree(path)


def main(argv=None):
    a=build_parser().parse_args(argv)
    if a.target_tokens<=0 or a.workers<1 or a.code_workers<1 or a.render_workers<1:raise SystemExit('invalid positive target/workers required')
    cfg=load_config(a.config);cfg['seed']=a.seed
    if a.max_sample_tokens is not None:cfg['render']['max_sample_tokens']=a.max_sample_tokens
    if a.output_format is not None:cfg['render']['output_format']=a.output_format
    if a.near_dedup!='config':cfg['dedup']['near']=(a.near_dedup=='on')
    a.stage_dir=a.stage_dir.expanduser().resolve();a.work_dir=a.work_dir.expanduser().resolve();a.output_dir=a.output_dir.expanduser().resolve();a.base_work_dir=a.base_work_dir.expanduser().resolve() if a.base_work_dir else None
    if a.rebuild:
        if a.work_dir.exists():shutil.rmtree(a.work_dir)
        if a.output_dir.exists():shutil.rmtree(a.output_dir)
    a.work_dir.mkdir(parents=True,exist_ok=True);_print_header(a,cfg)
    atomic_json(a.work_dir/'resolved_config.json',cfg)
    if a.phase=='report':
        p=a.output_dir/'processing_report.json'
        print(p.read_text() if p.exists() else 'no processing_report.json');return 0
    stop='trim' if a.phase=='all' else a.phase;stop_i=PHASES.index(stop)
    inv=a.work_dir/'01_inventory';dd=a.work_dir/'02_dedup';sem=a.work_dir/'03_semantic';plan=a.work_dir/'04_plan';rend=a.work_dir/'05_render'
    base_dd=(a.base_work_dir/'02_dedup') if a.base_work_dir else None;base_sem=(a.base_work_dir/'03_semantic') if a.base_work_dir else None
    if a.base_work_dir:
        required=[base_dd/'exact_survivors',base_dd/'drop_index.sqlite',base_dd/'_SUCCESS.json',base_sem/'candidates',base_sem/'_SUCCESS.json']
        missing=[str(x) for x in required if not x.exists()]
        if missing:raise SystemExit('incremental base reference incomplete: '+'; '.join(missing))
        if a.base_work_dir==a.work_dir:raise SystemExit('--base-work-dir must differ from augmentation --work-dir')
    icfg=cfg.get('inventory',{});dcfg=cfg['dedup'];scfg=cfg['semantic'];pcfg=cfg['planner'];rcfg=dict(cfg['render']);rcfg['seed']=a.seed
    inv_sig=_stable_hash({'revision':CACHE_COMPAT_REVISION,'stage':'inventory','input':_pack_manifest(a.stage_dir),'records_per_shard':int(icfg.get('records_per_shard',8192))})
    base_dedup_fp=_file_fingerprint(base_dd/'_SUCCESS.json') if base_dd else None
    dd_sig=_stable_hash({'revision':REVISION if base_dd else DEDUP_REVISION,'stage':'dedup','upstream':inv_sig,'config':dcfg,'base_dedup':base_dedup_fp})
    repo_buckets=int(a.repo_buckets if a.repo_buckets is not None else scfg.get('repo_buckets',32))
    max_parse_mib=float(a.max_parse_mib if a.max_parse_mib is not None else scfg.get('max_parse_mib',2.0))
    max_code_file_mib=float(a.max_code_file_mib if a.max_code_file_mib is not None else scfg.get('max_code_file_mib',8.0))
    ts_timeout_s=float(a.ts_timeout_s if a.ts_timeout_s is not None else scfg.get('ts_timeout_s',15.0))
    ts_crash_threshold=int(a.ts_crash_threshold if a.ts_crash_threshold is not None else scfg.get('ts_crash_threshold',2))
    ts_parser_cache=int(a.ts_parser_cache if a.ts_parser_cache is not None else scfg.get('ts_parser_cache',8))
    ts_max_nodes=int(a.ts_max_nodes if a.ts_max_nodes is not None else scfg.get('ts_max_nodes',250000))
    giant_threshold=int(scfg.get('giant_repo_files',20000));giant_row_cap=int(scfg.get('giant_repo_row_cap',12000));giant_target_cap=int(scfg.get('giant_repo_target_cap',4096));giant_neighbor_cap=int(scfg.get('giant_repo_neighbor_cap',512));tree_summary_lines=int(scfg.get('tree_summary_lines',500));target_tree_paths=int(scfg.get('target_tree_paths',24))
    if repo_buckets<1 or max_parse_mib<=0 or max_code_file_mib<=0 or ts_timeout_s<=0 or ts_crash_threshold<1 or ts_parser_cache<1 or ts_max_nodes<1000 or giant_threshold<1 or giant_row_cap<1 or giant_target_cap<1 or giant_neighbor_cap<0 or tree_summary_lines<1 or target_tree_paths<1:raise SystemExit('invalid semantic/tree-sitter/giant-repo safety parameters')
    max_parse_bytes=int(max_parse_mib*(1<<20));max_code_file_bytes=int(max_code_file_mib*(1<<20))
    sem_sig=_stable_hash({'revision':SEMANTIC_REVISION,'stage':'semantic','inventory':inv_sig,'dedup':dd_sig,'config':scfg,'disable_tree_sitter':a.disable_tree_sitter,'repo_buckets':repo_buckets,'max_parse_bytes':max_parse_bytes,'max_code_file_bytes':max_code_file_bytes,'ts_timeout_s':ts_timeout_s,'ts_crash_threshold':ts_crash_threshold,'ts_parser_cache':ts_parser_cache,'ts_max_nodes':ts_max_nodes})
    base_sem_fp=_file_fingerprint(base_sem/'_SUCCESS.json') if base_sem else None
    base_capacity_hint=None
    if a.base_work_dir:
        hp=a.work_dir/'base_capacity_hint.json'
        if hp.is_file():
            try:
                hd=json.loads(hp.read_text(encoding='utf-8'))
                hint_cfg_hash=_stable_hash({'target_tokens':a.target_tokens,'source_shares':cfg['source_shares'],'language_ratios':cfg['language_ratios'],
                    'overplan':float(pcfg.get('overplan',1.06)),'min_quality':float(scfg.get('min_quality',0.0)),'seed':a.seed})
                if hd.get('config_hash')==hint_cfg_hash and hd.get('base_semantic_sha256')==base_sem_fp.get('sha256'):
                    base_capacity_hint={str(k):int(v) for k,v in (hd.get('capacities') or {}).items()}
                    print(f'[plan] validated base capacity hint: {hp} buckets={len(base_capacity_hint)}',flush=True)
                else:
                    print(f'[plan] ignoring stale base capacity hint: {hp}',flush=True)
            except Exception as e:
                print(f'[plan] ignoring unreadable base capacity hint: {hp}: {e}',flush=True)
    plan_sig=_stable_hash({'revision':REVISION,'stage':'plan','upstream':sem_sig,'base_semantic':base_sem_fp,'target_tokens':a.target_tokens,'source_shares':cfg['source_shares'],'language_ratios':cfg['language_ratios'],'planner':pcfg,'seed':a.seed,'fim_policy':'online_training','no_oversampling':True})
    plan_scan_sig=_stable_hash({'revision':REVISION,'stage':'plan_scan','upstream':sem_sig,'base_semantic':base_sem_fp,'seed':a.seed,'min_quality':float(scfg.get('min_quality',0.0))})
    plan_select_sig=_stable_hash({'revision':REVISION,'stage':'plan_select','target_tokens':a.target_tokens,'source_shares':cfg['source_shares'],'language_ratios':cfg['language_ratios'],'overplan':float(pcfg.get('overplan',1.06)),'max_repo_fraction':float(pcfg.get('max_repo_fraction',.01))})
    render_sig=None;trim_sig=None;tokenizer_fp=None
    if stop_i>=4:
        if not a.tokenizer.exists():raise SystemExit(f'tokenizer not found: {a.tokenizer}')
        tokenizer_fp=_file_fingerprint(a.tokenizer)
        render_sig=_stable_hash({'revision':REVISION,'stage':'render','upstream':plan_sig,'tokenizer':tokenizer_fp,'render':rcfg})
    if stop_i>=5:
        source_targets=largest_remainder(a.target_tokens,cfg['source_shares']);language_targets=largest_remainder(source_targets.get('code',0),cfg['language_ratios'])
        trim_sig=_stable_hash({'revision':REVISION,'stage':'trim','upstream':render_sig,'source_targets':source_targets,'language_targets':language_targets,'shard_tokens':int(rcfg.get('shard_tokens',25_000_000)),'output_format':str(rcfg.get('output_format','structured_trainpack'))})
    def stage(i,name,fn,path,signature,resume_partial=False):
        if stop_i<i:return
        if _done(path,signature):print(f'[{i+1}/6 {name}] cached: {path}',flush=True);return
        path=Path(path);_prepare_stage_output(path,resume_partial=resume_partial)
        print(f'[{i+1}/6 {name}] START'+(' (resume)' if resume_partial and path.exists() else ''),flush=True);t=time.perf_counter();r=fn();_stamp(path,signature);print(f'[{i+1}/6 {name}] COMPLETE elapsed={time.perf_counter()-t:.1f}s',flush=True);return r
    auto_prune=int(os.environ.get('SBP_AUTO_PRUNE','1') or 0)
    stage(0,'inventory',lambda:run_inventory(a.stage_dir,inv,workers=a.workers,records_per_shard=int(icfg.get('records_per_shard',8192))),inv,inv_sig,resume_partial=True)
    stage(1,'dedup',lambda:run_dedup(inv,dd,workers=a.workers,exact_partitions=int(dcfg.get('exact_partitions',1024)),lsh_partitions=int(dcfg.get('lsh_partitions',1024)),
                                      near=bool(dcfg.get('near',True)),thresholds=dcfg.get('thresholds'),signature_size=int(dcfg.get('signature_size',32)),bands=int(dcfg.get('bands',8)),max_bucket=int(dcfg.get('max_bucket',512)),
                                      base_dedup_dir=base_dd,checkpoint_signature=dd_sig),dd,dd_sig,resume_partial=True)
    if base_sem is not None and stop_i>=2 and sem.exists() and not _done(sem,sem_sig):
        sm=sem/'semantic_resume_state.json';old_sem_sig=None
        if sm.is_file():
            try:old_sem_sig=json.loads(sm.read_text(encoding='utf-8')).get('signature')
            except Exception:old_sem_sig=None
        if str(old_sem_sig or '')!=str(sem_sig):
            print(f'[3/6 semantic] INVALIDATE stale augmentation semantic old_signature={old_sem_sig!r} new_signature={sem_sig!r}',flush=True)
            shutil.rmtree(sem)
    stage(2,'semantic',lambda:run_semantic(inv,dd,sem,workers=a.workers,code_workers=a.code_workers,repo_partitions=int(scfg.get('repo_partitions',256)),repo_buckets=repo_buckets,chunk_tokens=int(scfg.get('chunk_tokens',8192)),max_related=int(scfg.get('max_related',8)),disable_tree_sitter=a.disable_tree_sitter,max_parse_bytes=max_parse_bytes,max_code_file_bytes=max_code_file_bytes,ts_timeout_s=ts_timeout_s,ts_crash_threshold=ts_crash_threshold,ts_parser_cache=ts_parser_cache,ts_max_nodes=ts_max_nodes,giant_threshold=giant_threshold,giant_row_cap=giant_row_cap,giant_target_cap=giant_target_cap,giant_neighbor_cap=giant_neighbor_cap,tree_summary_lines=tree_summary_lines,target_tree_paths=target_tree_paths,resume_signature=sem_sig),sem,sem_sig,resume_partial=True)
    if stop_i>=2:_prune_milestone(a.work_dir,'semantic',auto_prune)
    plan_inputs=[('base',base_sem),('aug',sem)] if base_sem is not None else sem
    stage(3,'plan',lambda:run_plan(plan_inputs,plan,target_tokens=a.target_tokens,source_shares=cfg['source_shares'],language_ratios=cfg['language_ratios'],code_sample_ratios=cfg['code_sample_ratios'],
                                   overplan=float(pcfg.get('overplan',1.06)),max_repo_fraction=float(pcfg.get('max_repo_fraction',.01)),plan_shards=int(pcfg.get('plan_shards',128)),seed=a.seed,math_fim_rate=float(cfg.get('math_fim_rate',.25)),min_quality=float(scfg.get('min_quality',0.0)),
                                   base_capacity_hint=base_capacity_hint,resume_signature=plan_scan_sig,resume_select_signature=plan_select_sig),plan,plan_sig,resume_partial=True)
    if stop_i>=3:_prune_milestone(a.work_dir,'plan',auto_prune)
    if stop_i>=4:
        stage(4,'render',lambda:run_render(plan,rend,tokenizer_path=a.tokenizer,workers=a.render_workers,config=rcfg,resume_signature=render_sig),rend,render_sig,resume_partial=True)
    if stop_i>=5:
        stage(5,'trim',lambda:run_trim(rend,a.output_dir,source_targets=source_targets,language_targets=language_targets,shard_tokens=int(rcfg.get('shard_tokens',25_000_000)),output_format=str(rcfg.get('output_format','structured_trainpack')),rebuild=False),a.output_dir,trim_sig)
        _prune_milestone(a.work_dir,'trim',auto_prune)
        report_path=a.output_dir/'processing_report.json'
        if report_path.exists():
            rep=json.loads(report_path.read_text(encoding='utf-8'))
            code_sem_report={}
            sp=sem/'semantic_report.json'
            if sp.exists():
                try:code_sem_report=json.loads(sp.read_text(encoding='utf-8'))
                except Exception:code_sem_report={}
            rep.update({'pipeline_revision':REVISION,'stage_dir':str(a.stage_dir),'work_dir':str(a.work_dir),'target_tokens':a.target_tokens,'tokenizer':tokenizer_fp,'resolved_config':cfg,'stage_signature':trim_sig,'code_semantic':code_sem_report})
            atomic_json(report_path,rep)
    return 0



if __name__ == '__main__':
    raise SystemExit(main())
