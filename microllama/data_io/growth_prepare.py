"""Bounded, resumable FineWeb-Edu/DCLM-Edu acquisition and parallel packing."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sqlite3
import time
import numpy as np
from .corpus import content_hash
from .download import sha256
from .encoding import ENCODING_POLICY_VERSION
from .growth_reader import MODE, digest, load_catalog, publish_catalog, usable_targets
from .transfer import atomic_json
from ..common.resources import available_cpus

from .growth_sources import SOURCES

DEDUP_POLICY = 'normalized-blake2b16+sampled-5word-simhash128-hamming3-v1'
MASK64 = (1<<64)-1

def select_bounded_shard(files, selected, source, batch, raw_cap, pilot_cap, batch_cap):
    """Roll later byte-bounded batches without weakening the cumulative budget."""
    used={(f['source'],f['path']) for f in selected}
    remaining=[f for f in files[source] if (source,f['path']) not in used]
    if not remaining:
        raise RuntimeError(f'Pinned source inventory exhausted for {source}')
    total=sum(f['bytes'] for f in selected)
    eligible=[f for f in remaining if total+f['bytes']<=raw_cap]
    if not eligible:
        raise RuntimeError(f'Cumulative raw cap exhausted for {source}; review retained yield/shortfall')
    used_batch=sum(f['bytes'] for f in selected if f['batch']==batch)
    cap=pilot_cap if batch==0 else batch_cap
    choices=[f for f in eligible if used_batch+f['bytes']<=cap]
    if not choices:
        if batch==0:
            raise RuntimeError(f'Pilot raw cap exhausted for {source}; review retained yield/shortfall')
        choices=[f for f in eligible if f['bytes']<=batch_cap]
        if not choices:
            raise RuntimeError(f'No {source} shard fits the later batch cap')
        batch=max(batch,max((f['batch'] for f in selected),default=0))+1
    return {**choices[0],'sequence':len(selected)+1,'batch':batch}


def next_coverage_milestone(balanced, total_new):
    """Coverage milestones are independent of raw batch count and resume order."""
    target=2_000_000_000
    while balanced>=target and target<total_new+2_000_000:
        target=min(total_new+2_000_000,target+5_000_000_000)
    return target


def signature(text):
    words = text.lower().split()
    if len(words)<16:
        return None
    count=len(words)-4
    indices=np.unique(np.linspace(0,count-1,min(128,count),dtype=np.int64))
    raw=b''.join(hashlib.blake2b(' '.join(words[int(i):int(i)+5]).encode(),digest_size=16).digest() for i in indices)
    hashes=np.frombuffer(raw,dtype='<u8').reshape(-1,2)
    values=[]
    for half in range(2):
        bits=((hashes[:,half,None] >> np.arange(64,dtype=np.uint64)) & np.uint64(1)).sum(0)*2>=len(hashes)
        values.append(int((bits.astype(np.uint64)*(np.uint64(1)<<np.arange(64,dtype=np.uint64))).sum(dtype=np.uint64)))
    return f'{values[0]+(values[1]<<64):032x}'

def annotate(batch):
    return [{**doc,'content_hash':content_hash(doc.get('text') or ''),'signature':signature(doc.get('text') or '')} for doc in batch]

def annotated_rows(path,pool,columns=('text','id')):
    import pyarrow.parquet as pq
    pending=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256,columns=list(columns)):
        pending.append(pool.submit(annotate,batch.to_pylist()))
        if len(pending)>=max(2,pool._max_workers*2):
            yield from pending.pop(0).result()
    for future in pending:
        yield from future.result()

class DedupIndex:
    def __init__(self,path):
        self.db=sqlite3.connect(path,timeout=60)
        self.db.executescript('''PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS exact(hash TEXT PRIMARY KEY, split TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS ids(repo TEXT,id TEXT,hash TEXT,PRIMARY KEY(repo,id));
            CREATE TABLE IF NOT EXISTS near(hash TEXT PRIMARY KEY,sig TEXT,b0 INTEGER,b1 INTEGER,b2 INTEGER,b3 INTEGER);
            CREATE INDEX IF NOT EXISTS n0 ON near(b0); CREATE INDEX IF NOT EXISTS n1 ON near(b1);
            CREATE INDEX IF NOT EXISTS n2 ON near(b2); CREATE INDEX IF NOT EXISTS n3 ON near(b3);
            CREATE TABLE IF NOT EXISTS receipts(key TEXT PRIMARY KEY,value TEXT NOT NULL);''')
        self.db.commit()
    def receipt(self,key):
        row=self.db.execute('SELECT value FROM receipts WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None
    def save(self,key,value):
        self.db.execute('INSERT OR REPLACE INTO receipts VALUES (?,?)',(key,json.dumps(value,sort_keys=True)))
    def add_near(self,hash_,sig):
        if sig is not None:
            bands=[int(sig,16)>>(32*i)&((1<<32)-1) for i in range(4)]
            self.db.execute('INSERT OR IGNORE INTO near VALUES (?,?,?,?,?,?)',(hash_,sig,*bands))
    def admit(self,repo,doc):
        hash_=doc['content_hash']; id_=json.dumps(doc['id'],sort_keys=True)
        if self.db.execute('SELECT 1 FROM exact WHERE hash=?',(hash_,)).fetchone(): return 'exact_overlap'
        if self.db.execute('SELECT 1 FROM ids WHERE repo=? AND id=?',(repo,id_)).fetchone(): return 'source_id_overlap'
        sig=doc['signature']
        if sig is not None:
            bands=[int(sig,16)>>(32*i)&((1<<32)-1) for i in range(4)]
            candidates=self.db.execute('SELECT sig FROM near WHERE b0=? OR b1=? OR b2=? OR b3=? LIMIT 257',bands).fetchall()
            if len(candidates)>256: return 'ambiguous_near_bucket'
            if any((int(sig,16) ^ int(row[0],16)).bit_count()<=3 for row in candidates): return 'near_overlap'
        from .corpus import split_for
        split=split_for(doc['text'])
        self.db.execute('INSERT INTO exact VALUES (?,?)',(hash_,split))
        self.db.execute('INSERT INTO ids VALUES (?,?,?)',(repo,id_,hash_))
        self.add_near(hash_,sig)
        return 'accepted'

def verify_legacy(root,tokenizer_hash,sequence_length):
    summary=json.loads((root/'manifest.json').read_text())
    if not summary.get('complete') or summary['tokenizer_sha256']!=tokenizer_hash:
        raise ValueError('Complete checkpoint-compatible starting corpus required')
    chunks={}; total=0
    for file in sorted(root.glob('*/manifest.json')):
        if file.parent.name.endswith('.partial'): continue
        meta=json.loads(file.read_text())
        if meta['source_revision']!=summary['source_revision'] or meta['tokenizer_sha256']!=tokenizer_hash:
            raise ValueError('Legacy corpus identity mismatch')
        for split in ('train','validation'):
            path=file.parent/f'{split}.bin'
            if path.stat().st_size!=2*meta['splits'][split]['tokens'] or sha256(path)!=meta['splits'][split]['sha256']:
                raise ValueError('Legacy corpus checksum mismatch')
        valid=usable_targets(file.parent/'train.bin',sequence_length)
        chunks[file.parent.name]={'manifest_sha256':sha256(file),'valid_targets':valid}
        total+=valid
    if not chunks: raise ValueError('Empty legacy corpus')
    return {'chunks':chunks,'valid_targets_per_pass':total,'summary_sha256':sha256(root/'manifest.json'),
            'raw_train_tokens':summary['train_tokens']}

def import_legacy(index,root,raw,pool,status):
    repo=raw['repo']
    for file in sorted(root.glob('*/*.documents.jsonl')):
        key='sidecar:'+str(file)
        saved=index.receipt(key); file_hash=sha256(file)
        if saved:
            if saved['sha256']!=file_hash: raise ValueError('Imported legacy sidecar changed')
            continue
        split=file.name.split('.')[0]; count=0
        index.db.execute('BEGIN IMMEDIATE')
        with file.open() as stream:
            for line in stream:
                row=json.loads(line); hash_=row['content_hash']
                existing=index.db.execute('SELECT split FROM exact WHERE hash=?',(hash_,)).fetchone()
                if existing and existing[0]!=split: raise ValueError('Legacy split overlap')
                index.db.execute('INSERT OR IGNORE INTO exact VALUES (?,?)',(hash_,split))
                index.db.execute('INSERT OR IGNORE INTO ids VALUES (?,?,?)',(repo,json.dumps(row['id'],sort_keys=True),hash_))
                count+=1
        index.save(key,{'sha256':file_hash,'documents':count}); index.db.commit()
        status(phase='importing-existing-provenance',sidecar=file.name,sidecar_documents=count)
    for item in raw['files']:
        key='legacy-raw:'+item['path']; saved=index.receipt(key)
        if saved:
            if saved['sha256']!=item['sha256']: raise ValueError('Legacy raw source changed')
            continue
        path=Path(raw['data_dir'])/item['path']
        if sha256(path)!=item['sha256']: raise ValueError('Legacy raw source checksum mismatch')
        index.db.execute('BEGIN IMMEDIATE'); count=0
        for row in annotated_rows(path,pool):
            if not index.db.execute('SELECT 1 FROM exact WHERE hash=?',(row['content_hash'],)).fetchone():
                raise ValueError('Legacy raw document missing from imported sidecars')
            index.add_near(row['content_hash'],row['signature']); count+=1
            if count%100_000==0: status(phase='indexing-existing-near-duplicates',source=item['path'],documents=count)
        index.save(key,{'sha256':item['sha256'],'documents':count}); index.db.commit()
        status(phase='indexing-existing-near-duplicates',source=item['path'],documents=count,source_complete=True)

def filter_source(index,item,path,output,pool):
    import pyarrow as pa
    import pyarrow.parquet as pq
    key='filter:'+str(item['sequence']); saved=index.receipt(key)
    if saved:
        if saved['original_sha256']!=item['sha256']: raise ValueError('Source receipt changed')
        if saved.get('published'): return saved
        if not output.exists() or sha256(output)!=saved['filtered_sha256']: raise ValueError('Filtered source changed/missing')
        return saved
    output.parent.mkdir(parents=True,exist_ok=True)
    partial=output.with_suffix('.partial')
    columns=['text','id']+(['edu_int_score'] if item['source']=='dclm' else [])
    counts={'accepted':0,'quality_rejected':0,'empty_rejected':0}; rows=[]
    writer=pq.ParquetWriter(partial,pa.schema([('text',pa.string()),('id',pa.string())]))
    index.db.execute('BEGIN IMMEDIATE')
    try:
        for doc in annotated_rows(path,pool,columns):
            if not isinstance(doc['text'],str) or not doc['text'].strip(): reason='empty_rejected'
            elif item['source']=='dclm' and (doc.get('edu_int_score') is None or doc['edu_int_score']<3): reason='quality_rejected'
            else: reason=index.admit(SOURCES[item['source']]['repo'],doc)
            counts[reason]=counts.get(reason,0)+1
            if reason=='accepted': rows.append({'text':doc['text'],'id':str(doc['id'])})
            if len(rows)>=512:
                writer.write_table(pa.Table.from_pylist(rows,schema=writer.schema)); rows.clear()
        if rows: writer.write_table(pa.Table.from_pylist(rows,schema=writer.schema))
        writer.close()
        with partial.open('rb') as stream: os.fsync(stream.fileno())
        partial.replace(output)
        receipt={'original_sha256':item['sha256'],'filtered_sha256':sha256(output),'counts':counts,'published':False}
        index.save(key,receipt); index.db.commit()
        return receipt
    except BaseException:
        writer.close(); index.db.rollback(); raise

def tokenize_filtered(path,item,output,tokenizer_json,tokenizer_hash,eos,threads):
    path, output, tokenizer_json = Path(path), Path(output), Path(tokenizer_json)
    os.environ['RAYON_NUM_THREADS']=str(threads); os.environ['TOKENIZERS_PARALLELISM']='true'
    from .prepare import prepare_source
    from .encoding import load_ordinary_text_tokenizer
    source={'revision':SOURCES[item['source']]['revision']}
    output.mkdir(parents=True,exist_ok=True)
    prepare_source(Path(path),{'path':Path(path).name,'sha256':sha256(path)},source,
                   load_ordinary_text_tokenizer(tokenizer_json),tokenizer_hash,eos,output,8_388_608)
    for file in sorted(output.glob('*/manifest.json')):
        if file.parent.name.endswith('.partial'): continue
        meta=json.loads(file.read_text())
        meta.update(source_repo=SOURCES[item['source']]['repo'],original_file=item['path'],original_source_sha256=item['sha256'],
                    selection_sequence=item['sequence'],dedup_policy=DEDUP_POLICY,
                    document_sidecars={s:sha256(file.parent/f'{s}.documents.jsonl') for s in ('train','validation')})
        atomic_json(file,meta)
    return str(output)

def metadata():
    from huggingface_hub import HfApi
    api=HfApi(); result={}
    for name,pin in SOURCES.items():
        info=api.dataset_info(pin['repo'],revision=pin['revision'],files_metadata=True)
        if info.sha!=pin['revision']: raise ValueError('Source revision was not resolved exactly')
        files=[{'source':name,'path':f.rfilename,'bytes':f.size,'sha256':f.lfs.sha256 if f.lfs else None}
               for f in info.siblings if f.rfilename.startswith(pin['prefix']) and f.rfilename.endswith('.parquet')]
        if not files or any(not f['bytes'] or not f['sha256'] for f in files): raise ValueError('Source metadata lacks sizes/SHA')
        result[name]=sorted(files,key=lambda f:digest([42,pin['repo'],f['path']]))
    return result

def fetch_file(item,root,reserve):
    if shutil.disk_usage(root).free<item['bytes']+reserve: raise RuntimeError('Insufficient free storage for bounded source download')
    from huggingface_hub import hf_hub_download
    pin=SOURCES[item['source']]
    directory=root/'staging'/f"s{item['sequence']:06d}"/'raw'
    path=Path(hf_hub_download(pin['repo'],item['path'],repo_type='dataset',revision=pin['revision'],local_dir=directory))
    if path.stat().st_size!=item['bytes'] or sha256(path)!=item['sha256']: raise ValueError('Selected source checksum mismatch')
    return path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--growth-dir',type=Path,required=True); p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--selection-manifest',type=Path,help='Replay only these frozen shard selections; never select extra files')
    p.add_argument('--legacy-token-dir',type=Path,required=True); p.add_argument('--target-tokens',type=int,default=50_000_000_000)
    p.add_argument('--initial-tokens',type=int,default=10_000_000_000); p.add_argument('--raw-byte-cap',type=int,default=250_000_000_000)
    p.add_argument('--pilot-byte-cap',type=int,default=16_000_000_000); p.add_argument('--batch-byte-cap',type=int,default=30_000_000_000)
    p.add_argument('--reserve-gib',type=int,default=100); args=p.parse_args()
    if args.target_tokens<=args.initial_tokens or min(args.raw_byte_cap,args.pilot_byte_cap,args.batch_byte_cap,args.reserve_gib)<1: p.error('Positive bounded growth budgets required')
    root=args.growth_dir.resolve(); root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.prepare.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    data=args.data_dir.resolve(); legacy=args.legacy_token_dir.resolve()
    info=json.loads((data/'tokenizer/source_manifest.json').read_text()); tokenizer=data/'tokenizer/tokenizer.json'
    if sha256(tokenizer)!=info['tokenizer_sha256']: raise ValueError('Tokenizer checksum mismatch')
    status_value={}
    def status(**values):
        status_value.update(values,updated_unix=time.time()); atomic_json(root/'prepare-status.json',status_value)
        print(json.dumps(values),flush=True)
    cpus=available_cpus(); signature_workers=max(1,min(48,(cpus-8)//4)); token_workers=max(1,min(8,(cpus-8-signature_workers)//4))
    threads=max(1,min(16,(cpus-8-signature_workers)//token_workers))
    status(phase='resolving-pinned-shards',cpus=cpus,signature_workers=signature_workers,token_workers=token_workers,tokenizer_threads=threads)
    files=metadata(); index=DedupIndex(root/'admission.sqlite3'); selected_path=root/'selected-shards.json'
    selected=json.loads(selected_path.read_text()) if selected_path.exists() else []
    if args.selection_manifest:
        frozen=json.loads(args.selection_manifest.read_text())
        if selected and selected!=frozen:
            raise ValueError('Existing selection differs from the reproduction manifest')
        selected=frozen
        atomic_json(selected_path,selected)
    advertised={(f['source'],f['path']):f for group in files.values() for f in group}
    if len({(f['source'],f['path']) for f in selected}) != len(selected):
        raise ValueError('Repeated selected source file')
    for number,item in enumerate(selected,1):
        expected=advertised.get((item['source'],item['path']))
        if item['sequence']!=number or not expected or any(item[k]!=expected[k] for k in ('bytes','sha256')):
            raise ValueError('Frozen selection differs from pinned upstream metadata')
    if sum(f['bytes'] for f in selected)>args.raw_byte_cap:
        raise ValueError('Existing selection exceeds requested raw cap')
    reserve=args.reserve_gib*2**30
    identity={'sources':SOURCES,'tokenizer_sha256':info['tokenizer_sha256'],'sequence_length':8192,'seed':42,
              'eos':info['eos_token_id'],'pad':info['pad_token_id'],'dedup_policy':DEDUP_POLICY}
    inventory_path=root/'legacy-inventory.json'
    if not inventory_path.exists():
        status(phase='verifying-checkpoint-compatible-corpus')
        atomic_json(inventory_path,verify_legacy(legacy,info['tokenizer_sha256'],8192))
    identity['legacy']={'root':str(legacy),'inventory':str(inventory_path),'inventory_sha256':sha256(inventory_path)}
    catalog={'schema_version':1,'mode':MODE,'recipe':identity,'recipe_id':digest(identity),'initial_tokens':args.initial_tokens,
             'target_tokens':args.target_tokens,'status':'preparing','complete':False,
             'sources':{s:{'directory':str(root/'sources'/s/'tokens'),'chunks':{}} for s in SOURCES}}
    catalog_dir=root/'catalog'
    if (catalog_dir/'manifest.json').exists():
        old=load_catalog(catalog_dir,allow_failed=True)
        if old['recipe_id']!=catalog['recipe_id'] or args.target_tokens<old['target_tokens']: raise ValueError('Incompatible existing growth plan')
        catalog['sources']=old['sources']
    publish_catalog(catalog_dir,catalog)
    atomic_json(root/'plan.json',{'target_tokens':args.target_tokens,'initial_tokens':args.initial_tokens,'raw_byte_cap':args.raw_byte_cap,
                'pilot_byte_cap':args.pilot_byte_cap,'batch_byte_cap':args.batch_byte_cap,'recipe_id':catalog['recipe_id'],'recipe':identity,
                'acquisition_policy':'byte-bounded-later-batches-v1',
                'benchmark_exclusion':'Original held-outs excluded; no public-task contamination-free claim',
                'near_duplicate_limits':'Sampled 128 five-word shingles; Hamming<=3; conservative bucket rejection; not exhaustive semantic dedup'})
    ctx=multiprocessing.get_context('spawn')
    signatures=ProcessPoolExecutor(signature_workers,mp_context=ctx)
    tokens=ProcessPoolExecutor(token_workers,mp_context=ctx)
    downloads=ThreadPoolExecutor(2); jobs={}; fetched={}
    def coverage():
        return {s:sum(c['valid_targets'] for c in e['chunks'].values()) for s,e in catalog['sources'].items()}
    def publish_jobs(wait=False):
        changed=False
        for future,item in list(jobs.items()):
            if not wait and not future.done(): continue
            stage=Path(future.result()); source=item['source']; dest=Path(catalog['sources'][source]['directory']); dest.mkdir(parents=True,exist_ok=True)
            for file in sorted(stage.glob('*/manifest.json')):
                if file.parent.name.endswith('.partial'): continue
                name=file.parent.name; final=dest/name; temporary=dest/(name+'.partial')
                if not final.exists():
                    if temporary.exists(): shutil.rmtree(temporary)
                    temporary.mkdir()
                    for child in file.parent.iterdir(): os.link(child,temporary/child.name)
                    temporary.rename(final)
                if sha256(final/'manifest.json')!=sha256(file): raise ValueError('Published growth chunk collision')
                meta=json.loads(file.read_text())
                catalog['sources'][source]['chunks'][name]={'manifest_sha256':sha256(file),
                    'valid_targets':usable_targets(final/'train.bin',8192,identity['eos'],identity['pad'])}
            receipt=index.receipt('filter:'+str(item['sequence'])); receipt['published']=True
            receipt['selection']=item; index.save('filter:'+str(item['sequence']),receipt); index.db.commit()
            atomic_json(root/'receipts'/f"s{item['sequence']:06d}.json",receipt)
            del jobs[future]; changed=True
            # These are owned staging copies, never the original/shared HF cache.
            shutil.rmtree(root/'staging'/f"s{item['sequence']:06d}")
        if changed:
            publish_catalog(catalog_dir,catalog)
            status(phase='tokenizing-additional-data',coverage=coverage(),selected_raw_bytes=sum(f['bytes'] for f in selected))
    def select(source,batch):
        if args.selection_manifest:
            raise RuntimeError('Frozen selection cannot cover the requested target; no extra downloads permitted')
        item=select_bounded_shard(files,selected,source,batch,args.raw_byte_cap,args.pilot_byte_cap,args.batch_byte_cap)
        if item['batch']!=batch:
            status(phase='opening-bounded-batch',previous_batch=batch,batch=item['batch'],
                   reason='current raw batch cannot fit the next shard',
                   remaining_raw_bytes=args.raw_byte_cap-sum(f['bytes'] for f in selected))
        selected.append(item); atomic_json(selected_path,selected)
        return item
    try:
        # Start a strictly bounded first download per source while old provenance is indexed.
        initial=[]
        for s in SOURCES:
            item=next((f for f in selected if f['source']==s and not index.receipt('filter:'+str(f['sequence']))),None)
            if item is None and not selected: item=select(s,0)
            elif item is None and not any(f['source']==s for f in selected): item=select(s,0)
            if item: initial.append(item); fetched[item['sequence']]=downloads.submit(fetch_file,item,root,reserve)
        raw=json.loads((data/'raw/fineweb-edu-10bt/manifest.json').read_text())
        import_legacy(index,legacy,raw,signatures,status)
        batch=max((f['batch'] for f in selected),default=0)
        total_new=args.target_tokens-args.initial_tokens
        goals={s:int(total_new*pin['weight'])+1_000_000 for s,pin in SOURCES.items()}
        while True:
            publish_jobs()
            cov=coverage()
            if all(cov[s]>=goals[s] for s in SOURCES) and (not args.selection_manifest or all(
                    (index.receipt('filter:'+str(f['sequence'])) or {}).get('published') for f in selected)):
                break
            pending=[f for f in selected if not (index.receipt('filter:'+str(f['sequence'])) or {}).get('published') and f['sequence'] not in {i['sequence'] for i in jobs.values()}]
            if not pending:
                if args.selection_manifest and jobs:
                    publish_jobs(wait=True); continue
                if len(jobs)>=token_workers:
                    publish_jobs(wait=True); continue
                balanced=min(cov[s]/SOURCES[s]['weight'] for s in SOURCES)
                batch_target=next_coverage_milestone(balanced,total_new)
                if batch==0 and balanced>=2_000_000_000:
                    batch=1
                # The estimate only sizes in-flight work; published boundary-masked
                # counts determine completion. Selection remains byte-bounded.
                projected={s:cov[s]+sum(700_000_000 for i in jobs.values() if i['source']==s) for s in SOURCES}
                source=max(SOURCES,key=lambda s:min(goals[s],int(batch_target*SOURCES[s]['weight'])+1_000_000)-projected[s])
                try:
                    item=select(source,batch); batch=item['batch']; pending=[item]
                except RuntimeError:
                    if jobs:
                        publish_jobs(wait=True); continue
                    raise
            item=pending[0]; seq=item['sequence']; saved=index.receipt('filter:'+str(seq))
            stage=root/'staging'/f's{seq:06d}'; filtered=stage/f"s{seq:06d}.parquet"
            if not saved:
                future=fetched.pop(seq,None) or downloads.submit(fetch_file,item,root,reserve)
                path=future.result(); status(phase='admitting-shard',selection=item)
                receipt=filter_source(index,item,path,filtered,signatures)
                status(phase='admitting-shard',selection_sequence=seq,rejections=receipt['counts'])
            elif saved.get('published'): continue
            if shutil.disk_usage(root).free<reserve: raise RuntimeError('Growth preparation reached free-space reserve')
            future=tokens.submit(tokenize_filtered,str(filtered),item,str(stage/'tokens'),str(tokenizer),info['tokenizer_sha256'],identity['eos'],threads)
            jobs[future]=item
            if len(jobs)>=token_workers: publish_jobs(wait=True)
        publish_jobs(wait=True)
        catalog.update(complete=True,status='complete'); publish_catalog(catalog_dir,catalog)
        status(phase='complete',coverage=coverage(),selected_raw_bytes=sum(f['bytes'] for f in selected),target_tokens=args.target_tokens)
    except BaseException as error:
        index.db.rollback(); catalog.update(status='failed',error=f'{type(error).__name__}: {error}')
        publish_catalog(catalog_dir,catalog); status(phase='failed',error=catalog['error']); raise
    finally:
        signatures.shutdown(wait=True,cancel_futures=True); tokens.shutdown(wait=True,cancel_futures=True); downloads.shutdown(wait=True,cancel_futures=True)
        index.db.close()

if __name__=='__main__': main()
