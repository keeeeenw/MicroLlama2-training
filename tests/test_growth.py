"""Growth restart, no-replay, provenance, filter and cron-gate regressions."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import numpy as np
import pytest
from microllama.data_io.download import sha256
from microllama.data_io.transfer import atomic_json
from microllama.data_io.growth_reader import MODE, MixtureTokenBlocks, GrowthExhausted, digest, publish_catalog, usable_targets
from microllama.data_io.growth_prepare import DedupIndex, annotate, filter_source, SOURCES, select_bounded_shard, next_coverage_milestone
from scripts.cloud.grow import metric_gate, quality_gate, tick


def test_raw_batch_rollover_preserves_caps_order_and_restart_selection():
    files={'dclm':[{'source':'dclm','path':p,'bytes':size} for p,size in [('used',29),('next',3),('last',2)]]}
    selected=[{**files['dclm'][0],'sequence':1,'batch':6}]
    before=copy.deepcopy(selected)
    item=select_bounded_shard(files,selected,'dclm',6,34,16,30)
    assert selected==before
    assert item=={**files['dclm'][1],'sequence':2,'batch':7}
    selected.append(item)
    # Simulate fresh process reconstructing the byte batch from the frozen ledger.
    resumed=json.loads(json.dumps(selected))
    final=select_bounded_shard(files,resumed,'dclm',max(x['batch'] for x in resumed),34,16,30)
    assert final['path']=='last' and final['batch']==7 and final['sequence']==3
    resumed.append(final)
    assert sum(x['bytes'] for x in resumed)==34
    assert max(sum(x['bytes'] for x in resumed if x['batch']==batch) for batch in (6,7))<=30
    files['dclm'].append({'source':'dclm','path':'too-much','bytes':1})
    with pytest.raises(RuntimeError,match='Cumulative raw cap'):
        select_bounded_shard(files,resumed,'dclm',7,34,16,30)


def test_pilot_and_oversized_shards_do_not_escape_byte_caps():
    files={'edu':[{'source':'edu','path':'used','bytes':15},{'source':'edu','path':'next','bytes':2}]}
    selected=[{**files['edu'][0],'sequence':1,'batch':0}]
    with pytest.raises(RuntimeError,match='Pilot raw cap'):
        select_bounded_shard(files,selected,'edu',0,250,16,30)
    files['edu'][1]['bytes']=31
    with pytest.raises(RuntimeError,match='batch cap'):
        select_bounded_shard(files,selected,'edu',1,250,16,30)
    with pytest.raises(RuntimeError,match='inventory exhausted'):
        select_bounded_shard(files,[{**x,'batch':1} for x in files['edu']],'edu',1,250,16,30)


def test_coverage_milestones_do_not_depend_on_extra_raw_batches():
    assert next_coverage_milestone(0,40_000_000_000)==2_000_000_000
    assert next_coverage_milestone(31_762_223_077,40_000_000_000)==32_000_000_000
    assert next_coverage_milestone(32_000_000_000,40_000_000_000)==37_000_000_000
    assert next_coverage_milestone(39_000_000_000,40_000_000_000)==40_002_000_000
    assert next_coverage_milestone(40_002_000_000,40_000_000_000)==40_002_000_000


def chunk(directory,name,revision,values):
    folder=directory/name; folder.mkdir(parents=True)
    meta={'source_revision':revision,'tokenizer_sha256':'tok','splits':{}}
    for split in ('train','validation'):
        file=folder/f'{split}.bin'; file.write_bytes(np.asarray(values,dtype='<u2').tobytes())
        meta['splits'][split]={'tokens':len(values),'sha256':sha256(file)}
    atomic_json(folder/'manifest.json',meta)
    return {'manifest_sha256':sha256(folder/'manifest.json'),'valid_targets':usable_targets(folder/'train.bin',4)}


def catalog(tmp_path,complete=False):
    legacy=tmp_path/'legacy.json'; atomic_json(legacy,{'chunks':{'old':{'manifest_sha256':'old-sha'}}})
    recipe={'sources':{s:{'weight':w,'revision':s} for s,w in [('edu',.6),('dclm',.4)]},
            'sequence_length':4,'seed':42,'eos':7,'pad':0,'tokenizer_sha256':'tok',
            'legacy':{'inventory':str(legacy),'inventory_sha256':sha256(legacy),'root':str(tmp_path/'old')}}
    value={'schema_version':1,'mode':MODE,'recipe':recipe,'recipe_id':digest(recipe),'initial_tokens':100,
           'target_tokens':1000,'status':'complete' if complete else 'preparing','complete':complete,
           'sources':{s:{'directory':str(tmp_path/s),'chunks':{'a':chunk(tmp_path/s,'a',s,range(20,421))}} for s in recipe['sources']}}
    root=tmp_path/'catalog'; publish_catalog(root,value)
    return root,value


def migrated(root):
    reader=MixtureTokenBlocks(root,4)
    old={'mode':'growing','sequence_length':4,'seed':42,'known':{'old':'old-sha'},'cursor':88,'epoch':1,'current':'old','consumed':[]}
    reader.migrate(old,100)
    return reader


def test_cursor_migration_append_and_fresh_restart(tmp_path):
    root,value=catalog(tmp_path); reader=migrated(root)
    assert reader.retired_reader['cursor']==88 and reader.parent_tokens==100
    reader.next_batch(17); state=reader.state_dict()
    for source in value['sources']:
        value['sources'][source]['chunks']['b']=chunk(tmp_path/source,'b',source,range(500,901))
    publish_catalog(root,value)
    expected=reader.next_batch(60)
    fresh=MixtureTokenBlocks(root,4); fresh.load_state_dict(state)
    np.testing.assert_array_equal(fresh.next_batch(60),expected)
    assert fresh.state_dict()==reader.state_dict()
    total=sum(fresh.targets.values())
    assert abs(fresh.targets['edu']/total-.6)<.02
    assert total==77*4
    value['sources']['edu']['chunks'].pop('a')
    with pytest.raises(ValueError,match='remove/change'): publish_catalog(root,value)


def test_atomic_wait_and_exhaustion_never_repeat(tmp_path):
    root,value=catalog(tmp_path)
    reader=migrated(root); before=reader.state_dict()
    assert reader.try_batch(1000) is None
    assert reader.state_dict()==before
    value.update(complete=True,status='complete'); publish_catalog(root,value)
    with pytest.raises(GrowthExhausted): reader.next_batch(1000)
    assert reader.state_dict()==before
    data=reader.next_batch(80)
    assert len({tuple(row) for row in data})<80  # sources intentionally share synthetic payloads
    assert sum(reader.targets.values())==320
    binary=tmp_path/'edu'/'a'/'train.bin'; binary.write_bytes(b'x'*binary.stat().st_size)
    with pytest.raises(ValueError,match='Corrupt'): migrated(root).next_batch(1)


def test_filter_cross_source_heldout_and_near_overlap(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    index=DedupIndex(tmp_path/'registry.db')
    base=' '.join('word'+str(i) for i in range(40))
    old=annotate([{'text':base,'id':'old'}])[0]
    index.admit('old-repo',old); index.db.execute("UPDATE exact SET split='validation'"); index.db.commit()
    path=tmp_path/'raw.parquet'
    docs=[{'text':base,'id':'dup','edu_int_score':5},
          {'text':base.upper(),'id':'near','edu_int_score':5},
          {'text':'a fresh educational paragraph','id':'low','edu_int_score':2},
          {'text':'accepted educational explanation','id':'good','edu_int_score':3}]
    pq.write_table(pa.Table.from_pylist(docs),path)
    item={'sequence':1,'source':'dclm','sha256':sha256(path)}; output=tmp_path/'filtered.parquet'
    with ThreadPoolExecutor(2) as pool:
        receipt=filter_source(index,item,path,output,pool)
        assert receipt['counts']=={'accepted':1,'quality_rejected':1,'empty_rejected':0,'exact_overlap':1,'near_overlap':1}
        assert pq.read_table(output).to_pylist()==[{'text':docs[3]['text'],'id':'good'}]
        assert filter_source(index,item,path,output,pool)==receipt
    index.db.close()


def test_adam_and_cursor_next_update_survive_restart(tmp_path):
    import torch
    torch.set_num_threads(1); root,_=catalog(tmp_path); reader=migrated(root)
    torch.manual_seed(13); model=torch.nn.Linear(4,1); opt=torch.optim.AdamW(model.parameters(),lr=.003)
    def update(reader,model,opt):
        batch=torch.tensor(reader.next_batch(5)[:,:-1],dtype=torch.float32)/100
        opt.zero_grad(); model(batch).square().mean().backward(); opt.step()
    update(reader,model,opt)
    payload={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'reader':reader.state_dict(),'rng':torch.get_rng_state()}
    torch.save(payload,tmp_path/'state.pt'); update(reader,model,opt)
    expected=copy.deepcopy(model.state_dict()); expected_cursor=reader.state_dict()
    fresh=MixtureTokenBlocks(root,4); state=torch.load(tmp_path/'state.pt',weights_only=True)
    fresh.load_state_dict(state['reader']); restored=torch.nn.Linear(4,1); restored.load_state_dict(state['model'])
    restarted=torch.optim.AdamW(restored.parameters()); restarted.load_state_dict(state['optimizer']); torch.set_rng_state(state['rng'])
    update(fresh,restored,restarted)
    for key,value in restored.state_dict().items(): torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    assert fresh.state_dict()==expected_cursor
    assert all(value['step'].item()==2 for value in restarted.state.values())


def test_parent_in_progress_never_launches(tmp_path):
    data=tmp_path/'data'; logs=data/'logs'/'old'; logs.mkdir(parents=True)
    atomic_json(logs/'status.json',{'phase':'training','tokens':1_800_000_000})
    settings={'data_dir':str(data),'growth_dir':str(tmp_path/'growth'),'parent_run':'old','run_name':'new','target_tokens':50_000_000_000}
    result=tick(settings,{},launch=True)
    assert result['stage']=='waiting-for-10b'
    assert not (tmp_path/'growth'/'controller-state.json').exists()
    assert tick(settings,{'stage':'completed'},launch=True)=={'stage':'completed'}


def test_gates_reject_failed_missing_and_regressed_updates():
    rows=[{'step':s,'loss':3.,'grad_norm':1.,'valid_targets_per_second':57000.,'max_rank_peak_allocated_gib':25.,
           'sources/replay_targets':0,'sources/edu/valid_targets':600*s,'sources/dclm/valid_targets':400*s} for s in range(101,141)]
    assert metric_gate(rows,100,140,55000)['updates']==40
    early=copy.deepcopy(rows)
    early[0].update({'sources/edu/valid_targets':10*8192,'sources/dclm/valid_targets':6*8192})
    assert metric_gate(early,100,140,55000)['updates']==40
    for bad in (rows[:-1],[{**m,'grad_norm':float('nan')} for m in rows], [{**m,'sources/replay_targets':1} for m in rows]):
        with pytest.raises(ValueError): metric_gate(bad,100,140,55000)
    report={'step':140,'original':3.,'sources':{'edu':3.,'dclm':4.},'baseline':{'original':3.,'edu':3.,'dclm':4.}}
    assert quality_gate(report,140)['dclm']==4
    with pytest.raises(ValueError,match='regressed'): quality_gate({**report,'original':3.04},140)


def test_token_worker_accepts_serialized_paths_and_preserves_resume_bytes(tmp_path):
    """Exercise the actual spawn boundary, tokenizer and packer without GPUs."""
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from microllama.common.paths import data_root
    from microllama.data_io.corpus import split_for
    from microllama.data_io.growth_prepare import tokenize_filtered
    import pyarrow as pa
    import pyarrow.parquet as pq
    tokenizer=data_root()/'tokenizer/tokenizer.json'
    if not tokenizer.exists():
        pytest.skip('Pinned tokenizer is required for this integration regression')
    train=next(f'training words {i}' for i in range(10000) if split_for(f'training words {i}')=='train')
    validation=next(f'validation words {i}' for i in range(10000) if split_for(f'validation words {i}')=='validation')
    source=tmp_path/'s000001.parquet'
    pq.write_table(pa.Table.from_pylist([{'text':train,'id':'train'},{'text':validation,'id':'validation'}]),source)
    item={'source':'edu','path':'sample/100BT/fixture.parquet','sha256':'upstream-sha','sequence':1}
    output=tmp_path/'tokens'
    args=(str(source),item,str(output),str(tokenizer),sha256(tokenizer),7,1)
    with ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn')) as pool:
        assert pool.submit(tokenize_filtered,*args).result(timeout=60)==str(output)
        before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
        assert pool.submit(tokenize_filtered,*args).result(timeout=60)==str(output)
    assert before=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    chunk=output/'s000001-000000';meta=json.loads((chunk/'manifest.json').read_text())
    assert meta['source_repo']==SOURCES['edu']['repo']
    for split in ('train','validation'):
        assert meta['document_sidecars'][split]==sha256(chunk/f'{split}.documents.jsonl')
        assert meta['splits'][split]['sha256']==sha256(chunk/f'{split}.bin')
