"""Index finite M5 receipts without loading MLX, weights, data or credentials."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(value):
    if isinstance(value,dict):return {k:compact(v) for k,v in value.items() if k!='per_parameter'}
    if isinstance(value,list):return [compact(v) for v in value]
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot-qualification',type=Path)
    args=p.parse_args()
    results=[]
    for pattern in ('ultra-speed-*','ultra-fused-projections-*','ultra-ordered-gpu-*',
                    'ultra-custom-nax-*','ultra-engram-ablation-*','ultra-fast-qualification-*',
                    'ultra-operator-profile-*','ultra-roofline-*',
                    'ultra-stock-final-screen-*','ultra-stock-qualification-*'):
        for root in sorted(args.root.glob(pattern)):
            for path in sorted(root.rglob('result.json')):
                results.append({'path':str(path),'sha256':digest(path),'receipt':compact(json.loads(path.read_text()))})
            for path in sorted(root.glob('qualification.json')):
                results.append({'path':str(path),'sha256':digest(path),'receipt':compact(json.loads(path.read_text()))})
    repo=Path(__file__).resolve().parents[2]
    runtime=repo/'.venv-mac/lib/python3.12/site-packages'
    artifacts={str(n):digest(runtime/n) for n in map(Path,(
        'mlx/core.cpython-312-darwin.so','mlx/lib/libmlx.dylib','mlx/lib/mlx.metallib',
        'mlx-0.32.4.dist-info/METADATA','mlx_metal-0.32.4.dist-info/METADATA')) if (runtime/n).is_file()}
    launches=[]
    for path in sorted(args.root.glob('ultra-stock-production-*/launch.json')):
        item={'path':str(path),'sha256':digest(path),'receipt':json.loads(path.read_text())}
        events=path.parent/'run/events.jsonl'
        item['training_started']=events.exists() and any(
            json.loads(line).get('phase')=='training-started' for line in events.open())
        status=path.parent/'run/worker-status.json'
        if status.exists():item['status']=json.loads(status.read_text())
        launches.append(item)
    summary={'production_started':any(v['training_started'] for v in launches),
             'production_launches':launches,'receipts':results,'installed_runtime_sha256':artifacts,
             'source_snapshots':{str(p.parent):digest(p) for p in args.root.glob('ultra-fast-qualification-*/source-snapshot/manifest.json')}}
    if args.snapshot_qualification:
        root=args.snapshot_qualification
        contract=json.loads((root/'qualification-plan.json').read_text())
        target=root/'source-snapshot';target.mkdir(exist_ok=False)
        manifest={}
        for name,expected in contract['fast_backend']['sources'].items():
            source=(repo/'microllama'/name).resolve()
            relative=source.relative_to(repo)
            if digest(source)!=expected:raise ValueError('Qualified source changed: '+name)
            out=target/relative;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(source.read_bytes())
            manifest[str(relative)]=expected
        source=Path(__file__).with_name('qualify_ultra_fast.py')
        if digest(source)!=contract['source_sha256']:raise ValueError('Qualifier source changed')
        relative=source.relative_to(repo);out=target/relative;out.parent.mkdir(parents=True,exist_ok=True)
        out.write_bytes(source.read_bytes());manifest[str(relative)]=digest(source)
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        summary['qualification_source_snapshot']=str(target)
    args.output.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'receipts':len(results),'output':str(args.output),'runtime_artifacts':len(artifacts)}))


if __name__=='__main__':main()
