"""Export overlap screens from preserved full public evaluation documents."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


COUNTS = {'hellaswag': 10042, 'openbookqa': 500, 'winogrande': 1267,
          'arc_easy': 2376, 'arc_challenge': 1172, 'boolq': 3270, 'piqa': 1838}


def document_parts(task: str, doc: dict) -> list[str]:
    if task == 'hellaswag':
        question = doc['ctx']
        choices = doc['endings']
    elif task in {'arc_easy', 'arc_challenge', 'openbookqa'}:
        question = doc['question_stem'] if task == 'openbookqa' else doc['question']
        choices = doc['choices']['text']
    elif task == 'winogrande':
        question = doc['sentence']
        choices = [doc['option1'], doc['option2']]
    elif task == 'boolq':
        question = doc['question']
        choices = [doc['passage']]
    else:
        question = doc['goal']
        choices = [doc['sol1'], doc['sol2']]
    # Keep short questions separate from their much longer passages or answers.
    return list(dict.fromkeys([question, ' '.join([question, *choices])]))


def build(full_root: Path, output: Path, receipt: Path) -> dict:
    if output.exists() or receipt.exists():
        raise FileExistsError('Refusing to overwrite a frozen exclusion corpus')
    documents, sources = [], {}
    for task, expected in COUNTS.items():
        path = full_root / f'{task}.json'
        payload = path.read_bytes()
        result = json.loads(payload)
        samples = result['samples'][task]
        if len(samples) != expected:
            raise ValueError(f'{task}: expected {expected} full samples, found {len(samples)}')
        for index, sample in enumerate(samples):
            for part_index, content in enumerate(document_parts(task, sample['doc'])):
                documents.append({'id': f'{task}:{index}:{part_index}', 'text': content})
        sources[task] = {'samples': expected, 'sha256': hashlib.sha256(payload).hexdigest(),
                         'task_version': result['versions'][task]}
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(documents, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
    output.write_bytes(encoded)
    audit = {'schema': 'public-suite-science-exclusion-v1', 'sources': sources,
             'full_examples': sum(COUNTS.values()), 'text_segments': len(documents),
             'sha256': hashlib.sha256(encoded).hexdigest(),
             'screen_scope': 'Preserved full evaluation documents; no semantic-overlap guarantee'}
    receipt.write_text(json.dumps(audit, indent=2) + '\n')
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--receipt', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.full_root, args.output, args.receipt), indent=2))


if __name__ == '__main__':
    main()
