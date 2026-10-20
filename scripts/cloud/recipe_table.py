"""Render the documented dataset table from the recipe the launcher reads."""
import argparse
from pathlib import Path
from .reproduce import DEFAULT, load_recipe


def render(recipe):
    initial=recipe['initial']; growth=recipe['growth']
    lines=['# Selected dataset shards','',
        '- Generated from [the exact training recipe](../configs/data-recipes/edu-8k-engram.json).',
        '- The reproduction command downloads and checks these files automatically.',
        '', '## First 10B', '',
        f"- FineWeb-Edu `sample/10BT`: all {len(initial['files'])} shards.",
        f"- Revision: `{initial['revision']}`.",
        '', '| Dataset path | Bytes | SHA-256 |','| --- | ---: | --- |']
    for item in initial['files']:
        url=f"https://huggingface.co/datasets/{initial['repo']}/resolve/{initial['revision']}/{item['path']}"
        lines.append(f"| [{item['path']}]({url}) | {item['bytes']:,} | `{item['sha256']}` |")
    lines+=['','## Additional data toward 50B','',
        '- Frozen acquisition order; '+('the complete selection is published.' if growth['published'] else '**selection is incomplete.**'),
        '- New targets: 60% FineWeb-Edu / 40% DCLM-Edu.',
        '- DCLM-Edu filter: `edu_int_score >= 3`.',
        '- Old train/held-out overlap excluded; fixed validation splits.',
        '', '| Order | Dataset | Pinned revision | Dataset path | Bytes | SHA-256 |','| ---: | --- | --- | --- | ---: | --- |']
    for item in growth['files']:
        source=growth['sources'][item['source']]
        url=f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{item['path']}"
        lines.append(f"| {item['sequence']} | {source['repo']} | `{source['revision']}` | [{item['path']}]({url}) | {item['bytes']:,} | `{item['sha256']}` |")
    if growth.get('acquisition'):
        budget=growth['acquisition']
        lines+=['', '## Acquisition limits', '',
            f"- Selected raw total: {sum(f['bytes'] for f in growth['files']):,} bytes.",
            f"- Cumulative cap: {budget['raw_byte_cap']:,} bytes.",
            f"- Initial pilot cap: {budget['pilot_byte_cap']:,} bytes.",
            f"- Later batch cap: {budget['batch_byte_cap']:,} bytes.",
            f"- Free-space reserve: {budget['reserve_gib']}GiB.",
            '- Reproduction uses these explicit limits and selects no extra shards.']
    lines+=['', '## Add a later stage', '',
        '- Append verified paths, sizes and hashes in selection order.',
        '- Keep `growth.published=false` until the full selection is verified.',
        '- Regenerate with `python3 -m scripts.cloud.recipe_table`.',
        '- Preserve each published recipe version in Git.', '']
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=DEFAULT)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[2]/'docs/dataset-shards.md')
    args=p.parse_args(); recipe,_=load_recipe(args.recipe,10_000_000_000)
    args.output.write_text(render(recipe))

if __name__=='__main__': main()
