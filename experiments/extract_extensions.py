"""Pinned model extraction; GECO all mode loads once for all three output kinds."""
import argparse
import json
from pathlib import Path
from itertools import groupby
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.data import read_tsv,read_regions,key
from src.entropy import extract_probabilities
from src.representations import extract_story,metadata,sha256,BLOCKS
from src.extension_io import new_json,new_table


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['natural_stories','geco'],required=True)
    p.add_argument('--model',choices=list(BLOCKS),required=True)
    p.add_argument('--features',choices=['all','entropy','surprisal','representations'],default='all')
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--smoke-words',type=int)
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    if args.dataset=='natural_stories' and args.features!='entropy':
        p.error('Natural Stories extraction is entropy-only; reuse completed features')
    base=json.loads(Path(f'results/{args.model}_surprisal.json').read_text())
    if args.dataset=='geco':
        stimulus=Path('results/geco/processed/regions.tsv'); rows=read_tsv(stimulus)
        root=Path('results/geco')
    else:
        stimulus=Path('data/naturalstories/naturalstories_RTS/all_stories.tok'); rows=read_regions()
        root=Path('results/extensions/natural_stories')
    rows=sorted(rows,key=key)
    if args.smoke_words is not None:
        if not 2<=args.smoke_words<=100:
            p.error('Smoke words must be 2–100')
        rows=[r for r in rows if int(r['item'])==int(rows[0]['item'])][:args.smoke_words]
        root=root/'smoke'
    kinds=['surprisal','representations','entropy'] if args.features=='all' else [args.features]
    paths={k:root/k/f'{args.model}.tsv' for k in kinds}
    if any(path.exists() or path.with_suffix('.json').exists() for path in paths.values()):
        raise FileExistsError('Requested output already exists; refusing overwrite')
    device=args.device
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    torch.manual_seed(2026)
    tok=AutoTokenizer.from_pretrained(args.model,revision=base['model_revision'],local_files_only=args.local_files_only)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=base['model_revision'],local_files_only=args.local_files_only).to(device).eval()
    if model.config.n_layer!=BLOCKS[args.model] or model.config.max_position_embeddings!=base['context_window']:
        raise ValueError('Model configuration mismatch')
    if model.config._commit_hash!=base['model_revision']:
        raise ValueError('Model revision mismatch')
    tables={k:[] for k in kinds}
    contexts=[list(g) for _,g in groupby(rows,key=lambda r:int(r['item']))]
    for index,context in enumerate(contexts,1):
        if set(kinds)&{'surprisal','entropy'}:
            probabilities=extract_probabilities(context,tok,model,device,base['stride'],'surprisal' in kinds,args.dataset=='geco')
            if args.smoke_words:
                again=extract_probabilities(context,tok,model,device,base['stride'],'surprisal' in kinds,args.dataset=='geco')
                np.testing.assert_allclose([r['entropy_bits'] for r in probabilities],
                                           [r['entropy_bits'] for r in again],rtol=0,atol=0,equal_nan=True)
            for k in ('surprisal','entropy'):
                if k in kinds:
                    cols=['item','zone','word','n_bpe']+(['WORD_ID','PART','TRIAL','WORD_ID_WITHIN_TRIAL'] if args.dataset=='geco' else [])
                    metric='surprisal' if k=='surprisal' else 'entropy_bits'
                    tables[k].extend({**{c:r[c] for c in cols},metric:r[metric], 'model':args.model} for r in probabilities)
        if 'representations' in kinds:
            compact=[{c:r[c] for c in ('item','zone','word','WORD_ID','PART','TRIAL','WORD_ID_WITHIN_TRIAL')} for r in context]
            tables['representations'].extend(extract_story(compact,tok,model,device,base['stride'],allow_internal_whitespace=True))
        print(f'{args.dataset}/{args.model}: context {index}/{len(contexts)} ({len(context)} regions)',flush=True)
    info=metadata(args.model,model,base['stride'],base,sha256(Path(f'results/{args.model}_surprisal.tsv')),len(rows))
    info.update(dataset=args.dataset,tokenizer=args.model,device=device,context_boundary='trial (PART,TRIAL)' if args.dataset=='geco' else 'story',
        stimulus_sha256=sha256(stimulus),smoke_only=bool(args.smoke_words),
        entropy_definition='Shannon bits immediately before FIRST BPE; no BOS, first word NaN',
        probability_pass='Surprisal and entropy share one pass; representations reuse established extractor in a second pass',
        context_count=len(contexts))
    if args.dataset=='geco':
        info['revision_reference_sha256']=info.pop('surprisal_sha256')
        info['revision_reference']='Existing Natural Stories model revision metadata'
    for k,path in paths.items():
        table=tables[k]
        if len(table)!=len(rows) or len({key(r) for r in table})!=len(rows):
            raise ValueError('Extraction row coverage error')
        if [(key(r),r['word']) for r in table]!=[(key(r),r['word']) for r in rows]:
            raise ValueError('Extraction alignment error')
        if args.dataset=='natural_stories':
            reference={key(r):r for r in read_tsv(f'results/{args.model}_surprisal.tsv')}
            if any(r['word']!=reference[key(r)]['word'] or int(r['n_bpe'])!=int(reference[key(r)]['n_bpe']) for r in table):
                raise ValueError('Entropy/surprisal alignment mismatch')
        new_table(path,table)
        if args.dataset=='geco':
            corresponding=root/'surprisal'/f'{args.model}.tsv'
            info['surprisal_sha256']=sha256(corresponding) if corresponding.exists() else None
        new_json(path.with_suffix('.json'),dict(info,kind=k,
            multi_bpe_regions=sum(int(r['n_bpe'])>1 for r in table),
            first_region_nan_count=len(contexts) if k in ('surprisal','entropy') else 0))
    print(f'Saved {len(rows)} regions; no existing outputs overwritten.')


if __name__=='__main__':
    main()
