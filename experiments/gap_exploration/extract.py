"""Run with -B to avoid changing frozen bytecode. Only sandbox outputs are written."""
import argparse
import json
import time
from pathlib import Path
from itertools import groupby
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.data import read_regions,read_tsv,key
from src.representations import sha256
from src.extension_io import new_json,new_table
from src.gap_exploration.extraction import extract_context,COLUMNS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['gpt2','distilgpt2'],required=True)
    p.add_argument('--dataset',choices=['natural_stories','geco','both'],default='both')
    p.add_argument('--device',choices=['auto','mps','cpu','cuda'],default='auto')
    p.add_argument('--smoke-words',type=int)
    p.add_argument('--local-files-only',action='store_true')
    args=p.parse_args()
    if args.smoke_words and not 2<=args.smoke_words<=100:raise ValueError('Smoke requires 2–100 words')
    device=args.device
    if device=='auto':device='mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS inaccessible in this process; use outside-sandbox MPS or explicit CPU/CUDA')
    base=json.loads(Path(f'results/{args.model}_surprisal.json').read_text())
    revision=base['model_revision']
    datasets=['natural_stories','geco'] if args.dataset=='both' else [args.dataset]
    root=Path('results/gap_exploration/features')/('smoke' if args.smoke_words else 'full')
    for ds in datasets:
        if (root/ds/f'{args.model}.tsv').exists():raise FileExistsError('Extraction output already exists')
    torch.manual_seed(2026)
    tok=AutoTokenizer.from_pretrained(args.model,revision=revision,local_files_only=args.local_files_only)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=revision,local_files_only=args.local_files_only).to(device).eval()
    if model.config._commit_hash!=revision:raise ValueError('Revision mismatch')
    for ds in datasets:
        stimulus=Path('results/geco/processed/regions.tsv') if ds=='geco' else Path('data/naturalstories/naturalstories_RTS/all_stories.tok')
        source=Path(f'results/geco/surprisal/{args.model}.tsv') if ds=='geco' else Path(f'results/{args.model}_surprisal.tsv')
        rows=sorted(read_tsv(stimulus) if ds=='geco' else read_regions(),key=key)
        if args.smoke_words:rows=[r for r in rows if int(r['item'])==int(rows[0]['item'])][:args.smoke_words]
        contexts=[list(g) for _,g in groupby(rows,key=lambda r:int(r['item']))]
        result=[];start=time.perf_counter()
        for i,context in enumerate(contexts,1):
            features=extract_context(context,tok,model,device,base['stride'],ds=='geco')
            if args.smoke_words:
                repeat=extract_context(context,tok,model,device,base['stride'],ds=='geco')
                np.testing.assert_allclose([[r[c] for c in COLUMNS] for r in features],[[r[c] for c in COLUMNS] for r in repeat],atol=0,rtol=0,equal_nan=True)
            result.extend(features)
            if i==1 or i%25==0 or i==len(contexts):print(f'{ds}/{args.model}/{device}: {i}/{len(contexts)} contexts, {time.perf_counter()-start:.1f}s',flush=True)
        reference={key(r):r for r in read_tsv(source)}
        assert len({key(r) for r in result})==len(rows)
        assert all(r['word']==reference[key(r)]['word'] and r['n_bpe']==int(reference[key(r)]['n_bpe']) for r in result)
        path=root/ds/f'{args.model}.tsv';new_table(path,result)
        new_json(path.with_suffix('.json'),dict(model=args.model,model_revision=revision,tokenizer_revision=revision,dataset=ds,device=device,
            rows=len(rows),seconds=time.perf_counter()-start,smoke=bool(args.smoke_words),features=COLUMNS,
            first_region_policy='NaN; no BOS',alignment='Existing region offsets and assigned surprisal windows',
            context_policy='story' if ds=='natural_stories' else 'trial',stride=base['stride'],context_window=base['context_window'],
            probability_position='immediately before FIRST BPE',stimulus_sha256=sha256(stimulus),surprisal_sha256=sha256(source)))


if __name__=='__main__':main()
