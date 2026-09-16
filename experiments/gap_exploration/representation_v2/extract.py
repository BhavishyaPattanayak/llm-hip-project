"""Offline MPS extraction; all outputs are new, exclusive files."""
import argparse,json,time
from pathlib import Path
from itertools import groupby
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from src.data import read_regions,read_tsv,key
from src.representations import trajectory
from src.extension_io import new_json
from src.gap_exploration.data import load_case
from src.gap_exploration.representation_v2.extraction import extract


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['gpt2','distilgpt2'])
    p.add_argument('--device',default='mps');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    root=Path('results/gap_exploration/representation_v2')/('smoke' if args.smoke else 'features')
    root.mkdir(exist_ok=True)
    for ds in ('natural_stories','geco'):
        if (root/f'{ds}_{args.model}.npz').exists():raise FileExistsError('Extraction already exists')
    if args.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable in sandbox; run with GPU access')
    torch.manual_seed(2026)
    base=json.loads(Path(f'results/{args.model}_surprisal.json').read_text());revision=base['model_revision']
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=revision,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=revision,local_files_only=True).to(args.device).eval()
    assert model.config._commit_hash==revision
    for ds in ('natural_stories','geco'):
        case=load_case(ds,args.model,'rt' if ds=='natural_stories' else 'gaze')
        eligible={key(r):r for r in case['rows']}
        rows=sorted(read_regions() if ds=='natural_stories' else read_tsv('results/geco/processed/regions.tsv'),key=key)
        if args.smoke:rows=[r for r in rows if int(r['item'])==int(rows[0]['item'])][:32]
        contexts=[list(g) for _,g in groupby(rows,key=lambda r:int(r['item']))]
        pieces=[];start=time.perf_counter();max_error=0.
        for i,context in enumerate(contexts,1):
            if not any(key(r) in eligible for r in context):continue
            result,pooled=extract(context,tokenizer,model,args.device,eligible,base['stride'],ds=='geco')
            for j,k in enumerate(result['keys']):
                reference=eligible[tuple(k)]
                assert result['n_bpe'][j]==int(float(reference['n_bpe']))
                for c,v in trajectory(pooled[j]).items():
                    if c.startswith(('cosine_layer_','relative_l2_layer_')):
                        error=abs(v-float(reference[c]));max_error=max(max_error,error)
                        if error>2e-4:raise ValueError(f'Original trajectory mismatch {k} {c} {error}')
            if args.smoke:
                repeat,_=extract(context,tokenizer,model,args.device,eligible,base['stride'],ds=='geco')
                for name in result:np.testing.assert_allclose(result[name],repeat[name],rtol=0,atol=0)
            pieces.append(result)
            if i==1 or i%25==0 or i==len(contexts):print(ds,args.model,f'{i}/{len(contexts)}',round(time.perf_counter()-start,1),'seconds',flush=True)
        arrays={name:np.concatenate([r[name] for r in pieces]) for name in pieces[0]}
        if not args.smoke:assert [tuple(k) for k in arrays['keys']]==list(eligible)
        if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('Nonfinite extraction')
        with (root/f'{ds}_{args.model}.npz').open('xb') as f:np.savez(f,**arrays)
        new_json(root/f'{ds}_{args.model}.json',dict(model=args.model,dataset=ds,revision=revision,device=args.device,
            seconds=time.perf_counter()-start,n=len(arrays['keys']),smoke=args.smoke,max_original_trajectory_error=max_error,
            hidden_layers=[model.config.n_layer//3,2*model.config.n_layer//3,model.config.n_layer],
            state_policy='Mean pooled BPE raw residual states; final pre ln_f. Identical source token IDs, windows and ownership.',
            context_policy='Matched full-window position IDs; truncate prior BPE context to 0/16/64; retain complete current prefix. Average each owned piece exactly once.',
            relation_policy='Current owned-piece mean vs complete preceding regions within SAME forward window. Up to three preceding regions; no cross-context references.'))


if __name__=='__main__':main()
