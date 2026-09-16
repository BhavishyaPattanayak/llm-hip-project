"""Four pre-first-BPE distribution descriptors; no change to original extractors."""
import math
import numpy as np
import torch
from src.surprisal import encode_regions,context_windows

COLUMNS=['target_log_rank','top10_mass','top_margin','collision']


def distribution_features(logits,targets):
    logp=torch.log_softmax(logits.float(),dim=-1)
    p=logp.exp()
    target=logits.gather(1,targets[:,None])
    # Competition rank: tied target logits share rank (1 + strictly higher count).
    rank=1+(logits>target).sum(dim=-1)
    top=p.topk(min(10,p.shape[-1]),dim=-1).values
    return torch.stack([torch.log1p(rank.float()),top.sum(-1),top[:,0]-top[:,1],(p*p).sum(-1)],dim=-1)


def extract_context(rows,tokenizer,model,device,stride=256,geco=False):
    ids,owners=encode_regions([r['word'] for r in rows],tokenizer,allow_internal_whitespace=geco)
    first={}
    for i,owner in enumerate(owners):first.setdefault(owner,i)
    values=np.full((len(rows),len(COLUMNS)),np.nan)
    visits=np.zeros(len(rows),int)
    with torch.inference_mode():
        for begin,end,target in context_windows(len(ids),model.config.max_position_embeddings,stride):
            x=torch.tensor([ids[begin:end]],device=device)
            logits=model(input_ids=x,use_cache=False).logits[0]
            selected=[(owner,i) for owner,i in first.items() if target<=i<end]
            # Bound temporary full-vocabulary probabilities, especially on unified memory.
            for start in range(0,len(selected),64):
                batch=selected[start:start+64]
                positions=torch.tensor([i-begin-1 for _,i in batch],device=device)
                targets=torch.tensor([ids[i] for _,i in batch],device=device)
                features=distribution_features(logits[positions],targets).cpu().numpy()
                for (owner,_),v in zip(batch,features):values[owner]=v;visits[owner]+=1
            del logits
    if len(rows)>1 and (not np.all(visits[1:]==1) or not np.isfinite(values[1:]).all()):
        raise ValueError('Distribution extraction coverage failure')
    return [dict(item=r['item'],zone=r['zone'],word=r['word'],n_bpe=owners.count(i),
                 **dict(zip(COLUMNS,values[i].tolist()))) for i,r in enumerate(rows)]
