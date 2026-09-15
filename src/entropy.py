"""Pre-first-BPE contextual entropy, sharing the established target windows."""
import math
import numpy as np
import torch
from src.surprisal import encode_regions,context_windows


def entropy_from_logits(logits):
    logp=torch.log_softmax(logits.float(),dim=-1)
    return -(logp.exp()*logp).sum(dim=-1)/math.log(2)


def extract_probabilities(rows,tokenizer,model,device,stride=256,with_surprisal=True,allow_internal_whitespace=False):
    ids,owners=encode_regions([r['word'] for r in rows],tokenizer,allow_internal_whitespace)
    first={}
    for i,owner in enumerate(owners):
        first.setdefault(owner,i)
    entropy=[float('nan')]*len(rows); token_scores=[float('nan')]*len(ids)
    with torch.inference_mode():
        for begin,end,target in context_windows(len(ids),model.config.max_position_embeddings,stride):
            x=torch.tensor([ids[begin:end]],device=device)
            logits=model(input_ids=x,use_cache=False).logits[0]
            if with_surprisal:
                loss=torch.nn.functional.cross_entropy(logits[target-begin-1:-1].float(),x[0,target-begin:],reduction='none')
                token_scores[target:end]=(loss/math.log(2)).cpu().tolist()
            eligible=[(owner,i) for owner,i in first.items() if target<=i<end]
            if eligible:
                positions=torch.tensor([i-begin-1 for owner,i in eligible],device=device)
                values=entropy_from_logits(logits[positions]).cpu().tolist()
                for (owner,_),value in zip(eligible,values):
                    entropy[owner]=value
    sums=np.zeros(len(rows)); counts=np.bincount(owners,minlength=len(rows))
    if with_surprisal:
        np.add.at(sums,owners,token_scores)
        sums[0]=float('nan')
    result=[]
    for i,r in enumerate(rows):
        if i and (not math.isfinite(entropy[i]) or entropy[i]<0):
            raise ValueError('Invalid pre-first-BPE entropy')
        result.append(dict(r,n_bpe=int(counts[i]),entropy_bits=entropy[i],
                           **({'surprisal':float(sums[i])} if with_surprisal else {})))
    return result
