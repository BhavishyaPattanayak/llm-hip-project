"""Exact full-window ownership and matched-token, matched-position context contrasts."""
import numpy as np
import torch
from src.data import key
from src.surprisal import encode_regions, context_windows
from .features import geometry, distances


def extract(rows, tokenizer, model, device, eligible, stride=256, whitespace=False, batch_size=16):
    ids, owners = encode_regions([r['word'] for r in rows], tokenizer, whitespace)
    owners = np.asarray(owners); n = len(rows); layers = model.config.n_layer
    selected = [layers//3, 2*layers//3, layers]
    sums = np.zeros((n, layers+1, model.config.n_embd), np.float64)
    counts = np.zeros(n, int); visits = np.zeros(len(ids), int)
    shortened = np.zeros((n, 3, model.config.n_embd), np.float64)
    relations = np.zeros((n, 9), np.float64)
    included = {i for i,r in enumerate(rows) if key(r) in eligible}
    captured = {}
    def hook(module, args): captured['raw'] = args[0]
    handle = model.transformer.ln_f.register_forward_pre_hook(hook)
    starts = np.array([np.flatnonzero(owners == i)[0] for i in range(n)])
    windows = list(context_windows(len(ids), model.config.max_position_embeddings, stride))
    if len(ids) == 1: windows = [(0,1,1)]
    try:
        with torch.inference_mode():
            for begin,end,target in windows:
                first = 0 if begin == 0 and target == 1 else target
                output = model.transformer(input_ids=torch.tensor([ids[begin:end]], device=device),
                    output_hidden_states=True, use_cache=False, return_dict=True)
                states = np.stack([a[0].float().cpu().numpy() for a in (*output.hidden_states[:-1], captured['raw'])])
                own = owners[first:end]; local = first-begin
                np.add.at(sums, own, states[:,local:].transpose(1,0,2))
                np.add.at(counts, own, 1); visits[first:end] += 1
                tasks = []
                for word in sorted(set(own) & included):
                    positions = np.flatnonzero(owners[begin:end] == word)
                    scored = positions[positions >= local]
                    weight = len(scored)
                    current = states[selected][:,scored].mean(1)
                    predecessors = []
                    for lag in (1,2,3):
                        before = word-lag
                        if before < 0: continue
                        pos = np.flatnonzero(owners[begin:end] == before)
                        if not len(pos) or starts[before] < begin:
                            raise ValueError('Incomplete preceding region in assigned window')
                        predecessors.append(states[selected][:,pos].mean(1))
                    if len(predecessors) < 2: raise ValueError('Eligible row lacks two preceding words')
                    references = [predecessors[0], predecessors[1], np.mean(predecessors,axis=0)]
                    relations[word] += np.concatenate([distances(current,p)[:,0] for p in references])*weight
                    word_start = max(begin, int(starts[word])); word_end = begin+int(positions[-1])+1
                    for index,limit in enumerate((0,16,64)):
                        start = max(begin, word_start-limit)
                        if start == begin:
                            shortened[word,index] += states[-1,scored].sum(0)
                        else:
                            tasks.append((word,index,start,word_end,scored+begin-start))
                # Right padding is masked; position IDs retain their full-window values.
                tasks.sort(key=lambda t:t[3]-t[2])
                for offset in range(0,len(tasks),batch_size):
                    batch=tasks[offset:offset+batch_size]; width=max(t[3]-t[2] for t in batch)
                    tokens=np.full((len(batch),width),tokenizer.eos_token_id,np.int64)
                    positions=np.zeros_like(tokens);mask=np.zeros_like(tokens)
                    for j,(_,_,start,stop,_) in enumerate(batch):
                        length=stop-start;tokens[j,:length]=ids[start:stop]
                        positions[j,:length]=np.arange(start-begin,stop-begin);mask[j,:length]=1
                    model.transformer(input_ids=torch.tensor(tokens,device=device),
                        attention_mask=torch.tensor(mask,device=device),position_ids=torch.tensor(positions,device=device),
                        use_cache=False,return_dict=True)
                    raw=captured['raw'].float().cpu().numpy()
                    for j,(word,index,_,_,scored) in enumerate(batch):
                        shortened[word,index] += raw[j,scored].sum(0)
                del output, states
    finally: handle.remove()
    if not np.all(visits == 1) or not np.array_equal(counts,np.bincount(owners,minlength=n)):
        raise ValueError('BPE ownership/coverage failure')
    indices=sorted(included); pooled=sums[indices]/counts[indices,None,None]
    short=shortened[indices]/counts[indices,None,None]
    families=geometry(pooled)
    contrast=distances(pooled[:,-1,None],short)
    families['contextualization']=contrast[:,0]
    families['context_dependence']=contrast[:,1:].reshape(len(indices),4)
    rel=relations[indices]/counts[indices,None]
    families['previous_relations']=np.column_stack([rel,rel[:,[2,5,8]]-rel[:,[0,3,6]]])
    return dict(keys=np.array([key(rows[i]) for i in indices],int), n_bpe=counts[indices],
        hidden=pooled[:,selected].astype(np.float32), **families), pooled
