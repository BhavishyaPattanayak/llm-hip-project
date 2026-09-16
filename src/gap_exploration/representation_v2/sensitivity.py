"""Fixed post-hoc punctuation/position confound check, not a candidate search."""
import numpy as np
from itertools import groupby
from sklearn.model_selection import KFold
from src.data import key,read_tsv
from src.surprisal import context_windows
from src.nested_regression import ALPHAS
from .evaluation import prepare,ridge_predictions,index_hash


def nuisance_controls(rows,dataset,model):
    source=read_tsv(f'results/{model}_surprisal.tsv' if dataset=='natural_stories' else f'results/geco/surprisal/{model}.tsv')
    lookup={key(r):r for r in source};position={}
    for _,g in groupby(sorted(source,key=key),key=lambda r:key(r)[0]):
        context=list(g);counts=np.array([int(r['n_bpe']) for r in context]);owners=np.repeat(np.arange(len(context)),counts)
        total=len(owners);local=np.zeros(total)
        for begin,end,target in context_windows(total,1024,256):
            first=0 if begin==0 and target==1 else target
            local[first:end]=np.arange(first-begin,end-begin)
        prefix=np.r_[0,np.cumsum(counts)[:-1]]
        for i,r in enumerate(context):position[key(r)]=(float(prefix[i]),float(local[owners==i].mean()))
    result=[]
    for r in rows:
        item,zone=key(r);values=[]
        for lag in (0,1):
            word=lookup[(item,zone-lag)]['word']
            values.extend([float(any(c in word for c in '.!?')),float(',' in word),float(any(c in word for c in ':;')),
                float(any(c in word for c in '\"“”')),float(any(c.isdigit() for c in word)),float(bool(word) and word[0].isupper())])
        before,local=position[(item,zone)]
        values.extend([np.log1p(zone),np.log1p(before),local/1024,(local/1024)**2])
        result.append(values)
    return np.array(result)


def fixed_evaluation(base,original,y,folds,features,selections,spline=False,progress=None):
    names=['baseline','original_trajectory','layer_combination','combined_new']
    predictions={name:np.full(len(y),np.nan) for name in names};audit=[]
    for outer in sorted(set(folds)):
        train=np.flatnonzero(folds!=outer);test=np.flatnonzero(folds==outer)
        selected=selections[int(outer)];splits=[];pca_audit=[]
        for tr,va in KFold(5,shuffle=True,random_state=2026+int(outer)).split(train):
            it,iv=train[tr],train[va];ft,fv,pa=prepare(features,it,iv)
            splits.append((it,iv,ft,fv));pca_audit.append(dict(train_hash=index_hash(it),validation_hash=index_hash(iv),pca=pa))
        def matrix(name,idx,f):
            if name=='baseline':return base[idx]
            if name=='original_trajectory':return original[idx]
            chosen=['layer_combination'] if name=='layer_combination' else selected
            return np.column_stack([base[idx]]+[f[k] for k in chosen])
        tuned={}
        for name in names:
            losses=[]
            for it,iv,ft,fv in splits:
                p,_=ridge_predictions(matrix(name,it,ft),y[it],matrix(name,iv,fv),ALPHAS,
                    spline and name=='original_trajectory',base.shape[1])
                losses.append(np.sum((y[iv,None]-p)**2,axis=0))
            loss=np.sum(losses,axis=0);tuned[name]=dict(alpha=float(ALPHAS[np.argmin(loss)]),inner_sse=loss.tolist())
        ft,fv,pa=prepare(features,train,test)
        for name in names:
            p,prep=ridge_predictions(matrix(name,train,ft),y[train],matrix(name,test,fv),[tuned[name]['alpha']],
                spline and name=='original_trajectory',base.shape[1])
            predictions[name][test]=p.ravel();tuned[name]['preprocessing']=prep
        audit.append(dict(fold=int(outer),selected_unchanged=selected,train_hash=index_hash(train),test_indices=test.tolist(),inner_pca=pca_audit,outer_pca=pa,models=tuned))
        if progress:progress(int(outer))
    return predictions,audit
