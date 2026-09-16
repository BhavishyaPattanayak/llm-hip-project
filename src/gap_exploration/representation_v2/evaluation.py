"""Fold-local PCA/reference distributions and inner-only representation selection."""
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from src.nested_regression import Design,ridge_path,ALPHAS
from src.gap_exploration.evaluation import index_hash


def array_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class HiddenTransform:
    def __init__(self,components=16):self.components=components
    def fit(self,h):
        self.pcas=[]
        for layer in range(h.shape[1]):
            p=PCA(n_components=self.components,svd_solver='randomized',iterated_power=3,random_state=2026)
            p.fit(np.asarray(h[:,layer],dtype=np.float64));self.pcas.append(p)
        return self
    def transform(self,h):
        scores=[];typical=[]
        for layer,p in enumerate(self.pcas):
            centered=np.asarray(h[:,layer],dtype=np.float64)-p.mean_
            z=centered@p.components_.T;scores.append(z)
            squared=np.sum(centered**2,axis=1)
            floor=.1*np.mean(p.explained_variance_)
            radius=np.sqrt(np.mean(z**2/(p.explained_variance_+floor),axis=1))
            residual=np.sqrt(np.maximum(squared-np.sum(z**2,axis=1),0)/centered.shape[1])
            typical.extend([np.sqrt(squared/centered.shape[1]),radius,residual])
        return dict(zip(['layer_early','layer_middle','layer_late'],scores),
                    layer_combination=np.column_stack(scores),typicality=np.column_stack(typical))
    def audit(self):
        return [dict(mean_hash=array_hash(p.mean_),components_hash=array_hash(p.components_),
            variance=p.explained_variance_.tolist(),explained_variance_ratio=p.explained_variance_ratio_.tolist(),
            n_samples=int(p.n_samples_),n_components=int(p.n_components_)) for p in self.pcas]


def prepare(features,train,validation,components=16):
    transform=HiddenTransform(components).fit(features['hidden'][train])
    tr={k:v[train] for k,v in features.items() if k!='hidden'}
    va={k:v[validation] for k,v in features.items() if k!='hidden'}
    tr.update(transform.transform(features['hidden'][train]));va.update(transform.transform(features['hidden'][validation]))
    return tr,va,transform.audit()


def ridge_predictions(xtrain,ytrain,xvalidation,alphas,spline=False,n_linear=None):
    design=Design(xtrain.shape[1] if n_linear is None else n_linear,spline).fit(xtrain)
    coef,intercept=ridge_path(design.transform(xtrain),ytrain,alphas)
    return design.transform(xvalidation)@coef+intercept,design.audit()


def qualify(tuned,names):
    accepted=[name for name in names if tuned[name]['loss'].sum()<tuned['baseline']['loss'].sum()
              and np.sum(tuned[name]['loss']<tuned['baseline']['loss'])>=4]
    best=min(accepted,key=lambda n:tuned[n]['loss'].sum()) if accepted else None
    combined=[n for n in accepted if not ('layer_combination' in accepted and n in ('layer_early','layer_middle','layer_late'))]
    return accepted,best,combined


def diagnostic(base,families,fold):
    records=[]
    for name,x in families.items():
        for j,a in enumerate(x.T):
            correlation=[float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else 0. for b in base.T]
            records.append(dict(fold=int(fold),family=name,column=j,mean=float(a.mean()),sd=float(a.std()),
                p01=float(np.quantile(a,.01)),median=float(np.median(a)),p99=float(np.quantile(a,.99)),
                max_abs_control_correlation=max(map(abs,correlation)),control_correlations=','.join(map(str,correlation))))
    return records


def evaluate(baseline,original,y,folds,features,spline=False,alphas=ALPHAS,components=16,progress=None):
    y=np.asarray(y);folds=np.asarray(folds);n=len(y)
    names=[k for k in features if k!='hidden']+['layer_early','layer_middle','layer_late','layer_combination','typicality']
    models=['baseline','original_trajectory']+names+['best_new','combined_new']
    predictions={name:np.full(n,np.nan) for name in models};audits=[];diagnostics=[]
    for outer in sorted(set(folds)):
        train=np.flatnonzero(folds!=outer);test=np.flatnonzero(folds==outer)
        splits=[];pca_audit=[]
        for tr,va in KFold(5,shuffle=True,random_state=2026+int(outer)).split(train):
            it,iv=train[tr],train[va];ft,fv,audit=prepare(features,it,iv,components)
            splits.append((it,iv,ft,fv));pca_audit.append(dict(train_hash=index_hash(it),validation_hash=index_hash(iv),pca=audit))
        tuned={}
        for name in ['baseline','original_trajectory']+names:
            grid=[]
            for it,iv,ft,fv in splits:
                if name=='original_trajectory':xt,xv=original[it],original[iv]
                elif name=='baseline':xt,xv=baseline[it],baseline[iv]
                else:xt,xv=np.c_[baseline[it],ft[name]],np.c_[baseline[iv],fv[name]]
                pred,_=ridge_predictions(xt,y[it],xv,alphas,spline and name=='original_trajectory',baseline.shape[1])
                grid.append(np.sum((y[iv,None]-pred)**2,axis=0))
            grid=np.array(grid);choice=int(np.argmin(grid.sum(0)))
            tuned[name]=dict(alpha=float(alphas[choice]),loss=grid[:,choice],grid_sse=grid.sum(0).tolist())
        accepted,best,combined=qualify(tuned,names)
        if combined:
            grid=[]
            for it,iv,ft,fv in splits:
                xt=np.column_stack([baseline[it]]+[ft[name] for name in combined])
                xv=np.column_stack([baseline[iv]]+[fv[name] for name in combined])
                p,_=ridge_predictions(xt,y[it],xv,alphas)
                grid.append(np.sum((y[iv,None]-p)**2,axis=0))
            grid=np.array(grid);choice=int(np.argmin(grid.sum(0)))
            tuned['combined_new']=dict(alpha=float(alphas[choice]),loss=grid[:,choice],grid_sse=grid.sum(0).tolist())
        ft,fv,pa=prepare(features,train,test,components)
        diagnostics.extend(diagnostic(baseline[train],ft,outer))
        for name in ['baseline','original_trajectory']+names+(['combined_new'] if combined else []):
            if name=='original_trajectory':xt,xv=original[train],original[test]
            elif name=='baseline':xt,xv=baseline[train],baseline[test]
            elif name=='combined_new':
                xt=np.column_stack([baseline[train]]+[ft[k] for k in combined]);xv=np.column_stack([baseline[test]]+[fv[k] for k in combined])
            else:xt,xv=np.c_[baseline[train],ft[name]],np.c_[baseline[test],fv[name]]
            p,prep=ridge_predictions(xt,y[train],xv,[tuned[name]['alpha']],spline and name=='original_trajectory',baseline.shape[1])
            predictions[name][test]=p.ravel();tuned[name]['preprocessing']=prep
        predictions['best_new'][test]=predictions[best or 'baseline'][test]
        if not combined:predictions['combined_new'][test]=predictions['baseline'][test]
        for result in tuned.values():result['loss']=result['loss'].tolist()
        audits.append(dict(fold=int(outer),train_hash=index_hash(train),test_indices=test.tolist(),
            inner_pca=pca_audit,outer_pca=pa,accepted=accepted,best_family=best,combined_families=combined,models=tuned))
        if progress:progress(int(outer),accepted)
    if not all(np.isfinite(p).all() for p in predictions.values()):raise ValueError('Incomplete predictions')
    return predictions,audits,diagnostics
