"""Nested candidate and combination selection without outer-test access."""
import hashlib
import numpy as np
from sklearn.model_selection import KFold
from src.nested_regression import Design,ridge_path,ALPHAS


def index_hash(indices):return hashlib.sha256(np.asarray(indices,dtype='<i8').tobytes()).hexdigest()


def design_matrix(core,strong,extra,spline,with_trajectory=True):
    # Preserve the historical spline placement: only its final three original trajectory columns are splined.
    if with_trajectory and spline:
        matrix=np.column_stack([strong[:,:core.shape[1]],extra,strong[:,core.shape[1]:]])
        return matrix,core.shape[1]+extra.shape[1],True
    original=strong if with_trajectory else core
    return np.column_stack([original,extra]),original.shape[1]+extra.shape[1],False


def tune(x,y,train,outer,n_linear,spline,alphas=ALPHAS):
    losses=np.zeros((5,len(alphas)));audit=[]
    for fold,(tr,va) in enumerate(KFold(5,shuffle=True,random_state=2026+int(outer)).split(train)):
        it,iv=train[tr],train[va]
        design=Design(n_linear,spline).fit(x[it])
        coef,intercept=ridge_path(design.transform(x[it]),y[it],alphas)
        prediction=design.transform(x[iv])@coef+intercept
        losses[fold]=((y[iv,None]-prediction)**2).sum(axis=0)
        audit.append(dict(inner_fold=fold,train_n=len(it),validation_n=len(iv),train_hash=index_hash(it),validation_indices=iv.tolist()))
    chosen=int(np.argmin(losses.sum(0)))
    return dict(alpha=float(alphas[chosen]),inner_sse=losses[:,chosen],inner_grid_sse=losses.sum(0).tolist(),inner_splits=audit)


def fit_predict(x,y,train,test,n_linear,spline,alpha):
    design=Design(n_linear,spline).fit(x[train])
    coef,intercept=ridge_path(design.transform(x[train]),y[train],[alpha])
    return (design.transform(x[test])@coef+intercept).ravel(),design.audit()


def evaluate(core,strong,y,folds,families,spline=False,alphas=ALPHAS):
    y=np.asarray(y,float);folds=np.asarray(folds,int)
    n=len(y);empty=np.empty((n,0))
    names=['core','strong']+list(families)+['best_new','combined_new','trajectory_combined']
    prediction={name:np.full(n,np.nan) for name in names};audit=[]
    for outer in sorted(set(folds)):
        train=np.flatnonzero(folds!=outer);test=np.flatnonzero(folds==outer)
        assert not np.intersect1d(train,test).size
        tuned={};matrices={}
        for name,extra,trajectory in [('core',empty,False),('strong',empty,True)]+[(name,x,True) for name,x in families.items()]:
            x,linear,sp=design_matrix(core,strong,extra,spline,trajectory)
            result=tune(x,y,train,outer,linear,sp,alphas)
            pred,prep=fit_predict(x,y,train,test,linear,sp,result['alpha'])
            prediction[name][test]=pred;tuned[name]=result;matrices[name]=(x,linear,sp)
            result['preprocessing']=prep
        accepted=[name for name in families if tuned[name]['inner_sse'].sum()<tuned['strong']['inner_sse'].sum()
                  and int((tuned[name]['inner_sse']<tuned['strong']['inner_sse']).sum())>=4]
        best=min(accepted,key=lambda name:tuned[name]['inner_sse'].sum()) if accepted else None
        for label,selected,trajectory in [('best_new',[best] if best else [],False),('combined_new',accepted,False),('trajectory_combined',accepted,True)]:
            extra=np.column_stack([families[name] for name in selected]) if selected else empty
            if not selected:
                fallback='strong' if trajectory else 'core';prediction[label][test]=prediction[fallback][test]
                tuned[label]=dict(fallback=fallback)
                continue
            x,linear,sp=design_matrix(core,strong,extra,spline,trajectory)
            result=tune(x,y,train,outer,linear,sp,alphas)
            prediction[label][test],prep=fit_predict(x,y,train,test,linear,sp,result['alpha'])
            result['preprocessing']=prep;tuned[label]=result
        for result in tuned.values():
            if 'inner_sse' in result:result['inner_sse']=result['inner_sse'].tolist()
        audit.append(dict(outer_fold=int(outer),train_hash=index_hash(train),test_indices=test.tolist(),
            selected_families=accepted,best_family=best,models=tuned))
    if not all(np.isfinite(v).all() for v in prediction.values()):raise ValueError('Missing held-out predictions')
    return prediction,audit


def training_diagnostics(core,families,y,folds):
    """Descriptive diagnostics from each outer-training subset only; not evidence of success."""
    rows=[]
    for outer in sorted(set(folds)):
        train=folds!=outer
        for family,x in families.items():
            for col in range(x.shape[1]):
                a=x[train,col];controls=core[train]
                correlations=[]
                for j in range(controls.shape[1]):
                    correlations.append(float(np.corrcoef(a,controls[:,j])[0,1]) if np.std(a)>0 and np.std(controls[:,j])>0 else 0.)
                others=[abs(float(np.corrcoef(a,x[train,j])[0,1])) for j in range(x.shape[1]) if j!=col and np.std(a)>0 and np.std(x[train,j])>0]
                rows.append(dict(fold=int(outer),family=family,column=col,n=int(train.sum()),
                    mean=float(a.mean()),sd=float(a.std()),minimum=float(a.min()),p05=float(np.quantile(a,.05)),
                    median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(a.max()),
                    max_abs_control_correlation=max(map(abs,correlations)),control_correlations=','.join(map(str,correlations)),
                    max_abs_within_family_correlation=max(others,default=0.)))
    return rows
