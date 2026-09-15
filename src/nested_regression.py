"""Nested ridge with fold-local scaling and optional trajectory-only cubic splines."""
import numpy as np
from sklearn.preprocessing import StandardScaler,SplineTransformer
from sklearn.model_selection import KFold
from src.regression import metrics

ALPHAS=np.logspace(-4,4,25)
SPLINE=dict(degree=3,n_knots=4,knots='quantile',include_bias=False)


class Design:
    def __init__(self,n_baseline,spline=False):
        self.n_baseline=n_baseline; self.spline=spline
    def fit(self,x):
        self.transformer=SplineTransformer(**SPLINE) if self.spline else None
        extra=self.transformer.fit_transform(x[:,self.n_baseline:]) if self.spline else x[:,self.n_baseline:]
        raw=np.column_stack([x[:,:self.n_baseline],extra])
        self.scaler=StandardScaler().fit(raw)
        return self
    def transform(self,x):
        extra=self.transformer.transform(x[:,self.n_baseline:]) if self.spline else x[:,self.n_baseline:]
        return self.scaler.transform(np.column_stack([x[:,:self.n_baseline],extra]))
    def audit(self):
        return dict(mean=self.scaler.mean_.tolist(),scale=self.scaler.scale_.tolist(),
            spline_knots=[b.t.tolist() for b in self.transformer.bsplines_] if self.spline else None,
            baseline_linear_columns=self.n_baseline)


def ridge_path(x,y,alphas):
    """Centered SVD fits all alphas efficiently; intercept is not penalized."""
    xm=x.mean(axis=0); ym=y.mean()
    u,s,vt=np.linalg.svd(x-xm,full_matrices=False)
    coefficients=vt.T @ ((s[:,None]/(s[:,None]**2+np.asarray(alphas)[None,:]))*(u.T@(y-ym))[:,None])
    return coefficients,ym-xm@coefficients


def nested_predict(x,y,folds,n_baseline,spline=False,alphas=ALPHAS):
    x=np.asarray(x,float); y=np.asarray(y,float); folds=np.asarray(folds,int)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Nested regression requires finite complete cases')
    prediction=np.full(len(y),np.nan); visits=np.zeros(len(y),int); audits=[]
    for outer in sorted(set(folds)):
        train=np.flatnonzero(folds!=outer); test=np.flatnonzero(folds==outer)
        loss=np.zeros(len(alphas)); inner_audit=[]
        for tr,va in KFold(n_splits=5,shuffle=True,random_state=2026+int(outer)).split(train):
            it,iv=train[tr],train[va]
            assert not np.intersect1d(test,np.r_[it,iv]).size
            design=Design(n_baseline,spline).fit(x[it])
            coef,intercept=ridge_path(design.transform(x[it]),y[it],alphas)
            error=y[iv,None]-(design.transform(x[iv])@coef+intercept)
            loss+=(error**2).sum(axis=0)
            inner_audit.append(dict(train_indices=it.tolist(),validation_indices=iv.tolist(),preprocessing=design.audit()))
        choice=int(np.argmin(loss)); alpha=float(alphas[choice])
        design=Design(n_baseline,spline).fit(x[train])
        coef,intercept=ridge_path(design.transform(x[train]),y[train],[alpha])
        prediction[test]=(design.transform(x[test])@coef+intercept).ravel()
        visits[test]+=1
        audits.append(dict(fold=int(outer),alpha=alpha,inner_mse=(loss/len(train)).tolist(),
            outer_train_indices=train.tolist(),outer_test_indices=test.tolist(),inner=inner_audit,
            preprocessing=design.audit(),standardized_coefficients=coef[:,0].tolist(),intercept=float(intercept[0])))
    assert np.all(visits==1) and np.isfinite(prediction).all()
    return prediction,audits


def compare(y,baseline,prediction,folds,groups):
    bm=metrics(y,baseline); am=metrics(y,prediction)
    gain=bm['rmse']-am['rmse']
    report=dict(baseline=bm,augmented=am,rmse_improvement_ms=gain,
        rmse_improvement_percent=100*gain/bm['rmse'],delta_r2=am['r2']-bm['r2'])
    tables={}
    for kind,labels in [('fold',folds),('context',groups)]:
        labels=np.asarray(labels)
        table=[]
        for label in sorted(set(labels)):
            mask=labels==label
            a=float(np.sqrt(np.mean((y[mask]-baseline[mask])**2)))
            b=float(np.sqrt(np.mean((y[mask]-prediction[mask])**2)))
            table.append(dict(group=str(label),n=int(mask.sum()),baseline_rmse=a,augmented_rmse=b,
                              rmse_improvement_ms=a-b,win=bool(b<a)))
        tables[kind]=table
        report[f'{kind}_wins']=sum(r['win'] for r in table)
        report[f'{kind}_win_fraction']=report[f'{kind}_wins']/len(table)
    return report,tables
