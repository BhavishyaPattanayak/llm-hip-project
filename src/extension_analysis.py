"""Assemble new analyses without changing completed Natural Stories artifacts."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
from src.data import read_tsv,key
from src.geco import lag_columns
from src.regression import predictors,cross_validate,make_folds
from src.nested_regression import nested_predict,compare,ALPHAS,SPLINE
from src.representations import BLOCKS,feature_names,sha256
from src.extension_io import require_new,new_table,new_json


def merge_exact(rows,features,columns):
    lookup={key(r):r for r in features}
    if len(lookup)!=len(features):
        raise ValueError('Duplicate feature IDs')
    result=[]
    for r in rows:
        f=lookup.get(key(r))
        if f is None or f['word']!=r['word']:
            raise ValueError(f'Feature alignment disagreement {key(r)}')
        result.append(dict(r,**{c:f[c] for c in columns}))
    return result


def extraction_table(path,model):
    info=json.loads(path.with_suffix('.json').read_text())
    base=json.loads(Path(f'results/{model}_surprisal.json').read_text())
    if info['smoke_only'] or info['model_revision']!=base['model_revision']:
        raise ValueError('Smoke data or wrong model revision')
    rows=read_tsv(path)
    if len(rows)!=info['rows'] or len({key(r) for r in rows})!=len(rows):
        raise ValueError('Extraction metadata/coverage mismatch')
    metric='entropy_bits' if 'entropy_bits' in rows[0] else 'surprisal' if 'surprisal' in rows[0] else None
    if metric:
        for r in rows:
            v=float(r[metric])
            if not (np.isnan(v) if int(r['zone'])==1 else np.isfinite(v) and v>=0):
                raise ValueError('Invalid extraction boundary/value')
    return rows


def natural_rows(model,stage):
    rows=read_tsv(f'results/representations/analysis/{model}_joined.tsv')
    bpe=read_tsv(f'results/representations/analysis/bpe_control/{model}_predictions.tsv')
    rows=merge_exact(rows,bpe,['n_bpe','prev_n_bpe'])
    ref={key(r):r for r in read_tsv(f'results/regression/{model}_heldout_residuals.tsv')}
    if len(rows)!=len(ref):
        raise ValueError('Unexpected Natural Stories sample')
    for r in rows:
        if r['fold']!=ref[key(r)]['fold']:
            raise ValueError('Saved fold mismatch')
    if stage=='entropy':
        entropy=extraction_table(Path(f'results/extensions/natural_stories/entropy/{model}.tsv'),model)
        entropy=lag_columns(entropy,['entropy_bits'])
        rows=merge_exact(rows,entropy,['entropy_bits','prev_entropy_bits'])
    return sorted(rows,key=key),{key(r):r for r in bpe}


def geco_rows(model,dv):
    rows=read_tsv('results/geco/processed/regions.tsv')
    for kind,cols in [('surprisal',['surprisal','n_bpe']),('entropy',['entropy_bits']),
                       ('representations',feature_names(BLOCKS[model]))]:
        table=extraction_table(Path(f'results/geco/{kind}/{model}.tsv'),model)
        if len(table)!=len(rows):
            raise ValueError('Incomplete GECO extraction')
        rows=merge_exact(rows,table,cols)
    rows=lag_columns(rows,['surprisal','n_bpe','entropy_bits'])
    return [dict(r,observed_rt=r[f'mean_{dv}'],**{f'{model}_surprisal':r['surprisal'],
            f'prev_{model}_surprisal':r['prev_surprisal']}) for r in rows]


def complete_cases(rows,required):
    good=[]; excluded=[]; counts=Counter()
    for r in rows:
        reasons=[]
        for c in required:
            try:
                valid=np.isfinite(float(r[c]))
            except (ValueError,TypeError):
                valid=False
            if not valid:
                reasons.append(c); counts[c]+=1
        if reasons:
            excluded.append(dict(item=r['item'],zone=r['zone'],word=r['word'],reasons=';'.join(reasons)))
        else:
            good.append(r)
    return good,excluded,dict(counts)


def run_analysis(dataset,model,stage='trajectory',dv='gaze'):
    if dataset=='natural_stories':
        rows,saved=natural_rows(model,stage)
        out=Path(f'results/extensions/natural_stories/{"entropy_trajectory" if stage=="entropy" else "trajectory"}/{model}')
    else:
        rows=geco_rows(model,dv); saved=None
        out=Path(f'results/geco/extensions/{dv}/{model}')
    if out.exists():
        raise FileExistsError(out)
    base=predictors(model)+['n_bpe','prev_n_bpe']
    entropy=dataset=='geco' or stage=='entropy'
    entbase=base+['entropy_bits','prev_entropy_bits']
    required=(entbase if entropy else base)+feature_names(BLOCKS[model])+['observed_rt']
    total=len(rows); rows,excluded,counts=complete_cases(rows,required)
    rows=sorted(rows,key=key)
    if len(rows)<100:
        raise ValueError('Too few eligible observations')
    if dataset=='natural_stories':
        folds=np.array([int(r['fold']) for r in rows])
    else:
        # Common complete-case sample across all precommitted variants, per DV.
        folds=make_folds(len(rows),seed=2026)
        for r,f in zip(rows,folds): r['fold']=int(f)
        other='distilgpt2' if model=='gpt2' else 'gpt2'
        otherpath=Path(f'results/geco/extensions/{dv}/{other}/sample.tsv')
        if otherpath.exists():
            otherrows=read_tsv(otherpath)
            if [(key(r),int(r['fold'])) for r in rows]!=[(key(r),int(r['fold'])) for r in otherrows]:
                raise ValueError('GECO model samples/folds differ')
    y=np.array([float(r['observed_rt']) for r in rows]); groups=np.array([int(r['item']) for r in rows])
    def matrix(cols): return np.array([[float(r[c]) for c in cols] for r in rows])
    predictions={}; specifications={}; audits={}
    if dataset=='natural_stories':
        predictions['bpe_baseline']=np.array([float(saved[key(r)]['baseline_predicted_rt']) for r in rows])
        predictions['existing_linear_cosine']=np.array([float(saved[key(r)]['cosine_predicted_rt']) for r in rows])
        predictions['existing_linear_relative_l2']=np.array([float(saved[key(r)]['relative_l2_predicted_rt']) for r in rows])
    else:
        predictions['lexical_surprisal'],audits['lexical_surprisal']=cross_validate(matrix(predictors(model)),y,folds)
        predictions['bpe_baseline'],audits['bpe_baseline']=cross_validate(matrix(base),y,folds)
    if entropy:
        predictions['entropy_baseline'],audits['entropy_baseline']=cross_validate(matrix(entbase),y,folds)
    baseline_specs=([('bpe_baseline',base)] if dataset=='geco' or stage!='entropy' else [])+([('entropy_baseline',entbase)] if entropy else [])
    for baseline,cols in baseline_specs:
        # Matched ridge baseline separates representation gains from a change of estimator.
        specs=[('ridge_controls',[],False,'ridge')]
        for measure in ('cosine','relative_l2'):
            thirds=[f'{t}_mean_{measure}' for t in ('early','middle','late')]
            specs.extend([(f'all_layer_{measure}',[f'{measure}_layer_{i}' for i in range(1,BLOCKS[model]+1)],False,'ridge'),
                          (f'spline_{measure}',thirds,True,'ridge')])
            if baseline=='entropy_baseline' or dataset=='geco':
                specs.append((f'linear_{measure}',thirds,False,'ols'))
        for label,extra,spline,estimator in specs:
            name=f'{baseline}__{label}'
            x=matrix(cols+extra)
            print(f'{dataset}/{dv}/{model}: {name}',flush=True)
            if estimator=='ridge':
                pred,audit=nested_predict(x,y,folds,len(cols),spline)
            else:
                pred,audit=cross_validate(x,y,folds)
            predictions[name]=pred; audits[name]=audit
            specifications[name]=dict(baseline=baseline,features=cols+extra,estimator=estimator,spline=spline)
    comparisons={}
    if dataset=='geco':
        comparisons['bpe_baseline']='lexical_surprisal'
    if entropy:
        comparisons['entropy_baseline']='bpe_baseline'
    for name in predictions:
        if name.startswith('existing_'): comparisons[name]='bpe_baseline'
    comparisons.update({name:spec['baseline'] for name,spec in specifications.items()})
    require_new(out)
    sample=[{c:r[c] for c in ['item','zone','word','fold','observed_rt']+(['WORD_ID','PART','TRIAL'] if dataset=='geco' else [])} for r in rows]
    new_table(out/'sample.tsv',sample)
    if excluded: new_table(out/'exclusions.tsv',excluded)
    report=dict(dataset=dataset,model=model,dv='mean_RT' if dataset=='natural_stories' else dv,
        original_rows=total,n=len(rows),excluded=len(excluded),exclusions_overlapping=counts,
        seed=2026,outer_folds='saved Natural Stories folds' if dataset=='natural_stories' else 'deterministic common-sample word folds',
        inner_folds=5,inner_seed='2026 + outer fold ID',alpha_grid=ALPHAS.tolist(),spline_configuration=SPLINE,
        coefficients='standardized per-outer-fold fits are descriptive only',comparisons={})
    for name,baseline in comparisons.items():
        summary,tables=compare(y,predictions[baseline],predictions[name],folds,groups)
        summary['baseline_name']=baseline
        report['comparisons'][name]=summary
        new_table(out/f'{name}_fold.tsv',tables['fold']); new_table(out/f'{name}_context.tsv',tables['context'])
        new_table(out/f'{name}_predictions.tsv',[dict(r,baseline_prediction=float(predictions[baseline][i]),
            augmented_prediction=float(predictions[name][i]),residual=float(y[i]-predictions[name][i])) for i,r in enumerate(sample)])
    # Also report each ridge augmentation against its separately tuned ridge baseline.
    report['matched_ridge_comparisons']={}
    for name,spec in specifications.items():
        if spec['estimator']=='ridge' and not name.endswith('ridge_controls'):
            matched=spec['baseline']+'__ridge_controls'
            report['matched_ridge_comparisons'][name]=compare(y,predictions[matched],predictions[name],folds,groups)[0]
    sources=([Path(f'results/representations/analysis/{model}_joined.tsv'),Path(f'results/representations/analysis/bpe_control/{model}_predictions.tsv'),Path(f'results/regression/{model}_heldout_residuals.tsv')] if dataset=='natural_stories' else [Path('results/geco/processed/regions.tsv')]+[Path(f'results/geco/{k}/{model}.tsv') for k in ('surprisal','representations','entropy')])
    if dataset=='natural_stories' and entropy: sources.append(Path(f'results/extensions/natural_stories/entropy/{model}.tsv'))
    new_json(out/'input_provenance.json',{str(p):sha256(p) for p in sources})
    new_json(out/'summary.json',report)
    new_json(out/'specifications.json',specifications)
    new_json(out/'fit_audits.json',audits)
    new_table(out/'selected_alphas.tsv',[dict(variant=k,fold=a['fold'],alpha=a['alpha']) for k,entries in audits.items() for a in entries if 'alpha' in a])
    from src.extension_figures import plot_run
    plot_run(out,rows,report,predictions)
    return report
