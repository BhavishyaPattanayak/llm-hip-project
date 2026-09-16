"""All candidate choices are made inside outer training; writes only sandbox files."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.data import key
from src.regression import metrics
from src.nested_regression import compare
from src.extension_io import new_json,new_table
from src.gap_exploration.data import load_case,candidate_features
from src.gap_exploration.evaluation import evaluate,training_diagnostics
from src.gap_exploration.plan import FAMILIES,SELECTION


def run_case(dataset,model,target):
    label=f'{dataset}_{target}_{model}';root=Path('results/gap_exploration')
    out=root/'metadata'/label
    if out.exists():raise FileExistsError(f'Case already exists: {out}')
    case=load_case(dataset,model,target);rows=case['rows'];families=candidate_features(case)
    y=np.array([float(r['observed_rt']) for r in rows]);folds=np.array([int(r['fold']) for r in rows]);groups=np.array([int(r['item']) for r in rows])
    core=np.array([[float(r[c]) for c in case['core_columns']] for r in rows]);strong=np.array([[float(r[c]) for c in case['strong_columns']] for r in rows])
    diagnostics=training_diagnostics(core,families,y,folds)
    print(f'{label}: {len(rows)} frozen rows; historical reference {case["strongest"]}',flush=True)
    predictions,audit=evaluate(core,strong,y,folds,families,case['spline'])
    error=None
    if case['saved_prediction'] is not None:
        error=float(abs(predictions['strong']-case['saved_prediction']).max())
        if error>1e-6:raise ValueError(f'Historical strong reference did not reproduce: max difference {error}')
    out.mkdir(parents=True)
    summary=[];all_stability=[]
    for variant,pred in predictions.items():
        report,tables=compare(y,predictions['strong'],pred,folds,groups)
        core_comparison=compare(y,predictions['core'],pred,folds,groups)[0]
        delta=(y-predictions['strong'])**2-(y-pred)**2
        tail=np.argsort(abs(y-predictions['strong']))[-max(1,int(np.ceil(.01*len(y)))):]
        share=float(delta[tail].sum()/delta.sum()) if abs(delta.sum())>1e-8 else None
        summary.append(dict(dataset=dataset,target=target,model=model,variant=variant,n=len(rows),
            baseline=case['strongest'],baseline_rmse=report['baseline']['rmse'],rmse=report['augmented']['rmse'],r2=report['augmented']['r2'],
            improvement_ms=report['rmse_improvement_ms'],improvement_percent=report['rmse_improvement_percent'],delta_r2=report['delta_r2'],
            fold_wins=report['fold_wins'],context_wins=report['context_wins'],context_win_fraction=report['context_win_fraction'],
            improvement_over_entropy_controls_percent=core_comparison['rmse_improvement_percent'],
            net_sse_gain_from_largest_1pct_reference_errors_fraction=share if share is not None else '',
            outcome='improves' if report['rmse_improvement_ms']>1e-10 else 'worsens' if report['rmse_improvement_ms'] < -1e-10 else 'reference/equivalent'))
        for grouping,table in tables.items():
            all_stability.extend(dict(row,variant=variant,grouping=grouping) for row in table)
    prediction_rows=[]
    for i,r in enumerate(rows):
        identity={c:r[c] for c in ('item','zone','word','fold')}
        if dataset=='geco':identity.update({c:r[c] for c in ('WORD_ID','PART','TRIAL')})
        prediction_rows.append(dict(identity,observed_rt=float(y[i]),n_bpe=int(float(r['n_bpe'])),word_length=float(r['word_length']),
            surprisal=float(r[f"{model}_surprisal"]),entropy_bits=float(r['entropy_bits']),**{f'{name}_prediction':float(v[i]) for name,v in predictions.items()},
                                    **{f'{name}_residual':float(y[i]-v[i]) for name,v in predictions.items()}))
    new_table(root/'predictions'/f'{label}.tsv',prediction_rows)
    new_table(out/'summary.tsv',summary);new_table(out/'stability.tsv',all_stability)
    new_table(out/'training_diagnostics.tsv',diagnostics)
    new_json(out/'audit.json',audit)
    new_json(out/'case.json',dict(dataset=dataset,target=target,model=model,n=len(rows),sample_source=case['source'],
        historical_reference=case['strongest'],reference_reproduction_max_abs_error=error,
        secondary_total_policy='Same gaze sample, saved folds, and historical gaze architecture; total target only, no completed total baseline existed' if target=='total' else None,
        spline=case['spline'],core_columns=case['core_columns'],strong_columns=case['strong_columns'],
        family_definitions=FAMILIES,selection=SELECTION,additional_exclusions=0,
        estimator='Nested ridge using frozen 25-alpha grid and five inner folds; NumPy float64 CPU (MPS lacks float64)',
        missing_policy='Lag-2 context absence: zero plus availability flag. No outcome imputation or row drops.'))
    print({r['variant']:round(r['improvement_percent'],4) for r in summary},flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['natural_stories','geco','all'],default='all')
    p.add_argument('--model',choices=['gpt2','distilgpt2'])
    p.add_argument('--target',choices=['gaze','total','both'],default='both')
    args=p.parse_args()
    for ds in ['natural_stories','geco'] if args.dataset=='all' else [args.dataset]:
        for target in ['rt'] if ds=='natural_stories' else ['gaze','total'] if args.target=='both' else [args.target]:
            for model in [args.model] if args.model else ['gpt2','distilgpt2']:
                run_case(ds,model,target)


if __name__=='__main__':main()
