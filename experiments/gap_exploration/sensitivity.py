"""Post-hoc diagnostic: does lag-2 surprisal/entropy add beyond lag-2 lexical/BPE controls?"""
from pathlib import Path
import argparse
import json
import numpy as np
from src.data import read_tsv,key
from src.extension_io import new_table,new_json
from src.nested_regression import compare
from src.gap_exploration.data import load_case,candidate_features
from src.gap_exploration.evaluation import design_matrix,tune,fit_predict


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--combined',action='store_true');args=parser.parse_args()
    suffix='lag2_lexical_combined_sensitivity' if args.combined else 'lag2_lexical_sensitivity'
    root=Path('results/gap_exploration');out=root/'metadata'/suffix
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    new_json(out/'plan.json',dict(status='Post-hoc confound diagnostic after observing lag-2 family lead; not a replacement for primary results',
        definition='Add lag-2 lexical/BPE controls; compare adding original inner-selected combined families (unchanged selections)' if args.combined else 'Add lag-2 displayed length, log frequency, BPE count and frequency availability to historical strong reference. Compare with same plus lag-2 surprisal/entropy/availability.',
        missing_policy='Missing lag-2 frequency encoded zero plus flag. No rows dropped, no DV imputation.',selection='Same frozen folds, five inner folds, fixed 25 alpha grid. No additional candidate search.'))
    summaries=[]
    for dataset,target in [('natural_stories','rt'),('geco','gaze'),('geco','total')]:
        for model in ('gpt2','distilgpt2'):
            label=f'{dataset}_{target}_{model}';case=load_case(dataset,model,target);rows=case['rows']
            words=read_tsv('results/human_predictors.tsv' if dataset=='natural_stories' else 'results/geco/processed/regions.tsv')
            words={key(r):r for r in words}
            source=read_tsv(f'results/{model}_surprisal.tsv' if dataset=='natural_stories' else f'results/geco/surprisal/{model}.tsv')
            source={key(r):r for r in source}
            lexical=[]
            for r in rows:
                item,zone=key(r);k=(item,zone-2);w=words[k]
                frequency=float(w['log_frequency']) if w['log_frequency']!='' else float('nan')
                lexical.append([float(w['word_length']),frequency if np.isfinite(frequency) else 0.,float(source[k]['n_bpe']),float(np.isfinite(frequency))])
            lexical=np.array(lexical);families=candidate_features(case);spill=families['extended_spillover']
            selections={a['outer_fold']:a['selected_families'] for a in json.loads((root/'metadata'/label/'audit.json').read_text())}
            core=np.array([[float(r[c]) for c in case['core_columns']] for r in rows]);strong=np.array([[float(r[c]) for c in case['strong_columns']] for r in rows])
            y=np.array([float(r['observed_rt']) for r in rows]);folds=np.array([int(r['fold']) for r in rows]);groups=np.array([int(r['item']) for r in rows])
            predictions={};audit=[]
            for variant,extra in [('lag2_lexical_baseline',lexical),('lag2_lexical_plus_spillover',np.c_[lexical,spill])]:
                pred=np.full(len(y),np.nan)
                for fold in range(10):
                    train=np.flatnonzero(folds!=fold);test=np.flatnonzero(folds==fold)
                    if args.combined and variant!='lag2_lexical_baseline':
                        selected=selections[fold]
                        extra=np.column_stack([lexical]+[families[name] for name in selected])
                    x,n_linear,spline=design_matrix(core,strong,extra,case['spline'],True)
                    tuned=tune(x,y,train,fold,n_linear,spline)
                    pred[test],_=fit_predict(x,y,train,test,n_linear,spline,tuned['alpha'])
                    audit.append(dict(variant=variant,fold=fold,alpha=tuned['alpha']))
                predictions[variant]=pred
            report,tables=compare(y,predictions['lag2_lexical_baseline'],predictions['lag2_lexical_plus_spillover'],folds,groups)
            summaries.append(dict(dataset=dataset,target=target,model=model,n=len(rows),baseline='historical strong reference + lag2 lexical/BPE controls',
                baseline_rmse=report['baseline']['rmse'],rmse=report['augmented']['rmse'],r2=report['augmented']['r2'],
                improvement_ms=report['rmse_improvement_ms'],improvement_percent=report['rmse_improvement_percent'],delta_r2=report['delta_r2'],fold_wins=report['fold_wins'],context_win_fraction=report['context_win_fraction']))
            new_table(root/'predictions'/f'{label}_{suffix}.tsv',[dict(item=r['item'],zone=r['zone'],word=r['word'],fold=int(folds[i]),observed_rt=float(y[i]),
                baseline_prediction=float(predictions['lag2_lexical_baseline'][i]),augmented_prediction=float(predictions['lag2_lexical_plus_spillover'][i]),
                residual=float(y[i]-predictions['lag2_lexical_plus_spillover'][i])) for i,r in enumerate(rows)])
            new_table(out/f'{label}_fold.tsv',tables['fold']);new_table(out/f'{label}_context.tsv',tables['context']);new_json(out/f'{label}_alphas.json',audit)
            print(label,report['rmse_improvement_percent'],flush=True)
    new_table(out/'results.tsv',summaries)


if __name__=='__main__':main()
