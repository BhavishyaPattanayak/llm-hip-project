"""Do not overwrite or select new representation families."""
import argparse,json,time
import numpy as np
from src.extension_io import new_json,new_table,require_new
from src.nested_regression import compare
from src.gap_exploration.representation_v2.data import load,ROOT
from src.gap_exploration.representation_v2.sensitivity import nuisance_controls,fixed_evaluation


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',action='store_true')
    parser.add_argument('--dataset',choices=['natural_stories','geco']);parser.add_argument('--model',choices=['gpt2','distilgpt2']);args=parser.parse_args()
    if args.plan:
        new_json(ROOT/'sensitivity_plan.json',dict(status='Post-hoc diagnostic motivated by large Natural Stories DistilGPT-2 PCA gains; written before diagnostic evaluation.',
            controls='Current/previous indicators of .!?, comma, colon/semicolon, double quotation, digits and initial uppercase; log1p region index; log1p prior BPE count; mean original local position /1024 and its square.',
            comparisons='Stricter baseline, original trajectory, fixed three-layer PCA family, and unchanged PRIMARY INNER-SELECTED combined families.',
            invariants='Same rows and folds, no outcome-based exclusions, no selection changes, train-only PCA/scaling/tuning. All primary results retained. No further control variants planned.'))
        return
    if not args.dataset or not args.model:parser.error('Need dataset and model')
    case,base,original,features=load(args.dataset,args.model);label=f'{args.dataset}_{args.model}'
    selected={a['fold']:a['combined_families'] for a in json.loads((ROOT/'cases'/label/'audit.json').read_text())}
    controls=nuisance_controls(case['rows'],args.dataset,args.model)
    augmented=np.c_[base,controls];trajectory=np.c_[augmented,original[:,base.shape[1]:]]
    y=np.array([float(r['observed_rt']) for r in case['rows']]);folds=np.array([r['fold'] for r in case['rows']]);groups=np.array([int(r['item']) for r in case['rows']])
    out=require_new(ROOT/'sensitivity'/label);start=time.perf_counter()
    pred,audit=fixed_evaluation(augmented,trajectory,y,folds,features,selected,case['spline'],
        progress=lambda fold:print('sensitivity',label,fold,round(time.perf_counter()-start,1),'seconds',flush=True))
    result=[];stability=[]
    for name,p in pred.items():
        report,tables=compare(y,pred['baseline'],p,folds,groups);original_report,_=compare(y,pred['original_trajectory'],p,folds,groups)
        result.append(dict(dataset=args.dataset,model=args.model,target=case['target'],variant=name,n=len(y),rmse=report['augmented']['rmse'],r2=report['augmented']['r2'],
            baseline_rmse=report['baseline']['rmse'],improvement_ms=report['rmse_improvement_ms'],improvement_percent=report['rmse_improvement_percent'],delta_r2=report['delta_r2'],
            improvement_over_original_ms=original_report['rmse_improvement_ms'],improvement_over_original_percent=original_report['rmse_improvement_percent'],
            fold_wins=report['fold_wins'],fold_wins_over_original=original_report['fold_wins'],context_wins=report['context_wins'],context_count=len(tables['context'])))
        for kind,table in tables.items():stability.extend([dict(variant=name,kind=kind,**r) for r in table])
    new_table(out/'summary.tsv',result);new_table(out/'stability.tsv',stability);new_json(out/'audit.json',audit)
    new_table(out/'predictions.tsv',[dict(item=r['item'],zone=r['zone'],word=r['word'],fold=int(folds[i]),observed_rt=float(y[i]),
        **{name+'_prediction':float(p[i]) for name,p in pred.items()}) for i,r in enumerate(case['rows'])])


if __name__=='__main__':main()
