"""Four primary cases, exact frozen rows; never replace a completed case."""
import argparse,json,time
import numpy as np
from src.data import read_tsv,key
from src.extension_io import new_json,new_table,require_new
from src.regression import metrics
from src.nested_regression import compare
from src.gap_exploration.representation_v2.data import load,ROOT
from src.gap_exploration.representation_v2.evaluation import evaluate
from src.gap_exploration.representation_v2.features import DEFINITIONS


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['natural_stories','geco'])
    p.add_argument('--model',required=True,choices=['gpt2','distilgpt2']);args=p.parse_args()
    label=f'{args.dataset}_{args.model}';out=require_new(ROOT/'cases'/label)
    case,baseline,original,features=load(args.dataset,args.model);rows=case['rows']
    y=np.array([float(r['observed_rt']) for r in rows]);folds=np.array([r['fold'] for r in rows]);groups=np.array([int(r['item']) for r in rows])
    start=time.perf_counter()
    def progress(fold,selected):print(label,'fold',fold,'selected',','.join(selected),round(time.perf_counter()-start,1),'seconds',flush=True)
    prediction,audit,diagnostics=evaluate(baseline,original,y,folds,features,case['spline'],progress=progress)
    prior=read_tsv(f"results/gap_exploration/predictions/{args.dataset}_{case['target']}_{args.model}_lag2_lexical_sensitivity.tsv")
    assert [(key(r),int(r['fold'])) for r in prior]==[(key(r),int(r['fold'])) for r in rows]
    reference=np.array([float(r['augmented_prediction']) for r in prior])
    reproduction=float(np.max(abs(reference-prediction['original_trajectory'])))
    if reproduction>1e-6:raise ValueError(f'Prior stronger reference not reproduced: {reproduction}')
    summaries=[];stability=[]
    for name,pred in prediction.items():
        report,tables=compare(y,prediction['baseline'],pred,folds,groups)
        vs_original,_=compare(y,prediction['original_trajectory'],pred,folds,groups)
        accepted=sum(name in a['accepted'] for a in audit)
        summaries.append(dict(dataset=args.dataset,target=case['target'],model=args.model,variant=name,n=len(y),
            baseline_rmse=report['baseline']['rmse'],rmse=report['augmented']['rmse'],r2=report['augmented']['r2'],
            improvement_ms=report['rmse_improvement_ms'],improvement_percent=report['rmse_improvement_percent'],delta_r2=report['delta_r2'],
            original_trajectory_rmse=vs_original['baseline']['rmse'],improvement_over_original_ms=vs_original['rmse_improvement_ms'],
            improvement_over_original_percent=vs_original['rmse_improvement_percent'],delta_r2_over_original=vs_original['delta_r2'],
            fold_wins=report['fold_wins'],fold_wins_over_original=vs_original['fold_wins'],context_wins=report['context_wins'],
            context_count=len(tables['context']),inner_qualified_folds=accepted,
            definition=DEFINITIONS.get(name,{'baseline':'Strong nonrepresentation history controls','original_trajectory':case['strongest'],
            'best_new':'Inner-selected qualifying family or baseline fallback','combined_new':'Fixed inner-selected combination rule or baseline fallback'}.get(name)),
            retained='inner-only per fold' if accepted else 'not selected in any outer training set' if name in DEFINITIONS else 'reference/selection-rule',
            outcome='improves' if report['rmse_improvement_ms']>0 else 'worsens' if report['rmse_improvement_ms']<0 else 'reference/equivalent'))
        for kind,labels in [('fold',folds),('context',groups)]:
            for group in sorted(set(labels)):
                mask=labels==group;m=metrics(y[mask],pred[mask]);b=metrics(y[mask],prediction['baseline'][mask]);o=metrics(y[mask],prediction['original_trajectory'][mask])
                stability.append(dict(variant=name,kind=kind,group=int(group),n=int(mask.sum()),rmse=m['rmse'],r2=m['r2'],
                    improvement_ms=b['rmse']-m['rmse'],improvement_percent=100*(b['rmse']-m['rmse'])/b['rmse'],
                    delta_r2=m['r2']-b['r2'],improvement_over_original_ms=o['rmse']-m['rmse']))
    influence=[];error=(y-prediction['baseline'])**2;top=np.argsort(error)[-max(1,int(np.ceil(.01*len(y)))):]
    for name in prediction:
        gain=error-(y-prediction[name])**2;total=gain.sum()
        influence.append(dict(variant=name,net_sse_gain=float(total),largest_one_percent_error_gain_fraction=float(gain[top].sum()/total) if total else 0.))
    new_table(out/'summary.tsv',summaries);new_table(out/'stability.tsv',stability)
    new_table(out/'training_diagnostics.tsv',diagnostics);new_table(out/'influence.tsv',influence)
    new_json(out/'audit.json',audit)
    new_table(out/'predictions.tsv',[dict(item=r['item'],zone=r['zone'],word=r['word'],fold=int(folds[i]),observed_rt=float(y[i]),
        **{name+'_prediction':float(v[i]) for name,v in prediction.items()}) for i,r in enumerate(rows)])
    new_json(out/'case.json',dict(dataset=args.dataset,model=args.model,target=case['target'],n=len(rows),n_change=0,
        source=case['source'],original_trajectory=case['strongest'],prior_reference_max_prediction_error=reproduction,
        baseline_columns=case['core_columns']+['lag2_length','lag2_frequency','lag2_n_bpe','lag2_frequency_available','lag2_surprisal','lag2_entropy','lag2_model_available'],
        seconds=time.perf_counter()-start,selection='Inner folds only; never outer scores',pca_components=16))
    print(label,'complete',flush=True)


if __name__=='__main__':main()
