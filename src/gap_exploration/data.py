"""Read frozen samples/folds. Never regenerate or overwrite them."""
import json
from pathlib import Path
import numpy as np
from src.data import read_tsv,key
from src.extension_analysis import natural_rows,geco_rows
from src.regression import predictors
from src.gap_exploration.extraction import COLUMNS


def source_directory(dataset,model):
    return Path(f'results/extensions/natural_stories/entropy_trajectory/{model}') if dataset=='natural_stories' else Path(f'results/geco/extensions/gaze/{model}')


def load_case(dataset,model,target):
    source=source_directory(dataset,model)
    report=json.loads((source/'summary.json').read_text())
    specifications=json.loads((source/'specifications.json').read_text())
    candidates={name:r for name,r in report['comparisons'].items() if name.startswith('entropy_baseline__') and not name.endswith('ridge_controls')}
    strongest=min(candidates,key=lambda name:candidates[name]['augmented']['rmse'])
    spec=specifications[strongest]
    if spec['estimator']!='ridge':raise ValueError('Selected historical model is not ridge; implement faithful reproduction before proceeding')
    full=natural_rows(model,'entropy')[0] if dataset=='natural_stories' else geco_rows(model,'gaze')
    lookup={key(r):r for r in full}
    sample=read_tsv(source/'sample.tsv')
    saved=read_tsv(source/f'{strongest}_predictions.tsv')
    if [(key(r),r['fold']) for r in sample]!=[(key(r),r['fold']) for r in saved]:raise ValueError('Frozen reference alignment mismatch')
    rows=[]
    for row in sample:
        r=dict(lookup[key(row)])
        if r['word']!=row['word'] or not np.isclose(float(r['observed_rt']),float(row['observed_rt']),rtol=0,atol=1e-10):raise ValueError('Frozen sample mismatch')
        r['fold']=int(row['fold'])
        if target=='total':r['observed_rt']=r['mean_total']
        if not np.isfinite(float(r['observed_rt'])):raise ValueError('Secondary DV unavailable on frozen sample: do not silently drop rows')
        rows.append(r)
    if len({key(r) for r in rows})!=len(rows):raise ValueError('Duplicate frozen keys')
    return dict(dataset=dataset,model=model,target=target,rows=rows,source=str(source),strongest=strongest,
        strong_columns=spec['features'],spline=spec['spline'],core_columns=predictors(model)+['n_bpe','prev_n_bpe','entropy_bits','prev_entropy_bits'],
        saved_prediction=None if target=='total' else np.array([float(r['augmented_prediction']) for r in saved]),
        historical_prediction=np.array([float(r['augmented_prediction']) for r in saved]),
        historical_rmse=candidates[strongest]['augmented']['rmse'])


def candidate_features(case):
    dataset,model=case['dataset'],case['model']
    path=Path(f'results/gap_exploration/features/full/{dataset}/{model}.tsv')
    info=json.loads(path.with_suffix('.json').read_text())
    base=json.loads(Path(f'results/{model}_surprisal.json').read_text())
    if info['smoke'] or info['model_revision']!=base['model_revision']:raise ValueError('Invalid distribution-feature provenance')
    distribution=read_tsv(path);d={key(r):r for r in distribution}
    if len(d)!=len(distribution):raise ValueError('Duplicate extracted features')
    spath=Path(f'results/{model}_surprisal.tsv') if dataset=='natural_stories' else Path(f'results/geco/surprisal/{model}.tsv')
    hpath=Path(f'results/extensions/natural_stories/entropy/{model}.tsv') if dataset=='natural_stories' else Path(f'results/geco/entropy/{model}.tsv')
    s={key(r):r for r in read_tsv(spath)};h={key(r):r for r in read_tsv(hpath)}
    if set(d)!=set(s) or set(h)!=set(s):raise ValueError('Extraction coverage differs from original sources')
    families={k:[] for k in ['distribution_shape','extended_spillover','nonlinear_uncertainty','trajectory_shape']}
    for r in case['rows']:
        k=key(r);previous=(k[0],k[1]-1);lag2=(k[0],k[1]-2)
        if d[k]['word']!=r['word'] or int(d[k]['n_bpe'])!=int(float(r['n_bpe'])):raise ValueError('Candidate word/BPE mismatch')
        families['distribution_shape'].append([float(d[pos][c]) for pos in (k,previous) for c in COLUMNS])
        a=float(s.get(lag2,{}).get('surprisal','nan'));b=float(h.get(lag2,{}).get('entropy_bits','nan'))
        available=np.isfinite(a) and np.isfinite(b)
        families['extended_spillover'].append([a if available else 0,b if available else 0,float(available)])
        nonlinear=[]
        for prefix in ('','prev_'):
            surprisal=float(r[f'{prefix}{model}_surprisal']);entropy=float(r[f'{prefix}entropy_bits'])
            nonlinear.extend([surprisal**2,entropy**2,surprisal*entropy,2**(-surprisal)])
        families['nonlinear_uncertainty'].append(nonlinear)
        shape=[]
        for measure in ('cosine','relative_l2'):
            profile=np.array([float(r[f'{measure}_layer_{i}']) for i in range(1,(12 if model=='gpt2' else 6)+1)])
            total=max(profile.sum(),1e-12)
            shape.extend([abs(np.diff(profile)).sum()/total,np.dot(profile,np.linspace(0,1,len(profile)))/total,profile.max()/total])
        families['trajectory_shape'].append(shape)
    result={name:np.array(values,float) for name,values in families.items()}
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite candidate: no row exclusions allowed')
    return result
