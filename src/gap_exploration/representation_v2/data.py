"""Exact frozen samples and stricter two-word-history controls."""
from pathlib import Path
import numpy as np
from src.data import read_tsv,key
from src.gap_exploration.data import load_case,candidate_features

ROOT=Path('results/gap_exploration/representation_v2')


def load(dataset,model):
    case=load_case(dataset,model,'rt' if dataset=='natural_stories' else 'gaze');rows=case['rows']
    words={key(r):r for r in read_tsv('results/human_predictors.tsv' if dataset=='natural_stories' else 'results/geco/processed/regions.tsv')}
    source={key(r):r for r in read_tsv(f'results/{model}_surprisal.tsv' if dataset=='natural_stories' else f'results/geco/surprisal/{model}.tsv')}
    lexical=[]
    for r in rows:
        item,zone=key(r);k=(item,zone-2);w=words[k]
        f=float(w['log_frequency']) if w['log_frequency'] else float('nan')
        lexical.append([float(w['word_length']),f if np.isfinite(f) else 0.,float(source[k]['n_bpe']),float(np.isfinite(f))])
    core=np.array([[float(r[c]) for c in case['core_columns']] for r in rows])
    strong=np.array([[float(r[c]) for c in case['strong_columns']] for r in rows])
    history=np.c_[lexical,candidate_features(case)['extended_spillover']]
    baseline=np.c_[core,history]
    # Put spline variables at the end, exactly as in the original analysis.
    original=np.c_[baseline,strong[:,core.shape[1]:]]
    with np.load(ROOT/'features'/f'{dataset}_{model}.npz') as f:features={k:f[k] for k in f.files}
    expected=np.array([key(r) for r in rows])
    if not np.array_equal(features.pop('keys'),expected):raise ValueError('Feature/sample row identity mismatch')
    if not np.array_equal(features.pop('n_bpe'),[int(float(r['n_bpe'])) for r in rows]):raise ValueError('BPE mismatch')
    if not all(np.isfinite(a).all() for a in features.values()):raise ValueError('Nonfinite feature; cannot drop observations')
    return case,baseline,original,features
