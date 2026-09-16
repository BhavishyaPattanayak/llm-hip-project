"""Validate saved results against frozen IDs/folds and recompute every pooled metric."""
import json
import numpy as np
from src.data import read_tsv,key
from src.extension_io import new_json
from src.regression import metrics
from src.gap_exploration.data import load_case
from src.gap_exploration.representation_v2.data import ROOT


def main():
    checked=[];identities={}
    for dataset in ('natural_stories','geco'):
        for model in ('gpt2','distilgpt2'):
            label=f'{dataset}_{model}';case=load_case(dataset,model,'rt' if dataset=='natural_stories' else 'gaze')
            expected=[(key(r),int(r['fold'])) for r in case['rows']]
            if dataset in identities:assert identities[dataset]==expected
            identities[dataset]=expected
            y=np.array([float(r['observed_rt']) for r in case['rows']])
            primary_audit=json.loads((ROOT/'cases'/label/'audit.json').read_text())
            for phase in ('cases','sensitivity'):
                path=ROOT/phase/label;prediction=read_tsv(path/'predictions.tsv');summary=read_tsv(path/'summary.tsv')
                assert [(key(r),int(r['fold'])) for r in prediction]==expected
                np.testing.assert_array_equal(y,[float(r['observed_rt']) for r in prediction])
                for row in summary:
                    p=np.array([float(r[row['variant']+'_prediction']) for r in prediction])
                    assert np.isfinite(p).all()
                    m=metrics(y,p)
                    for field in ('rmse','r2'):assert abs(m[field]-float(row[field]))<1e-9
                audit=json.loads((path/'audit.json').read_text())
                for a in audit:
                    fold=a['fold'];train=np.array([i for i,(_,f) in enumerate(expected) if f!=fold])
                    test=np.array([i for i,(_,f) in enumerate(expected) if f==fold])
                    assert a['test_indices']==test.tolist()
                    assert all(p['n_samples']==len(train) and p['n_components']==16 for p in a['outer_pca'])
                    assert len(a['inner_pca'])==5
                    for inner in a['inner_pca']:
                        assert all(p['n_samples'] in (int(.8*len(train)),int(.8*len(train))+1) for p in inner['pca'])
                    if phase=='sensitivity':assert a['selected_unchanged']==primary_audit[fold]['combined_families']
                checked.append(dict(case=label,phase=phase,n=len(y),prediction_variants=len(summary),all_metrics_recomputed=True,exact_folds_and_rows=True))
            info=json.loads((ROOT/'cases'/label/'case.json').read_text())
            assert info['n_change']==0 and info['prior_reference_max_prediction_error']<1e-6
    registry=read_tsv(ROOT/'exploration_log.tsv')
    assert len(registry)==sum(r['prediction_variants'] for r in checked)
    new_json(ROOT/'artifact_validation.json',dict(cases=checked,registry_rows=len(registry),model_samples_identical=True))
    print(json.dumps(checked,indent=2))


if __name__=='__main__':main()
