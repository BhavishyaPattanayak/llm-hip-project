"""Tokenization-control robustness; reads saved folds without generating folds."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.data import key, read_tsv, write_tsv
from src.regression import predictors, cross_validate, metrics

MODELS = ('gpt2', 'distilgpt2')


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(root).rglob('*') if p.is_file()}


def add_bpe_predictors(joined, trajectories, residuals):
    """Use the actual preceding region, including regions excluded from regression."""
    full = {key(r): r for r in trajectories}
    saved = {key(r): r for r in residuals}
    if len(full) != len(trajectories) or len(saved) != len(residuals):
        raise ValueError('Duplicate source keys')
    if len(joined) != len(saved) or {key(r) for r in joined} != set(saved):
        raise ValueError('Joined sample differs from established regression sample')
    if len({key(r) for r in joined}) != len(joined):
        raise ValueError('Duplicate joined keys')
    output = []
    for r in sorted(joined, key=key):
        item, zone = key(r)
        current = full.get((item, zone))
        previous = full.get((item, zone-1)) if zone > 1 else None
        if current is None or current['word'] != r['word'] or saved[item, zone]['word'] != r['word']:
            raise ValueError('Word/key disagreement')
        if int(r['fold']) != int(saved[item, zone]['fold']):
            raise ValueError('Fold differs from established regression')
        if not np.isclose(float(r['observed_rt']), float(saved[item, zone]['observed_rt']), rtol=0, atol=1e-10):
            raise ValueError('Observed RT disagreement')
        if 'n_bpe' in r and int(r['n_bpe']) != int(current['n_bpe']):
            raise ValueError('BPE count disagreement')
        n = float(current['n_bpe'])
        prev = float(previous['n_bpe']) if previous else float('nan')
        if n < 1 or not n.is_integer() or (previous and (prev < 1 or not prev.is_integer())):
            raise ValueError('Invalid BPE count')
        output.append(dict(r, n_bpe=n, prev_n_bpe=prev))
    return output


def evaluate(rows, model):
    names = predictors(model) + ['n_bpe', 'prev_n_bpe']
    folds = np.array([int(r['fold']) for r in rows])
    if set(folds) != set(range(10)):
        raise ValueError('Expected the ten established fold IDs')
    y = np.array([float(r['observed_rt']) for r in rows])
    output = [dict(item=r['item'], zone=r['zone'], word=r['word'], fold=int(folds[i]),
                   observed_rt=float(y[i]), **{c: float(r[c]) for c in names}) for i,r in enumerate(rows)]
    report, audits, predictions = {}, {}, {}
    for variant in ('baseline', 'cosine', 'relative_l2'):
        cols = names + ([] if variant == 'baseline' else
                        [f'{third}_mean_{variant}' for third in ('early', 'middle', 'late')])
        x = np.array([[float(r[c]) for c in cols] for r in rows])
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Missing/nonfinite required predictors: refusing to silently change eligible N')
        pred, audit = cross_validate(x, y, folds)
        predictions[variant] = pred
        audits[variant] = dict(predictors=cols, folds=audit)
        report[variant] = metrics(y, pred)
        if variant != 'baseline':
            gain = report['baseline']['rmse']-report[variant]['rmse']
            report[variant].update(rmse_improvement_ms=gain,
                rmse_improvement_percent=100*gain/report['baseline']['rmse'],
                delta_r2=report[variant]['r2']-report['baseline']['r2'])
        for i,r in enumerate(output):
            r[f'{variant}_predicted_rt'] = float(pred[i])
            r[f'{variant}_residual'] = float(y[i]-pred[i])
    stability = {}
    for grouping in ('fold', 'item'):
        groups = np.array([int(r[grouping]) for r in rows])
        table = []
        for group in sorted(set(groups)):
            mask = groups == group
            a = metrics(y[mask], predictions['baseline'][mask])['rmse']
            b = metrics(y[mask], predictions['cosine'][mask])['rmse']
            table.append(dict(group=int(group), n=int(mask.sum()), baseline_rmse=a,
                cosine_rmse=b, cosine_rmse_improvement_ms=a-b,
                cosine_rmse_improvement_percent=100*(a-b)/a, cosine_beats_baseline=bool(b<a)))
        stability[grouping] = table
    summary = dict(model=model, original_n=len(rows), n=len(rows), change_in_n=0,
        additional_exclusions=0, sample_policy='Exact saved regression/representation sample; fail on unusable predictors',
        models=report, cosine_winning_folds=sum(r['cosine_beats_baseline'] for r in stability['fold']),
        cosine_winning_stories=sum(r['cosine_beats_baseline'] for r in stability['item']),
        folds_policy='Read saved fold IDs; never regenerate',
        scaling='Training fold means and population SDs only; outcome unscaled',
        bpe_policy='Raw counts, current and actual previous (item, zone-1) from full trajectory table',
        stability_policy='Subset RMSE of existing out-of-fold predictions; story rows are not leave-one-story-out fits')
    return summary, output, stability, audits


def run(root=Path('results')):
    root = Path(root)
    dest = root / 'representations/analysis/bpe_control'
    # Never replace earlier results, including an earlier robustness run.
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing output directory: {dest}')
    before = snapshot(root)
    products = {}
    for model in MODELS:
        joined = read_tsv(root / f'representations/analysis/{model}_joined.tsv')
        trajectory = read_tsv(root / f'representations/{model}_trajectories.tsv')
        residual = read_tsv(root / f'regression/{model}_heldout_residuals.tsv')
        rows = add_bpe_predictors(joined, trajectory, residual)
        # Verify all baseline predictor values against the established regression input.
        lookup = {key(r): r for r in residual}
        for r in rows:
            if not np.allclose([float(r[c]) for c in predictors(model)],
                               [float(lookup[key(r)][c]) for c in predictors(model)], rtol=0, atol=1e-10):
                raise ValueError('Baseline predictor disagreement')
        products[model] = evaluate(rows, model)
    identities = [[(key(r), int(r['fold']), float(r['observed_rt'])) for r in products[m][1]] for m in MODELS]
    if identities[0] != identities[1]:
        raise ValueError('Model prediction rows/folds/outcomes differ')
    dest.mkdir(parents=True)
    for model, (summary, rows, stability, audits) in products.items():
        summary['model_row_ids_and_folds_identical'] = True
        write_tsv(dest / f'{model}_predictions.tsv', rows)
        write_tsv(dest / f'{model}_fold_stability.tsv', stability['fold'])
        write_tsv(dest / f'{model}_story_stability.tsv', stability['item'])
        (dest / f'{model}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        (dest / f'{model}_fold_audit.json').write_text(json.dumps(audits, indent=2, allow_nan=False)+'\n')
    after = snapshot(root)
    if any(after.get(p) != digest for p,digest in before.items()):
        raise AssertionError('An existing output file changed')
    audit = dict(existing_files_checked=len(before), all_existing_files_unchanged=True,
                 sha256_before=before, exact_cross_model_row_alignment=True)
    (dest / 'preservation_validation.json').write_text(json.dumps(audit, indent=2)+'\n')
    return {m: result[0] for m,result in products.items()}
