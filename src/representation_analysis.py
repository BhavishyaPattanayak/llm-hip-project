"""Associational trajectory analyses and held-out incremental RT tests."""
import numpy as np
from scipy import stats
from src.data import key
from src.regression import predictors, cross_validate, descriptive, fit_ols, predict, metrics, make_folds
from src.representations import MEASURES, feature_names


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError('Undefined correlation p-value')
    order = np.argsort(p)
    corrected = np.minimum.accumulate((p[order]*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.clip(corrected, 0, 1)
    return result


def summary_names(measure):
    return [f'{third}_mean_{measure}' for third in ('early', 'middle', 'late')]


def join_residuals(features, residuals, n_blocks):
    lookup = {key(r): r for r in features}
    if len(lookup) != len(features) or len({key(r) for r in residuals}) != len(residuals):
        raise ValueError('Duplicate keys')
    joined = []
    for r in sorted(residuals, key=key):
        f = lookup.get(key(r))
        if f is None or f['word'] != r['word']:
            raise ValueError(f'Missing feature/word disagreement: {key(r)}')
        if not np.isfinite([float(f[c]) for c in feature_names(n_blocks)]).all():
            raise ValueError('Nonfinite features; no automatic exclusions permitted')
        row = dict(r, n_bpe=int(f['n_bpe']), **{c: float(f[c]) for c in feature_names(n_blocks)})
        if not np.isclose(float(r['residual']), float(r['observed_rt'])-float(r['predicted_rt']), atol=1e-10):
            raise ValueError('Invalid saved residual')
        joined.append(row)
    # Compare to established deterministic definition, but use the SAVED assignments.
    saved = np.array([int(r['fold']) for r in joined])
    if not np.array_equal(saved, make_folds(len(joined), seed=2026)):
        raise ValueError('Saved folds disagree with established baseline schedule')
    return joined


def correlate(x, y, method):
    result = (stats.pearsonr if method == 'pearson' else stats.spearmanr)(x, y)
    if not np.isfinite([result.statistic, result.pvalue]).all():
        raise ValueError('Undefined correlation (constant feature/outcome)')
    return float(result.statistic), float(result.pvalue)


def layer_correlations(rows, n_blocks):
    signed = np.array([float(r['residual']) for r in rows])
    output = []
    # FDR separately within model x measure x target x correlation method.
    for measure in MEASURES:
        for target, y in [('signed', signed), ('absolute', abs(signed))]:
            for method in ('pearson', 'spearman'):
                family = []
                for layer in range(1, n_blocks+1):
                    x = np.array([float(r[f'{measure}_layer_{layer}']) for r in rows])
                    effect, p = correlate(x, y, method)
                    family.append(dict(measure=measure, target=target, method=method,
                                       layer=layer, n=len(rows), correlation=effect, p_value=p))
                q = bh_fdr([r['p_value'] for r in family])
                output.extend(dict(r, q_value=float(v), fdr_05=bool(v < .05)) for r,v in zip(family,q))
    return output


def residual_regressions(rows):
    residual = np.array([float(r['residual']) for r in rows])
    output = []
    for measure in MEASURES:
        names = summary_names(measure)
        x = np.array([[float(r[c]) for c in names] for r in rows])
        for target, y in [('signed', residual), ('absolute', abs(residual))]:
            fit = fit_ols(x, y)
            r2 = metrics(y, predict(fit, x))['r2']
            output.extend(dict(c, measure=measure, target=target, n=len(y), r2=r2)
                          for c in descriptive(x, y, names))
    return output


def incremental_rt(rows, model):
    folds = np.array([int(r['fold']) for r in rows])
    y = np.array([float(r['observed_rt']) for r in rows])
    baseline = np.array([float(r['predicted_rt']) for r in rows])
    x = np.array([[float(r[c]) for c in predictors(model)] for r in rows])
    # Check reproducibility without replacing the saved baseline predictions/residuals.
    check, _ = cross_validate(x, y, folds)
    if not np.allclose(check, baseline, rtol=1e-9, atol=1e-7):
        raise ValueError('Baseline predictions do not reproduce under saved folds')
    base_metrics = metrics(y, baseline)
    report = {'baseline': base_metrics}
    output = [dict(item=r['item'], zone=r['zone'], word=r['word'], fold=int(folds[i]),
                   observed_rt=float(y[i]), baseline_predicted_rt=float(baseline[i])) for i,r in enumerate(rows)]
    audits = {}
    for measure in MEASURES:
        names = summary_names(measure)
        extra = np.array([[float(r[c]) for c in names] for r in rows])
        prediction, audit = cross_validate(np.column_stack([x, extra]), y, folds)
        m = metrics(y, prediction)
        report[measure] = dict(m, rmse_improvement=base_metrics['rmse']-m['rmse'],
            rmse_improvement_percent=100*(base_metrics['rmse']-m['rmse'])/base_metrics['rmse'],
            delta_r2=m['r2']-base_metrics['r2'])
        audits[measure] = dict(predictors=predictors(model)+names, folds=audit)
        for i, r in enumerate(output):
            r[f'{measure}_predicted_rt'] = float(prediction[i])
            r[f'{measure}_residual'] = float(y[i]-prediction[i])
    return report, output, audits


def confound_correlations(rows):
    output = []
    for measure in MEASURES:
        names = [f'{s}_{measure}' for s in ('total', 'mean', 'maximum', 'peak_layer',
                 'early_mean', 'middle_mean', 'late_mean', 'late_minus_early')]
        for name in names:
            x = [float(r[name]) for r in rows]
            for confound in ('surprisal', 'word_length', 'log_frequency', 'n_bpe'):
                y = [float(r[confound]) for r in rows]
                for method in ('pearson', 'spearman'):
                    # A peak-layer summary can legitimately be constant.
                    if np.ptp(x) == 0 or np.ptp(y) == 0:
                        effect, p, status = '', '', 'undefined_constant'
                    else:
                        effect, p = correlate(x, y, method)
                        status = 'ok'
                    output.append(dict(measure=measure, feature=name, confound=confound,
                                       method=method, correlation=effect, p_value=p, status=status, n=len(rows)))
    return output
