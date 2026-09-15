"""OLS psychometric baseline and out-of-fold mismatch in milliseconds."""
import math
from collections import Counter
import numpy as np
from scipy import stats
from src.data import key

CONTROLS = ['word_length', 'prev_word_length', 'log_frequency', 'prev_log_frequency']
MODELS = ('gpt2', 'distilgpt2')


def number(value):
    return float(value) if value != '' else float('nan')


def predictors(model):
    return [f'{model}_surprisal', f'prev_{model}_surprisal'] + CONTROLS


def prepare(rows, model):
    """Audit overlapping exclusion reasons; retain every eligible row unchanged."""
    eligible, excluded = [], []
    counts = Counter()
    for r in sorted(rows, key=key):
        reasons = []
        zone = key(r)[1]
        if zone == 1:
            reasons += ['story_initial_surprisal', 'story_boundary_previous_predictors']
        if zone == 2 and not math.isfinite(number(r[f'prev_{model}_surprisal'])):
            reasons.append('previous_story_initial_surprisal')
        if not math.isfinite(number(r['log_frequency'])):
            reasons.append('current_frequency_unavailable')
        if zone > 1 and not math.isfinite(number(r['prev_log_frequency'])):
            reasons.append('previous_frequency_unavailable')
        missing = [c for c in predictors(model) if not math.isfinite(number(r[c]))]
        explained = set()
        if zone == 1:
            explained.update([f'{model}_surprisal', f'prev_{model}_surprisal', 'prev_word_length', 'prev_log_frequency'])
        if zone == 2:
            explained.add(f'prev_{model}_surprisal')
        explained.update(['log_frequency', 'prev_log_frequency'])
        if any(c not in explained for c in missing):
            reasons.append('other_missing_or_nonfinite_predictor')
        if not math.isfinite(number(r['mean_RT'])) or number(r['mean_RT']) <= 0:
            reasons.append('invalid_observed_rt')
        if reasons:
            counts.update(reasons)
            excluded.append(dict(item=r['item'], zone=r['zone'], word=r['word'],
                                 reasons=';'.join(reasons), nonfinite_predictors=';'.join(missing)))
        else:
            eligible.append(r)
    return eligible, excluded, dict(counts)


def make_folds(n, seed=2026, n_folds=10):
    if n < n_folds:
        raise ValueError('Fewer observations than folds')
    folds = np.empty(n, dtype=int)
    for fold, indices in enumerate(np.array_split(np.random.default_rng(seed).permutation(n), n_folds)):
        folds[indices] = fold
    return folds


def fit_ols(x, y):
    """Estimate scaling on supplied training rows only; unweighted OLS with intercept."""
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    if np.any(scale == 0):
        raise ValueError('Constant training predictor')
    design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    beta, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank != design.shape[1]:
        raise ValueError('Rank-deficient regression')
    return dict(mean=mean, scale=scale, beta=beta, design=design)


def predict(fit, x):
    return np.column_stack([np.ones(len(x)), (x - fit['mean']) / fit['scale']]) @ fit['beta']


def cross_validate(x, y, folds):
    predictions = np.full(len(y), np.nan)
    visits = np.zeros(len(y), dtype=int)
    audit = []
    for fold in sorted(set(folds)):
        test = np.flatnonzero(folds == fold)
        train = np.flatnonzero(folds != fold)
        assert not np.intersect1d(train, test).size
        fit = fit_ols(x[train], y[train])
        predictions[test] = predict(fit, x[test])
        visits[test] += 1
        audit.append(dict(fold=int(fold), train_n=len(train), test_n=len(test),
                          training_mean=fit['mean'].tolist(), training_scale=fit['scale'].tolist()))
    assert np.all(visits == 1) and np.all(np.isfinite(predictions))
    return predictions, audit


def metrics(y, predicted):
    errors = y - predicted
    return dict(rmse=float(np.sqrt(np.mean(errors ** 2))),
                r2=float(1 - np.sum(errors ** 2) / np.sum((y - y.mean()) ** 2)))


def descriptive(x, y, names):
    fit = fit_ols(x, y)
    df = len(y) - len(fit['beta'])
    errors = y - predict(fit, x)
    covariance = np.linalg.inv(fit['design'].T @ fit['design']) * (errors @ errors / df)
    se = np.sqrt(np.diag(covariance))
    t = fit['beta'] / se
    p = 2 * stats.t.sf(np.abs(t), df)
    return [dict(predictor=name, coefficient=float(b), standard_error=float(s),
                 t_value=float(tv), p_value=float(pv))
            for name, b, s, tv, pv in zip(['intercept'] + names, fit['beta'], se, t, p)]


def analyze(rows, model, folds):
    names = predictors(model)
    x = np.array([[number(r[c]) for c in names] for r in rows])
    y = np.array([number(r['mean_RT']) for r in rows])
    full, full_audit = cross_validate(x, y, folds)
    controls, controls_audit = cross_validate(x[:, 2:], y, folds)
    residual = y - full
    output = [dict(r, observed_rt=float(y[i]), predicted_rt=float(full[i]),
                   residual=float(residual[i]), surprisal=number(r[f'{model}_surprisal']),
                   fold=int(folds[i]), controls_predicted_rt=float(controls[i]),
                   controls_residual=float(y[i] - controls[i])) for i, r in enumerate(rows)]
    assert len({key(r) for r in output}) == len(output)
    assert np.all(np.isfinite(residual))
    assert all(r['residual'] == r['observed_rt'] - r['predicted_rt'] for r in output)
    cm, fm = metrics(y, controls), metrics(y, full)
    summary = dict(n=len(rows), controls=cm, full=fm,
        rmse_improvement=cm['rmse']-fm['rmse'],
        rmse_improvement_percent=100*(cm['rmse']-fm['rmse'])/cm['rmse'],
        r2_change=fm['r2']-cm['r2'],
        residual=dict(mean=float(residual.mean()), sd=float(residual.std(ddof=1)),
            median=float(np.median(residual)), p05=float(np.quantile(residual, .05)),
            p95=float(np.quantile(residual, .95)), minimum=float(residual.min()),
            maximum=float(residual.max()), observed_predicted_correlation=float(np.corrcoef(y, full)[0,1])))
    return output, summary, descriptive(x, y, names), dict(predictors=names, full=full_audit, controls=controls_audit)
