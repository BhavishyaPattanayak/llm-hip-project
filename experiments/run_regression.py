"""Validate saved model data and run deterministic word-level 10-fold OLS."""
import json
from pathlib import Path
import numpy as np
from src.data import read_tsv, read_regions, validate_surprisals, build_analysis, key, write_tsv
from src.regression import MODELS, number, prepare, make_folds, analyze


def same(a, b):
    try:
        return bool(np.isclose(number(a), number(b), rtol=1e-10, atol=1e-10, equal_nan=True))
    except (ValueError, TypeError):
        return str(a) == str(b)


def validate_inputs(root):
    rows = sorted(read_tsv(root / 'analysis.tsv'), key=key)
    regions = read_regions()
    paths = {m: root / f'{m}_surprisal.tsv' for m in MODELS}
    for m, path in paths.items():
        validate_surprisals(read_tsv(path), regions)
        metadata = json.loads((root / f'{m}_surprisal.json').read_text())
        assert metadata['rows'] == 10256 and metadata['model'] == m
    # Recompute for validation only; never rewrite the supplied analysis/surprisal.
    expected, report = build_analysis(model_paths=paths)
    human = sorted(read_tsv(root / 'human_predictors.tsv'), key=key)
    assert len(rows) == len(human) == 10256
    assert len({key(r) for r in rows}) == 10256
    for actual, h, exp in zip(rows, human, expected):
        assert key(actual) == key(h) == key(exp)
        assert all(same(actual[c], exp[c]) for c in exp), f'Analysis mismatch {key(exp)}'
        assert all(same(h[c], exp[c]) for c in h), f'Human mismatch {key(exp)}'
    assert sum(int(r['n_observations']) for r in rows) == 848875
    for filename in ('analysis_validation.json', 'human_predictors_validation.json'):
        saved = json.loads((root / filename).read_text())
        assert saved['rows'] == 10256 and saved['participant_observations'] == 848875
    return rows, report


def figures(output, model, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.dpi': 140})
    observed = np.array([r['observed_rt'] for r in output])
    predicted = np.array([r['predicted_rt'] for r in output])
    residual = observed - predicted
    surprisal = np.array([r['surprisal'] for r in output])
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / f'{model}_{name}.png', dpi=220)
        fig.savefig(directory / f'{model}_{name}.pdf')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(predicted, observed, s=7, alpha=.18, color='#26778e', rasterized=True)
    limits = [min(predicted.min(), observed.min()), max(predicted.max(), observed.max())]
    ax.plot(limits, limits, '--', color='gray', lw=1)
    ax.set(xlabel='Held-out predicted RT (ms)', ylabel='Observed mean RT (ms)', title=model)
    save(fig, 'observed_predicted')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=65, color='#26778e', edgecolor='white', linewidth=.3)
    ax.axvline(0, color='#b64b32', lw=1)
    ax.set(xlabel='Held-out residual (ms)', ylabel='Words', title=model)
    save(fig, 'residual_distribution')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(surprisal, observed, s=7, alpha=.15, color='#26778e', rasterized=True)
    xx = np.array([surprisal.min(), surprisal.max()])
    ax.plot(xx, np.polyval(np.polyfit(surprisal, observed, 1), xx), color='#b64b32', label='Unadjusted linear trend')
    ax.set(xlabel='Current-word surprisal (bits)', ylabel='Observed mean RT (ms)', title=model)
    ax.legend(frameon=False)
    save(fig, 'surprisal_rt')


def main():
    root = Path('results')
    dest = root / 'regression'
    dest.mkdir(exist_ok=True)
    figure_dir = root / 'figures'
    figure_dir.mkdir(exist_ok=True)
    rows, validation = validate_inputs(root)
    prepared = {m: prepare(rows, m) for m in MODELS}
    keys = [[key(r) for r in prepared[m][0]] for m in MODELS]
    if keys[0] != keys[1]:
        raise ValueError('Model eligibility differs; inspect exclusions before comparing models')
    folds = make_folds(len(keys[0]), seed=2026)
    report = dict(seed=2026, folds=10, split='shuffled word-level, shared across models',
                  source_validation=validation, equivalent_model_observations=True,
                  validation=dict(unique_region_keys=True, exactly_one_prediction_per_eligible_word=True,
                      train_test_disjoint=True, scaling_fit_on_training_only=True,
                      finite_predictions_and_residuals=True, residual_identity_verified=True,
                      metrics_from_heldout_predictions=True), models={})
    for model in MODELS:
        eligible, exclusions, reasons = prepared[model]
        output, summary, coefficients, audit = analyze(eligible, model, folds)
        summary.update(excluded=len(exclusions), exclusion_reasons_overlapping=reasons)
        report['models'][model] = summary
        write_tsv(dest / f'{model}_heldout_residuals.tsv', output)
        write_tsv(dest / f'{model}_exclusions.tsv', exclusions)
        write_tsv(dest / f'{model}_coefficients.tsv', coefficients)
        extremes = sorted(output, key=lambda r: r['residual'])
        write_tsv(dest / f'{model}_extremes.tsv',
                  [dict(r, tail='negative') for r in extremes[:20]] +
                  [dict(r, tail='positive') for r in extremes[-20:][::-1]])
        (dest / f'{model}_fold_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        figures(output, model, figure_dir)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, metric, label in zip(axes, ['rmse', 'r2'], ['Held-out RMSE (ms)', 'Held-out R²']):
        for j, (variant, color) in enumerate([('controls', '#929ca3'), ('full', '#26778e')]):
            values = [report['models'][m][variant][metric] for m in MODELS]
            bars = ax.bar(np.arange(2) + (j-.5)*.32, values, width=.32, color=color,
                          label='Controls only' if variant == 'controls' else 'With surprisal')
            ax.bar_label(bars, fmt='%.3f', fontsize=9, padding=3)
        ax.set_xticks(range(2), MODELS)
        ax.set_ylabel(label)
        ax.margins(y=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .90))
    for extension in ('png', 'pdf'):
        fig.savefig(figure_dir / f'cv_performance.{extension}', dpi=220)
    plt.close(fig)
    (dest / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
