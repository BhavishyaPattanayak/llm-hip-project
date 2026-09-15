"""Analyze complete trajectories against the immutable saved baseline residuals."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.data import read_tsv, write_tsv
from src.representations import BLOCKS, MEASURES, validate_features, sha256
from src.representation_analysis import (join_residuals, layer_correlations,
    residual_regressions, incremental_rt, confound_correlations, summary_names)


def figures(rows, layers, performance, model, n_blocks, dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    def save(fig, name):
        fig.tight_layout()
        for extension in ('png', 'pdf'):
            fig.savefig(dest / f'{model}_{name}.{extension}', dpi=220)
        plt.close(fig)
    for target in ('signed', 'absolute'):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, measure in zip(axes, MEASURES):
            for method, color in [('pearson', '#26778e'), ('spearman', '#bb5837')]:
                subset = [r for r in layers if r['target']==target and r['measure']==measure and r['method']==method]
                ax.plot([r['layer'] for r in subset], [r['correlation'] for r in subset], '-o', ms=3, color=color, label=method)
                significant = [r for r in subset if r['fdr_05']]
                ax.scatter([r['layer'] for r in significant], [r['correlation'] for r in significant], marker='*', s=65, color=color)
            ax.axhline(0, color='gray', lw=.8)
            ax.set(xlabel='Destination block (1-based)', title=measure, xticks=range(1,n_blocks+1))
        axes[0].set_ylabel(f'Correlation with {target} residual')
        axes[1].legend(frameon=False, fontsize=9)
        fig.suptitle(f'{model}: stars denote within-family FDR < .05', fontsize=11)
        save(fig, f'{target}_layer_profile')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, measure in zip(axes, MEASURES):
        values = [[float(r[c]) for r in rows] for c in summary_names(measure)]
        ax.boxplot(values, tick_labels=['Early', 'Middle', 'Late'], showfliers=True,
                   flierprops=dict(marker='.', markersize=2, alpha=.15))
        ax.set(title=measure, ylabel='Mean adjacent-layer change')
    fig.suptitle(model)
    save(fig, 'third_distributions')
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = ['Baseline', '+ Cosine', '+ Rel. L2']
    for ax, metric in zip(axes, ['rmse', 'r2']):
        bars = ax.bar(labels, [performance[k][metric] for k in ('baseline', *MEASURES)],
                      color=['#929ca3', '#26778e', '#bb5837'])
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
        ax.set_ylabel('Held-out RMSE (ms)' if metric=='rmse' else 'Held-out R²')
        ax.margins(y=.2)
    fig.suptitle(model)
    save(fig, 'incremental_performance')
    absolute = abs(np.array([float(r['residual']) for r in rows]))
    lo, hi = np.quantile(absolute, [.25, .75])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, measure in zip(axes, MEASURES):
        values = np.array([[float(r[f'{measure}_layer_{i}']) for i in range(1,n_blocks+1)] for r in rows])
        for mask, label, color in [(absolute<=lo, 'Bottom quartile |residual|', '#26778e'),
                                   (absolute>=hi, 'Top quartile |residual|', '#bb5837')]:
            ax.plot(range(1,n_blocks+1), values[mask].mean(axis=0), '-o', ms=3, label=f'{label} (n={mask.sum()})', color=color)
        ax.set(xlabel='Destination block (1-based)', ylabel='Mean adjacent-layer change', title=measure)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f'{model}: descriptive groups only')
    save(fig, 'residual_group_trajectories')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(BLOCKS), help='Default: both models')
    args = p.parse_args()
    root = Path('results/representations')
    models = [args.model] if args.model else list(BLOCKS)
    # Fail before producing partial analyses if full extraction has not been done.
    for model in models:
        if not (root / f'{model}_trajectories.tsv').exists():
            raise FileNotFoundError(f'Full {model} extraction missing. Run experiments.extract_representations; smoke files are not analysis data.')
    for model in models:
        path = root / f'{model}_trajectories.tsv'
        info = json.loads(path.with_suffix('.json').read_text())
        source = Path(f'results/{model}_surprisal.tsv')
        if info['smoke_only'] or info['rows'] != 10256 or info['surprisal_sha256'] != sha256(source):
            raise ValueError('Extraction metadata/source mismatch')
        features = read_tsv(path)
        validate_features(features, read_tsv(source), BLOCKS[model])
        residual_path = Path(f'results/regression/{model}_heldout_residuals.tsv')
        rows = join_residuals(features, read_tsv(residual_path), BLOCKS[model])
        if len(rows) != 10216:
            raise ValueError('Unexpected baseline sample size')
        layers = layer_correlations(rows, BLOCKS[model])
        coefficients = residual_regressions(rows)
        performance, predictions, audits = incremental_rt(rows, model)
        confounds = confound_correlations(rows)
        output = root / 'analysis'
        output.mkdir(exist_ok=True)
        for name, table in [('joined', rows), ('layer_correlations', layers),
                            ('residual_coefficients', coefficients), ('incremental_predictions', predictions),
                            ('confound_correlations', confounds)]:
            write_tsv(output / f'{model}_{name}.tsv', table)
        summary = dict(model=model, n=len(rows), feature_rows=len(features),
            baseline_excluded_rows=len(features)-len(rows), additional_exclusions=0,
            incremental_rt=performance, fdr_family='model x measure x target x method, across blocks',
            residual_sha256=sha256(residual_path), trajectory_sha256=sha256(path),
            folds='Read exactly from saved baseline residual table; no refolding',
            significant_layer_tests=[r for r in layers if r['fdr_05']])
        (output / f'{model}_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        (output / f'{model}_fold_audit.json').write_text(json.dumps(audits, indent=2)+'\n')
        dest = Path('results/figures/representations')
        dest.mkdir(parents=True, exist_ok=True)
        figures(rows, layers, performance, model, BLOCKS[model], dest)
        print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
