"""Present all attempted families, never only the best observed family."""
import json
from pathlib import Path
import numpy as np
from src.data import read_tsv
from src.extension_io import new_json,new_table
from src.gap_exploration.plan import FAMILIES,DEFERRED


def main():
    root=Path('results/gap_exploration');rows=[];cases={}
    for path in sorted((root/'metadata').glob('*/summary.tsv')):
        case=json.loads((path.parent/'case.json').read_text());table=read_tsv(path)
        cases[path.parent.name]=(case,table)
        rows.extend(table)
    if len(cases)!=6:raise ValueError(f'Expected six complete cases, found {len(cases)}; do not silently omit a dataset/model')
    sensitivity_rows=[]
    for sensitivity in ('lag2_lexical_sensitivity','lag2_lexical_combined_sensitivity'):
        path=root/'metadata'/sensitivity/'results.tsv'
        if path.exists():
            for r in read_tsv(path):
                row={k:'' for k in rows[0]};row.update(r);row['variant']=sensitivity
                row['outcome']='improves' if float(r['improvement_percent'])>0 else 'worsens'
                sensitivity_rows.append(row)
    new_table(root/'summary.tsv',rows+sensitivity_rows)
    log=[]
    for r in rows+sensitivity_rows:
        variant=r['variant'];definition=FAMILIES.get(variant,{}).get('definition',
            'Fixed historical reference' if variant=='strong' else 'Nested training-selected family combination' if variant in ('best_new','combined_new','trajectory_combined') else 'Entropy/lexical/surprisal/BPE controls')
        if variant.startswith('lag2_lexical'):
            definition='Post-hoc older-word lexical/BPE confound check; unchanged sample/folds and primary inner selections. '+variant
        log.append(dict(r,definition=definition,exploratory=True,selection='post-hoc confound sensitivity' if variant.startswith('lag2_lexical') else 'fixed individual family' if variant in FAMILIES else 'inner-training-only selection' if variant in ('best_new','combined_new','trajectory_combined') else 'reference'))
    new_table(root/'exploration_log.tsv',log)
    new_json(root/'metadata/deferred_candidates.json',DEFERRED)
    figures=root/'figures';figures.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(figures/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    variants=['strong',*FAMILIES,'best_new','combined_new','trajectory_combined']
    for metric,name,ylabel in [('rmse','baseline_candidate_rmse','Held-out RMSE (ms)'),('delta_r2','incremental_r2','Δ held-out R² vs strong reference'),('improvement_percent','dataset_model_comparison','RMSE improvement (%) vs strong reference')]:
        fig,axes=plt.subplots(3,2,figsize=(13,11))
        for ax,(label,(case,table)) in zip(axes.flat,cases.items()):
            lookup={r['variant']:r for r in table}
            ax.bar(range(len(variants)),[float(lookup[v][metric]) for v in variants],color=['#929ca3']+['#26778e']*4+['#c28c48']*3)
            ax.set_xticks(range(len(variants)),[v.replace('_','\n') for v in variants],fontsize=7)
            ax.set_title(f"{case['dataset']} / {case['target']} / {case['model']}")
            ax.set_ylabel(ylabel);ax.axhline(0,color='gray',lw=.6)
        save(fig,name)
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    for ax,(label,(case,table)) in zip(axes.flat,cases.items()):
        stability=read_tsv(root/'metadata'/label/'stability.tsv')
        for variant in [*FAMILIES,'trajectory_combined']:
            subset=[r for r in stability if r['variant']==variant and r['grouping']=='fold']
            ax.plot([int(r['group']) for r in subset],[float(r['rmse_improvement_ms']) for r in subset],'-o',ms=2,label=variant)
        ax.axhline(0,color='gray',lw=.7);ax.set_title(label);ax.set(xlabel='Frozen fold',ylabel='RMSE improvement (ms)')
    axes[0,0].legend(fontsize=6,frameon=False);save(fig,'fold_stability')
    fig,axes=plt.subplots(3,2,figsize=(12,10))
    comparison=['core','strong','best_new','combined_new','trajectory_combined']
    for ax,(label,(case,table)) in zip(axes.flat,cases.items()):
        lookup={r['variant']:r for r in table}
        ax.bar(range(5),[float(lookup[v]['rmse']) for v in comparison],color=['#929ca3','#26778e','#c28c48','#c28c48','#ad5746'])
        ax.set_xticks(range(5),['Entropy\ncontrols','+ Existing\ntrajectory','+ Inner-best\nnew family','+ Inner-selected\nnew families','Trajectory +\nnew families'],fontsize=7)
        ax.set_title(label);ax.set_ylabel('Held-out RMSE (ms)')
    save(fig,'best_combined_comparison')
    # Training-only diagnostic distributions are shown for outer fold zero, never used as evidence of success.
    diagnostics=[]
    for label,(case,_) in cases.items():
        table=read_tsv(root/'metadata'/label/'training_diagnostics.tsv')
        for family in FAMILIES:
            subset=[r for r in table if r['fold']=='0' and r['family']==family]
            diagnostics.append(dict(case=label,family=family,max_control_correlation=max(float(r['max_abs_control_correlation']) for r in subset),
                max_redundancy=max(float(r['max_abs_within_family_correlation']) for r in subset)))
    new_table(root/'metadata/confound_overview.tsv',diagnostics)
    lines=['# Exploratory gap analysis','',
        'All comparisons use the frozen sample and saved folds. Individual definitions were fixed before new evaluation. Best/combined family selection uses only outer-training inner folds. No positive result was required.','',
        'Strong references retain the historically best entropy-controlled trajectory model, selected from frozen prior results. GECO total is secondary: no completed total baseline existed; it inherits the gaze sample, folds and architecture.','',
        '| Dataset / target / model | Reference RMSE | Best observed individual family (exploratory) | Individual gain % | Trajectory + inner-selected combined gain % | Combined fold wins |',
        '|---|---:|---|---:|---:|---:|']
    for label,(case,table) in cases.items():
        individual=[r for r in table if r['variant'] in FAMILIES];best=max(individual,key=lambda r:float(r['improvement_percent']))
        combined=next(r for r in table if r['variant']=='trajectory_combined')
        lines.append(f"| {case['dataset']} / {case['target']} / {case['model']} | {float(best['baseline_rmse']):.4f} | {best['variant']} | {float(best['improvement_percent']):.4f} | {float(combined['improvement_percent']):.4f} | {combined['fold_wins']}/10 |")
    lines+=['','## Older-word lexical confound checks (explicitly post-hoc)','',
        '| Dataset / target / model | Diagnostic | RMSE gain over lag-2 lexical-controlled reference (%) | Fold wins |',
        '|---|---|---:|---:|']
    for r in sensitivity_rows:
        lines.append(f"| {r['dataset']} / {r['target']} / {r['model']} | {r['variant']} | {float(r['improvement_percent']):.4f} | {r['fold_wins']}/10 |")
    lines+=['','The leading lag-2 family shares much of its gain with omitted older-word lexical/BPE properties. The combined diagnostic reuses the original inner-training-selected families; primary predictions are not replaced.','',
        '## Main interpretation','',
        'Longer model-prediction history helps most among the individual families, but its gain is partly a lexical-history proxy. Combining families yields about 4–5% further error reduction on Natural Stories and less than 1% on GECO before the older-word lexical sensitivity. This is a modest dataset-dependent improvement, not evidence that richer Transformer signals substantially close the gap across paradigms. Distribution-shape extraction alone adds little. Nonlinear uncertainty and trajectory-profile shape are largely alternative readouts of existing signals.','',
        'No tested family has to be hidden to obtain this conclusion. Individual pooled effects are positive in this run, but several outer folds and many contexts worsen, and several families add very little. All comparisons and diagnostic distributions remain available.','',
        '## Definitions and caveats','']
    for name,definition in FAMILIES.items():lines.append(f"- **{name}**: {definition['definition']} {definition['confound']}")
    lines+=['','The individually best observed family in this table is a descriptive comparison, not a newly validated winner. The best/combined prediction files instead use inner-selected families, which can differ by fold.','',
        'All negative comparisons remain in summary.tsv and exploration_log.tsv. Correlations/distributions are from outer-training rows only. Fold/story/trial stability is descriptive, not independent replication or unseen-context validation. The richest reference itself was selected historically; we did not weaken it or tune it against the new candidate results.','',
        'Current/previous lexical controls, BPE counts, surprisal, entropy, and historical trajectory features remain in strong-reference comparisons. Nonlinear uncertainty is a new readout of existing information, not an independently extracted signal. Large gain shares from the largest 1% reference errors indicate concentration; no observations were removed. GECO frequency coverage and scalar/whitespace stimulus caveats are inherited unchanged.','',
        'No inference about irreducible human variability, causality, or model architecture follows from these results. Five-percent incremental RMSE reduction was a descriptive benchmark, never a selection objective.','',
        'Extraction ran on the local MPS GPU. Float64 nested SVD regressions ran on CPU. No new models or downloads were used.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
