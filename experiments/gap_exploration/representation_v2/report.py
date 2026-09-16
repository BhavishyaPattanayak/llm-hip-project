"""Consolidate all experiments, including failures, without replacing artifacts."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import read_tsv
from src.extension_io import new_table,new_json,require_new
from src.gap_exploration.data import load_case
from src.gap_exploration.representation_v2.data import ROOT
from src.gap_exploration.representation_v2.features import DEFINITIONS


def main():
    labels=[f'{ds}_{m}' for ds in ('natural_stories','geco') for m in ('gpt2','distilgpt2')]
    primary=[];sensitivity=[];folds=[];registry=[];subgroups=[];correlations=[];influence=[]
    for label in labels:
        case=ROOT/'cases'/label;rows=read_tsv(case/'summary.tsv');primary.extend(rows)
        sens=read_tsv(ROOT/'sensitivity'/label/'summary.tsv');sensitivity.extend(sens)
        info=json.loads((case/'case.json').read_text());dataset,model=info['dataset'],info['model']
        for r in read_tsv(case/'stability.tsv'):
            if r['kind']=='fold':folds.append(dict(dataset=dataset,model=model,**r))
        for r in rows:
            registry.append(dict(phase='primary',dataset=dataset,model=model,target=info['target'],family=r['variant'],
                definition=r['definition'],variant='fixed definition in plan.json',n=r['n'],baseline='lexical+BPE+surprisal+entropy+lag2 history',
                rmse=r['rmse'],r2=r['r2'],improvement_ms=r['improvement_ms'],improvement_percent=r['improvement_percent'],delta_r2=r['delta_r2'],
                improvement_over_original_percent=r['improvement_over_original_percent'],fold_wins=r['fold_wins'],
                retained=r['retained'],inner_qualified_folds=r['inner_qualified_folds'],outcome=r['outcome']))
        for r in sens:
            registry.append(dict(phase='post_hoc_punctuation_position_sensitivity',dataset=dataset,model=model,target=info['target'],family=r['variant'],
                definition='Added nuisance controls per sensitivity_plan.json; original primary inner-selected combination unchanged',
                variant='one fixed sensitivity; no new family selection',n=r['n'],baseline='primary controls + current/previous punctuation/case/digits + word/BPE/model position',
                rmse=r['rmse'],r2=r['r2'],improvement_ms=r['improvement_ms'],improvement_percent=r['improvement_percent'],delta_r2=r['delta_r2'],
                improvement_over_original_percent=r['improvement_over_original_percent'],fold_wins=r['fold_wins'],retained='diagnostic only',inner_qualified_folds='',
                outcome='improves' if float(r['improvement_ms'])>0 else 'worsens' if float(r['improvement_ms'])<0 else 'reference/equivalent'))
        diag=read_tsv(case/'training_diagnostics.tsv')
        for family in DEFINITIONS:
            values=[float(r['max_abs_control_correlation']) for r in diag if r['family']==family]
            correlations.append(dict(dataset=dataset,model=model,family=family,max_abs_training_control_correlation=max(values)))
        influence.extend([dict(phase='primary',dataset=dataset,model=model,**r) for r in read_tsv(case/'influence.tsv')])
        predictions=read_tsv(case/'predictions.tsv');y=np.array([float(r['observed_rt']) for r in predictions])
        sp=read_tsv(ROOT/'sensitivity'/label/'predictions.tsv')
        reference_error=(y-np.array([float(r['baseline_prediction']) for r in sp]))**2
        top=np.argsort(reference_error)[-max(1,int(np.ceil(.01*len(y)))):]
        for name in ('baseline','original_trajectory','layer_combination','combined_new'):
            gain=reference_error-(y-np.array([float(r[name+'_prediction']) for r in sp]))**2;total=gain.sum()
            influence.append(dict(phase='post_hoc_sensitivity',dataset=dataset,model=model,variant=name,
                net_sse_gain=float(total),largest_one_percent_error_gain_fraction=float(gain[top].sum()/total) if total else 0.))
        source=load_case(dataset,model,info['target'])['rows'];f=np.array([int(r['fold']) for r in source])
        masks={'single_BPE':np.array([int(float(r['n_bpe']))==1 for r in source]),
               'punctuation':np.array([any(c in r['word'] for c in ',.!?;:') for r in source])}
        for name,column in [('length','word_length'),('frequency','log_frequency'),('surprisal',f'{model}_surprisal')]:
            values=np.array([float(r[column]) for r in source]);mask=np.zeros(len(source),bool)
            for fold in range(10):mask[f==fold]=values[f==fold]>np.median(values[f!=fold])
            masks['above_train_median_'+name]=mask
        for group,mask in masks.items():
            for level in (True,False):
                selected=mask==level
                for name in ('layer_combination','combined_new'):
                    b=np.array([float(r['baseline_prediction']) for r in predictions]);p=np.array([float(r[name+'_prediction']) for r in predictions])
                    br=np.sqrt(np.mean((y[selected]-b[selected])**2));pr=np.sqrt(np.mean((y[selected]-p[selected])**2))
                    subgroups.append(dict(dataset=dataset,model=model,variant=name,group=group,level=level,n=int(selected.sum()),
                        baseline_rmse=float(br),rmse=float(pr),improvement_ms=float(br-pr),improvement_percent=float(100*(br-pr)/br)))
    new_table(ROOT/'summary.tsv',primary);new_table(ROOT/'sensitivity_summary.tsv',sensitivity)
    new_table(ROOT/'exploration_log.tsv',registry);new_table(ROOT/'fold_results.tsv',folds)
    new_table(ROOT/'subgroup_diagnostics.tsv',subgroups);new_table(ROOT/'control_correlations.tsv',correlations);new_table(ROOT/'influence.tsv',influence)
    comparisons=[r for r in primary if r['variant'] in ('baseline','original_trajectory','best_new','combined_new')]
    new_table(ROOT/'abcd_comparison.tsv',comparisons)
    md=['# Better uses of hidden representations: exploratory results','',
        'All existing project and earlier gap-exploration artifacts are frozen. These are new analyses using exactly the saved samples and 10 outer folds. No observations were dropped (Natural Stories N=10,216; GECO gaze N=30,155 per model). GECO total was deferred as secondary.', '',
        '## Main comparison','',
        'The strong nonrepresentation baseline includes current/previous lexical, BPE, surprisal and entropy controls **and** two-word-back lexical/BPE/surprisal/entropy controls. The reference trajectory is the historically strongest architecture, refitted with these controls. C and D use only inner-training validation for family selection.', '',
        '| Dataset / model | A: baseline RMSE | B: original trajectory | C: inner-best new | D: inner-combined | D gain vs A | D gain vs B | D wins vs B |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        rs={r['variant']:r for r in primary if f"{r['dataset']}_{r['model']}"==label};d=rs['combined_new']
        md.append(f"| {label} | {float(rs['baseline']['rmse']):.6f} | {float(rs['original_trajectory']['rmse']):.6f} | {float(rs['best_new']['rmse']):.6f} | {float(d['rmse']):.6f} | {float(d['improvement_percent']):.4f}% | {float(d['improvement_over_original_percent']):.4f}% | {d['fold_wins_over_original']}/10 |")
    md+=['','## Every individual variant','',
        'Percent held-out RMSE improvement over A. Negative values are worse. Layer variants use 16 training-only PCs each; the combination has 48. The table is exploratory screening; it does not select the reported C/D models.', '',
        '| Family | NS GPT-2 | NS DistilGPT-2 | GECO GPT-2 | GECO DistilGPT-2 |','|---|---:|---:|---:|---:|']
    for family in DEFINITIONS:
        values=[next(float(r['improvement_percent']) for r in primary if f"{r['dataset']}_{r['model']}"==label and r['variant']==family) for label in labels]
        md.append('| '+family+' | '+' | '.join(f'{v:.4f}%' for v in values)+' |')
    md+=['','## Post-hoc punctuation and position sensitivity','',
        'The large initial Natural Stories PCA result motivated one further **confound diagnostic**. Current/previous punctuation, capitalization and digit indicators, plus word/BPE/model-position controls were added. Primary feature definitions and inner-selected combinations were held unchanged; preprocessing and ridge tuning were refitted within training folds. These controls sharply reduce the Natural Stories headline gain. This is not a preregistered primary result.', '',
        '| Dataset / model | Stricter baseline RMSE | Original trajectory gain | Layer PCA gain | Combined RMSE | Combined gain | Combined gain vs original | Combined wins vs original |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label in labels:
        rs={r['variant']:r for r in sensitivity if f"{r['dataset']}_{r['model']}"==label};d=rs['combined_new']
        md.append(f"| {label} | {float(rs['baseline']['rmse']):.6f} | {float(rs['original_trajectory']['improvement_percent']):.4f}% | {float(rs['layer_combination']['improvement_percent']):.4f}% | {float(d['rmse']):.6f} | {float(d['improvement_percent']):.4f}% | {float(d['improvement_over_original_percent']):.4f}% | {d['fold_wins_over_original']}/10 |")
    md+=['','## Interpretation and limitations','',
        '- The layerwise PCA readout retains information discarded by adjacent-distance summaries. It is a more useful predictive representation approach in these saved-fold experiments, especially on Natural Stories.',
        '- Much of the apparently large Natural Stories gain is accounted for by basic punctuation/position controls. The sensitivity result, rather than the primary headline alone, should guide substantive interpretation.',
        '- GECO gains are small. Consistent fold-level signs do not imply a large reduction in the human-model mismatch or broad cross-paradigm replication of the Natural Stories effect size.',
        '- The no-context contextualization family slightly worsens GECO GPT-2 and is essentially neutral for GECO DistilGPT-2. Several handcrafted geometric families fail to outperform the original trajectory reference. All failures are retained. No outer-test winner was used to select C or D.',
        '- The largest 1% of baseline-error observations account for about 20–26% of the primary combined-model SSE gain. This is concentrated, but not a majority of the gain. The corresponding sensitivity concentration is separately reported in influence.tsv. No observations were removed.',
        '- **After punctuation/position controls, Natural Stories gains are much more concentrated: the largest 1% of stricter-baseline errors contribute 56.2% (GPT-2) and 59.6% (DistilGPT-2) of the net SSE improvement.** Thus the remaining gain fails the ideal of being broadly distributed across observations. GECO sensitivity shares are 14.1% and 20.9%. These are descriptive gain decompositions using unchanged predictions, not trimmed/refitted analyses.',
        '- All hidden features are at the current word, so they include lexical identity. They may reflect syntax, punctuation, token identity or position; they are not evidence of causal processing effort or literal information flow. Even the sensitivity cannot establish a context-specific mechanism.',
        '- Saved random word folds allow word types and story/trial contexts in both train and test. Story/trial tables describe stability, not unseen-text generalization. These exploratory comparisons share test sets; there is no fresh confirmatory dataset or formal multiple-comparison inference.',
        '- Current/previous/two-back lexical, BPE, surprisal and entropy controls remain present. Training-only correlations and subgroup diagnostics assess redundancy; they do not prove independence. Some gains can concentrate among high-error observations, as quantified in influence.tsv. No observations were trimmed.',
        '- PCA dimension, candidate definitions, seeds, selection thresholds and ridge grid were fixed before primary evaluation. PCA, centroid references, splines and standardization were fit separately inside every inner and outer training split. No full-dataset PCA or typicality reference was used.',
        '- These experiments do not establish irreducible human variability or rule out other representation methods. The implemented typicality method uses centroid/PCA distances, not an unbounded density/nearest-neighbor search.', '',
        '## Exact artifacts','',
        '- `summary.tsv`: all primary RMSE, R², absolute/percentage gains and comparisons against original trajectory.',
        '- `abcd_comparison.tsv`: requested A/B/C/D table with full-precision metrics.',
        '- `sensitivity_summary.tsv`: the explicitly post-hoc stronger nuisance-control comparison.',
        '- `exploration_log.tsv`: every serious primary and sensitivity comparison, definitions and retention status.',
        '- `fold_results.tsv`: all 10 outer-fold primary metrics; each case also has story/trial stability tables.',
        '- `cases/*/predictions.tsv`, `cases/*/audit.json`: aligned predictions, nested PCA hashes/variance, selected families, alpha grids and outer preprocessing.',
        '- `subgroup_diagnostics.tsv`, `control_correlations.tsv`, `influence.tsv`: confound and concentration diagnostics.',
        '- `features/*.json`: MPS device, pinned revisions, timing and reconstruction agreement with old trajectories.',
        '- `frozen_manifest.json`, `final_integrity.json`: preservation of original and earlier exploration artifacts.', '',
        'Feature definitions, exact reference-method audit and reproducible commands are documented in `src/gap_exploration/representation_v2/README.md`. All GPU extraction completed locally on MPS; float64 regressions ran on CPU. No downloads or commits were needed.']
    with (ROOT/'summary.md').open('x') as f:f.write('\n'.join(md)+'\n')
    figures=require_new(ROOT/'figures')
    variants=list(DEFINITIONS)
    values=np.array([[next(float(r['improvement_percent']) for r in primary if f"{r['dataset']}_{r['model']}"==label and r['variant']==v) for label in labels] for v in variants])
    fig,ax=plt.subplots(figsize=(10,7));im=ax.imshow(values,cmap='coolwarm',vmin=-max(abs(values.min()),values.max()),vmax=max(abs(values.min()),values.max()),aspect='auto')
    ax.set_xticks(range(4),['Natural Stories\nGPT-2','Natural Stories\nDistilGPT-2','GECO gaze\nGPT-2','GECO gaze\nDistilGPT-2']);ax.set_yticks(range(len(variants)),variants)
    for i in range(len(variants)):
        for j in range(4):ax.text(j,i,f'{values[i,j]:.2f}%',ha='center',va='center',color='black' if abs(values[i,j])<values.max()*.6 else 'white')
    ax.set_title('Primary incremental RMSE reduction beyond strong history controls');fig.colorbar(im,ax=ax,label='% RMSE improvement');fig.tight_layout()
    fig.savefig(figures/'families.png',dpi=180);fig.savefig(figures/'families.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,source,title in zip(axes,[primary,sensitivity],['Primary history controls','Additional punctuation / position controls (post-hoc)']):
        for i,(variant,label) in enumerate([('original_trajectory','Original trajectory'),('layer_combination','Layer PCA'),('combined_new','Combined representations')]):
            v=[next(float(r['improvement_percent']) for r in source if f"{r['dataset']}_{r['model']}"==case and r['variant']==variant) for case in labels]
            ax.bar(np.arange(4)+(i-1)*.25,v,width=.25,label=label)
        ax.set_xticks(range(4),['NS\nGPT-2','NS\nDistil','GECO\nGPT-2','GECO\nDistil']);ax.set_ylabel('RMSE reduction (%)');ax.set_title(title,fontsize=10);ax.axhline(0,color='black',lw=.5)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(figures/'primary_and_sensitivity.png',dpi=180);fig.savefig(figures/'primary_and_sensitivity.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for ax,label in zip(axes.flat,labels):
        for variant in ('original_trajectory','layer_combination','combined_new'):
            rs=[r for r in folds if f"{r['dataset']}_{r['model']}"==label and r['variant']==variant]
            ax.plot([int(r['group'])+1 for r in rs],[float(r['improvement_percent']) for r in rs],marker='o',label=variant)
        ax.axhline(0,color='black',lw=.5);ax.set_title(label,fontsize=10);ax.set_xlabel('Existing fold');ax.set_ylabel('RMSE reduction (%)')
    axes[0,0].legend(fontsize=7);fig.tight_layout();fig.savefig(figures/'fold_stability.png',dpi=180);fig.savefig(figures/'fold_stability.pdf');plt.close(fig)


if __name__=='__main__':main()
