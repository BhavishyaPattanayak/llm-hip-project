"""All specified comparisons are shown, including null/negative results."""
import numpy as np
from src.data import read_tsv


def plot_run(out,rows,report,predictions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    directory=out/'figures'; directory.mkdir()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(directory/f'{name}.{ext}',dpi=200)
        plt.close(fig)
    comparisons=report['comparisons']; names=list(comparisons)
    for metric in ('rmse_improvement_percent','delta_r2'):
        fig,ax=plt.subplots(figsize=(10,max(4,len(names)*.3)))
        ax.barh(names,[comparisons[n][metric] for n in names],color='#26778e')
        ax.axvline(0,color='gray',lw=1); ax.set_xlabel(metric.replace('_',' ')); ax.invert_yaxis()
        save(fig,metric)
    selected=[n for n in names if 'all_layer_cosine' in n or 'spline_cosine' in n or n=='existing_linear_cosine']
    fig,ax=plt.subplots(figsize=(9,5))
    for name in selected:
        table=read_tsv(out/f'{name}_fold.tsv')
        ax.plot([int(r['group']) for r in table],[float(r['rmse_improvement_ms']) for r in table],'-o',ms=3,label=name)
    ax.axhline(0,color='gray',lw=1); ax.set(xlabel='Saved outer fold',ylabel='RMSE improvement (ms)')
    ax.legend(fontsize=7,frameon=False); save(fig,'fold_stability')
    fig,ax=plt.subplots(figsize=(9,5))
    for name in selected:
        table=read_tsv(out/f'{name}_context.tsv')
        if report['dataset']=='natural_stories':
            ax.plot([int(r['group']) for r in table],[float(r['rmse_improvement_ms']) for r in table],'-o',ms=3,label=name)
        else:
            ax.hist([float(r['rmse_improvement_ms']) for r in table],bins=40,histtype='step',label=name)
    ax.axhline(0,color='gray',lw=1) if report['dataset']=='natural_stories' else ax.axvline(0,color='gray',lw=1)
    ax.set_xlabel('Story (descriptive stability)' if report['dataset']=='natural_stories' else 'Trial RMSE improvement (ms; descriptive)')
    ax.legend(fontsize=7,frameon=False); save(fig,'context_stability')
    # Descriptive raw trajectory–RT association, not causal or independent-word inference.
    y=np.array([float(r['observed_rt']) for r in rows]); layers=12 if report['model']=='gpt2' else 6
    fig,ax=plt.subplots(figsize=(6,4))
    for measure in ('cosine','relative_l2'):
        values=[pearsonr([float(r[f'{measure}_layer_{i}']) for r in rows],y).statistic for i in range(1,layers+1)]
        ax.plot(range(1,layers+1),values,'-o',ms=3,label=measure)
    ax.axhline(0,color='gray',lw=1); ax.set(xlabel='Destination block',ylabel='Pearson r with mean RT (descriptive)')
    ax.legend(frameon=False); save(fig,'layer_rt_relationship')
