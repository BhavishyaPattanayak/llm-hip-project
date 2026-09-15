"""Summarize all available datasets/stages without hiding losing comparisons."""
import argparse
import json
from pathlib import Path
from src.extension_io import require_new,new_table


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/extensions/overview')
    args=p.parse_args()
    rows=[]
    sources=list(Path('results/extensions/natural_stories').glob('*/*/summary.json'))+list(Path('results/geco/extensions').glob('*/*/summary.json'))
    for path in sorted(sources):
        report=json.loads(path.read_text())
        for name,r in report['comparisons'].items():
            rows.append(dict(dataset=report['dataset'],dv=report['dv'],model=report['model'],
                analysis=path.parent.parent.name,comparison=name,n=report['n'],baseline=r['baseline_name'],
                baseline_rmse=r['baseline']['rmse'],rmse=r['augmented']['rmse'],r2=r['augmented']['r2'],
                improvement_ms=r['rmse_improvement_ms'],improvement_percent=r['rmse_improvement_percent'],
                delta_r2=r['delta_r2'],fold_wins=r['fold_wins'],context_win_fraction=r['context_win_fraction']))
    if not rows: raise ValueError('No completed extension runs found')
    out=require_new(args.output);new_table(out/'all_comparisons.tsv',rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=[r for r in rows if r['comparison'] in ('existing_linear_cosine','bpe_baseline__linear_cosine',
        'bpe_baseline__all_layer_cosine','bpe_baseline__spline_cosine','entropy_baseline__all_layer_cosine')]
    # Existing linear baseline may appear as a reference in both NS stages; show it once.
    unique={ (r['dataset'],r['dv'],r['model'],r['comparison']):r for r in primary}
    primary=list(unique.values())
    friendly={'existing_linear_cosine':'Linear thirds','bpe_baseline__linear_cosine':'Linear thirds',
        'bpe_baseline__all_layer_cosine':'All-layer ridge','bpe_baseline__spline_cosine':'Spline thirds',
        'entropy_baseline__all_layer_cosine':'All-layer ridge + entropy control'}
    labels=[f"{'Natural Stories' if r['dataset']=='natural_stories' else 'GECO '+r['dv']} · {r['model']} · {friendly[r['comparison']]}" for r in primary]
    for metric in ('improvement_percent','delta_r2'):
        fig,ax=plt.subplots(figsize=(12,max(4,len(primary)*.3)))
        bars=ax.barh(labels,[r[metric] for r in primary],color='#26778e')
        ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
        ax.margins(x=.15)
        ax.axvline(0,color='gray',lw=1);ax.invert_yaxis();ax.set_xlabel(metric.replace('_',' '))
        fig.tight_layout()
        for ext in ('png','pdf'): fig.savefig(out/f'{metric}.{ext}',dpi=220)
        plt.close(fig)


if __name__=='__main__':main()
