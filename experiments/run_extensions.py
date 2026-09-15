import argparse
import json
from src.extension_analysis import run_analysis


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['natural_stories','geco'],required=True)
    p.add_argument('--stage',choices=['trajectory','entropy'],default='trajectory')
    p.add_argument('--model',choices=['gpt2','distilgpt2'])
    p.add_argument('--dv',choices=['gaze','total'],default='gaze')
    args=p.parse_args()
    for model in [args.model] if args.model else ['gpt2','distilgpt2']:
        report=run_analysis(args.dataset,model,args.stage,args.dv)
        print(json.dumps({k:v for k,v in report.items() if k not in ('alpha_grid',)},indent=2))


if __name__=='__main__': main()
