"""Build lexical/RT preparation, or require both models for the analysis table."""
import argparse
import json
from pathlib import Path
from src.data import build_analysis, write_tsv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--human-only', action='store_true', help='Prepare RTs and lexical controls before model extraction')
    args = parser.parse_args()
    paths = {} if args.human_only else {
        m: Path('results') / f'{m}_surprisal.tsv' for m in ('gpt2', 'distilgpt2')}
    rows, report = build_analysis(model_paths=paths)
    name = 'human_predictors' if args.human_only else 'analysis'
    write_tsv(Path('results') / f'{name}.tsv', rows)
    Path(f'results/{name}_validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
