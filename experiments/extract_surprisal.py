"""Run from the repository root: python -m experiments.extract_surprisal."""
import argparse
import json
from pathlib import Path
from src.data import read_regions, validate_surprisals, write_tsv
from src.surprisal import MODELS, load_model, score_story


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--stride', type=int, default=256)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    tokenizer, model, device = load_model(args.model, args.device, args.local_files_only)
    regions = read_regions()
    rows = []
    for item in range(1, 11):
        story = [r for r in regions if r['item'] == item]
        rows.extend(score_story(story, tokenizer, model, device, args.stride))
        print(f'Scored story {item}/10', flush=True)
    validate_surprisals(rows, regions)
    out = Path('results') / f'{args.model}_surprisal.tsv'
    write_tsv(out, rows)
    metadata = dict(model=args.model, device=device, stride=args.stride,
        context_window=model.config.max_position_embeddings, rows=len(rows),
        multi_bpe_words=sum(r['n_bpe'] > 1 for r in rows),
        first_word_policy='NaN for entire first reading region of each story; no BOS',
        units='bits', model_revision=getattr(model.config, '_commit_hash', None))
    out.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
