"""Extract full-corpus trajectories, or a clearly separated local smoke subset."""
import argparse
import json
from pathlib import Path
from src.data import read_regions, read_tsv, write_tsv, key
from src.surprisal import load_model
from src.representations import BLOCKS, extract_story, validate_features, metadata, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(BLOCKS), required=True)
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'mps', 'cuda'])
    p.add_argument('--local-files-only', action='store_true')
    p.add_argument('--smoke-words', type=int, help='Only first N words of story 1; output goes to smoke/; N must fit first window')
    args = p.parse_args()
    root = Path('results')
    source = root / f'{args.model}_surprisal.tsv'
    baseline = json.loads(source.with_suffix('.json').read_text())
    tokenizer, model, device = load_model(args.model, args.device, args.local_files_only)
    if model.config.n_layer != BLOCKS[args.model]:
        raise ValueError('Unexpected block count')
    if getattr(model.config, '_commit_hash', None) != baseline['model_revision']:
        raise ValueError('Model revision differs from saved surprisal revision')
    if model.config.max_position_embeddings != baseline['context_window']:
        raise ValueError('Context-window mismatch')
    regions = read_regions()
    if args.smoke_words is not None:
        if not 2 <= args.smoke_words <= 100:
            raise ValueError('Smoke subset must contain 2–100 words')
        regions = [r for r in regions if r['item'] == 1][:args.smoke_words]
    output = []
    for item in sorted({r['item'] for r in regions}):
        output.extend(extract_story([r for r in regions if r['item'] == item], tokenizer,
                                    model, device, baseline['stride']))
        print(f'{args.model}: completed story {item}', flush=True)
    keys = {key(r) for r in regions}
    ref = [r for r in read_tsv(source) if key(r) in keys]
    validate_features(output, ref, model.config.n_layer)
    dest = root / 'representations'
    if args.smoke_words is not None:
        dest /= 'smoke'
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f'{args.model}_trajectories.tsv'
    write_tsv(path, output)
    info = metadata(args.model, model, baseline['stride'], baseline, sha256(source), len(output))
    info.update(device=device, smoke_only=args.smoke_words is not None,
                multi_bpe_words=sum(r['n_bpe'] > 1 for r in output))
    path.with_suffix('.json').write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
