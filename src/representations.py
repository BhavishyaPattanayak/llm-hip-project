"""Compact layerwise representational change, with no cognitive/causal interpretation."""
import hashlib
import numpy as np
import torch
from src.data import key
from src.surprisal import encode_regions, context_windows

BLOCKS = {'gpt2': 12, 'distilgpt2': 6}
MEASURES = ('cosine', 'relative_l2')
EPSILON = 1e-12


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def thirds(n_blocks):
    return {name: indices.tolist() for name, indices in
            zip(('early', 'middle', 'late'), np.array_split(np.arange(1, n_blocks+1), 3))}


def feature_names(n_blocks):
    return [f'{measure}_layer_{i}' for measure in MEASURES for i in range(1, n_blocks+1)] + [
        f'{name}_{measure}' for measure in MEASURES for name in
        ('total', 'mean', 'maximum', 'peak_layer', 'early_mean', 'middle_mean', 'late_mean', 'late_minus_early')]


def trajectory(pooled):
    """Input: [embedding + raw block outputs, hidden dimension], pooled per word."""
    h = np.asarray(pooled, dtype=np.float64)
    if h.ndim != 2 or len(h) < 4 or not np.isfinite(h).all():
        raise ValueError('Invalid pooled hidden states')
    norms = np.linalg.norm(h, axis=1)
    if np.any(norms <= EPSILON):
        raise ValueError('Cosine distance undefined for a near-zero representation')
    cosine = 1 - np.clip(np.sum(h[:-1] * h[1:], axis=1) / (norms[:-1]*norms[1:]), -1, 1)
    relative = np.linalg.norm(h[1:]-h[:-1], axis=1) / (norms[:-1]+EPSILON)
    result = {}
    for measure, values in zip(MEASURES, (cosine, relative)):
        result.update({f'{measure}_layer_{i}': float(v) for i, v in enumerate(values, 1)})
        result.update({f'total_{measure}': float(values.sum()), f'mean_{measure}': float(values.mean()),
                       f'maximum_{measure}': float(values.max()), f'peak_layer_{measure}': int(values.argmax()+1)})
        for name, indices in thirds(len(values)).items():
            result[f'{name}_mean_{measure}'] = float(values[np.array(indices)-1].mean())
        result[f'late_minus_early_{measure}'] = result[f'late_mean_{measure}']-result[f'early_mean_{measure}']
    return result


def pool_states(states, owners, n_words):
    """Mean across BPE pieces at every depth, before computing distances."""
    sums = np.zeros((n_words, states.shape[0], states.shape[2]), dtype=np.float64)
    counts = np.bincount(owners, minlength=n_words)
    if np.any(counts == 0):
        raise ValueError('Unrepresented word')
    np.add.at(sums, owners, states.transpose(1, 0, 2))
    return sums / counts[:, None, None]


def extract_story(rows, tokenizer, model, device, stride=256):
    ids, owners = encode_regions([r['word'] for r in rows], tokenizer)
    n_blocks = model.config.n_layer
    sums = np.zeros((len(rows), n_blocks+1, model.config.n_embd), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=int)
    visits = np.zeros(len(ids), dtype=int)
    captured = {}
    def before_final_norm(module, args):
        captured['raw_final'] = args[0]
    hook = model.transformer.ln_f.register_forward_pre_hook(before_final_norm)
    windows = list(context_windows(len(ids), model.config.max_position_embeddings, stride))
    if len(ids) == 1:
        windows = [(0, 1, 1)]
    try:
        with torch.inference_mode():
            for begin, end, target_start in windows:
                # Token zero has a valid representation, though no surprisal.
                first = 0 if begin == 0 and target_start == 1 else target_start
                captured.clear()
                outputs = model.transformer(input_ids=torch.tensor([ids[begin:end]], device=device),
                    output_hidden_states=True, use_cache=False, return_dict=True)
                hidden = outputs.hidden_states
                if len(hidden) != n_blocks+1 or 'raw_final' not in captured:
                    raise ValueError('Unexpected Hugging Face hidden-state structure')
                # Native last state is post-ln_f; compare raw residual-stream block outputs instead.
                hidden = (*hidden[:-1], captured['raw_final'])
                local = first-begin
                states = np.stack([h[0, local:].float().cpu().numpy() for h in hidden])
                owned = np.array(owners[first:end])
                np.add.at(sums, owned, states.transpose(1, 0, 2))
                np.add.at(counts, owned, 1)
                visits[first:end] += 1
                del outputs, hidden, states
    finally:
        hook.remove()
    if not np.all(visits == 1) or not np.array_equal(counts, np.bincount(owners, minlength=len(rows))):
        raise ValueError('Token ownership/coverage failure')
    return [dict(r, n_bpe=int(counts[i]), **trajectory(sums[i]/counts[i])) for i, r in enumerate(rows)]


def validate_features(rows, reference, n_blocks):
    ref = {key(r): r for r in reference}
    if len(rows) != len(ref) or len({key(r) for r in rows}) != len(rows):
        raise ValueError('Missing/duplicate representation rows')
    for r in rows:
        other = ref.get(key(r))
        if other is None or r['word'] != other['word'] or int(r['n_bpe']) != int(other['n_bpe']):
            raise ValueError(f'Representation text/BPE mismatch: {key(r)}')
        if not np.isfinite([float(r[c]) for c in feature_names(n_blocks)]).all():
            raise ValueError('Nonfinite trajectory')


def metadata(model_name, model, stride, baseline, source_hash, n_rows):
    return dict(model=model_name, model_revision=getattr(model.config, '_commit_hash', None),
        baseline_model_revision=baseline['model_revision'], n_blocks=model.config.n_layer,
        n_hidden_states=model.config.n_layer+1, thirds=thirds(model.config.n_layer),
        pooling='Arithmetic mean across all BPE states per word and depth before distances',
        state_policy='Embedding+position state then raw residual-stream block outputs; final state captured before ln_f',
        token_policy='State at own BPE position (includes that BPE); not the preceding position used to predict it',
        context_window=model.config.max_position_embeddings, stride=stride,
        context_policy='Exact surprisal windows and target ownership; token zero from initial window; local position reset; no BOS',
        split_word_policy='Pieces spanning target windows retain their respective surprisal-window contexts before pooling',
        epsilon=EPSILON, peak_layer_policy='1-based destination block; first maximum on ties',
        feature_definitions=dict(cosine='1-cosine(mean-pooled adjacent states)',
            relative_l2='norm(next-current)/(norm(current)+1e-12)', total='sum over blocks',
            mean='mean over blocks', maximum='maximum over blocks',
            late_minus_early='late third mean minus early third mean'),
        surprisal_sha256=source_hash, rows=n_rows)
