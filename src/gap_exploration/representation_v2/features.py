"""Outcome-independent features from raw residual streams."""
import numpy as np

EPS = 1e-12


def distances(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.stack([1-np.clip(np.sum(a*b, axis=-1)/np.maximum(na*nb, EPS), -1, 1),
                     np.linalg.norm(a-b, axis=-1)/np.maximum(nb, EPS)], axis=-1)


def geometry(h):
    """h has shape [words, embedding+blocks, hidden]. No post-final LayerNorm."""
    h = np.asarray(h, dtype=np.float64)
    blocks = h.shape[1]-1
    early, middle = blocks//3, 2*blocks//3
    displacement = distances(h[:, -1, None], h[:, [0, early, middle]]).reshape(len(h), -1)
    delta = np.diff(h, axis=1)
    norms = np.linalg.norm(h, axis=-1)
    movement = np.linalg.norm(delta, axis=-1)
    valid = (movement[:, :-1] > EPS) & (movement[:, 1:] > EPS)
    curvature = distances(delta[:, 1:], delta[:, :-1])[..., 0]
    curvature = np.where(valid, curvature, 0.)
    curves = np.column_stack([curvature.mean(1), curvature.max(1),
        *[p.mean(1) for p in np.array_split(curvature, 3, axis=1)], (~valid).mean(1)])
    log_norm = np.log(np.maximum(norms, EPS)); change = np.diff(log_norm, axis=1)
    dynamics = np.column_stack([log_norm[:, 0], log_norm[:, -1], log_norm[:, -1]-log_norm[:, 0],
        abs(change).max(1), *[p.mean(1) for p in np.array_split(log_norm[:, 1:], 3, axis=1)]])
    return dict(displacement=displacement, curvature=curves, norm_dynamics=dynamics)


DEFINITIONS = {
    'displacement': 'Final vs embedding, block L/3 and block 2L/3: cosine distance and L2 normalized by source norm (6).',
    'curvature': 'Cosine distance between successive displacement VECTORS: mean, max, thirds and zero-transition fraction (6).',
    'norm_dynamics': 'Raw residual-stream log norms: input, final, net change, max absolute step, three block-third means (7).',
    'contextualization': 'Final full-context vs zero preceding BPE context: cosine and relative L2; same token IDs and absolute positions (2).',
    'context_dependence': 'Final full-context vs last 16 and 64 preceding BPEs: cosine and relative L2; same IDs and positions (4).',
    'previous_relations': 'Same-window cosine to previous word, two-back word, mean of up to three predecessors at early/middle/final blocks, plus final-minus-early differences (12).',
    'layer_early': '16 training-only centered PCA scores from raw block L/3, then nested ridge.',
    'layer_middle': '16 training-only centered PCA scores from raw block 2L/3, then nested ridge.',
    'layer_late': '16 training-only centered PCA scores from raw final block, then nested ridge.',
    'layer_combination': 'Concatenation of the three 16-component layer PCAs (48), nested ridge.',
    'typicality': 'At each selected layer: RMS distance to training centroid, shrinkage-whitened PCA radius, and RMS PCA reconstruction residual (9). Training reference only; no neighbor search.',
}
