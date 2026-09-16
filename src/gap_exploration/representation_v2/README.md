# Representation exploration, phase 2

This phase writes only new files in the four `gap_exploration/representation_v2/` namespaces. The phase-entry manifest includes the original project **and** all previous exploration files. None may be overwritten. No commits are made.

## What the reference methods already capture

The original extraction mean-pools a region's BPE vectors at each depth, then calculates distances. States are the embedding-plus-position input and each block's raw residual output. The final state is captured **before** GPT-2's final LayerNorm, preserving comparability across blocks. States are at the word's own BPE positions, including the word itself; they are not the previous-token states used to score surprisal. Assigned windows, stride 256, and split-region ownership exactly follow original surprisal extraction.

| Existing method | Information retained | Information lost |
|---|---|---|
| Adjacent cosine | `1 - cos(h[l], h[l-1])`, one scalar per transition | Vector identity, movement direction, norm changes |
| Adjacent relative L2 | `norm(h[l]-h[l-1]) / norm(h[l-1])` | Direction and endpoints; conflates radial and angular movement |
| Early/middle/late means | Mean adjacent distances over equal thirds of block transitions | Within-third variation and vector coordinates |
| All-layer ridge | All 12 or 6 adjacent distances, with train-fitted scaling and inner-tuned ridge | Still no hidden vectors or directional geometry |
| Nonlinear trajectory model | Additive cubic splines of the three distance means: four training-quantile knots, degree 3, no bias, nested ridge | No cross-layer interaction surface, no vector coordinates |
| Previous exploration's trajectory profile | Total variation of adjacent scalar distances / sum; distance-weighted layer depth; max / sum, separately for cosine and L2 | These summarize the **distance profile**, not curvature of the vector path |

Previous nonlinear uncertainty features were transformations of surprisal and entropy, not hidden-state readouts. They are not retested here.

## New candidates and motivations

Exact definitions and a fixed selection rule were saved in `results/gap_exploration/representation_v2/plan.json` before evaluation.

- **Displacement:** compares final state with input, one-third-depth and two-thirds-depth states. Captures net revision, which can differ from cumulative adjacent movement.
- **Curvature:** compares successive displacement **vectors**. Captures redirection even when step magnitudes are unchanged. Zero-length transitions are explicitly flagged.
- **Norm dynamics:** uses log raw-state norms and their evolution. Residual-stream magnitude can reflect activation changes; post-final-normalized norms would obscure these changes and are not used.
- **Contextualization:** compares final full-context representation with a representation of the same region with no preceding BPEs.
- **Context dependence:** compares full-context representation with 16- and 64-BPE prior contexts. These are representation contrasts, not surprisal differences.
- **Relations to preceding words:** same-window similarities to the previous word, two-back word and mean of up to three predecessors at three depths; also final-minus-early changes. Captures contextual differentiation/integration, without claiming retrieval effort.
- **Layerwise readout:** separate early/middle/late PCAs and their concatenation. Sixteen components per layer, fixed before evaluation. Captures distributed coordinates rather than handcrafted distances; potential lexical/syntactic/position encoding is an important alternative explanation.
- **Typicality:** training-centroid RMS distance, shrinkage-whitened retained-PC radius, and discarded-component RMS radius at each selected layer. Tests unusualness relative to the training distribution, without nearest-neighbor hyperparameter search.

Early/middle/final blocks are 4/8/12 for GPT-2 and 2/4/6 for DistilGPT-2. These are **selected individual layers**, not averages of vector states across thirds.

### Alignment and contextual contrasts

The same full-text BPE IDs are reused in every context condition; leading-space tokens are not retokenized as isolated words. Truncated runs retain the absolute **local full-window position IDs**, preventing positional reindexing from being mistaken for context effects. Within-word causal prefixes remain intact. If the full available prefix is shorter than a cutoff, the full state is reused. A region split across scoring windows retains each piece's original ownership before pooling in both conditions. No BOS is inserted. These artificial masked-prefix contrasts quantify a controlled perturbation, not a claim about naturally encountered standalone-word distributions.

Previous-word comparisons use complete preceding regions in the **same forward window** as each current owned piece group. They do not compare a current representation with a predecessor extracted under a different sliding-window origin. Context boundaries never cross stories/trials. Existing first/second-region exclusions ensure two predecessors for eligible observations.

### Baselines, selection and leakage

The nonrepresentation baseline includes current/previous length, log frequency, BPE count, surprisal and entropy; plus two-back length, frequency, BPE, surprisal and entropy. Missing lag-2 frequency/model values use the inherited zero-plus-availability flags. The history controls are held fixed, not investigated as new candidate families.

The original trajectory architecture is the historically strongest frozen entropy-controlled method, refitted with these stronger controls: GPT-2 all-layer relative L2; DistilGPT-2 Natural Stories cosine splines; DistilGPT-2 GECO all-layer relative L2. Predictions must reproduce the previous lexical-history-plus-spillover sensitivity reference within `1e-6`.

Outer samples/folds are loaded verbatim. Every inner and outer training subset gets its own PCA, centroid, typicality reference, scaler and (where applicable) spline knots. PCA uses centered raw vectors, randomized SVD with three power iterations and fixed seed 2026; it does not use outcomes. Ridge uses the existing 25-alpha grid. All regression uses CPU float64; MPS handles offline extraction.

A candidate qualifies within an outer training set only if it improves pooled inner validation SSE and wins at least four of five inner folds. Best-family and combined predictions use those inner decisions. If the all-layer PCA concatenation qualifies, individual-layer PCAs are omitted from the combination to avoid duplicate columns. No family is dropped or added after inspecting outer scores. The full outer table is exploratory screening, not a basis for selecting the reported best/combined prediction rule.

## Robustness and limits

Primary observations remain 10,216 Natural Stories regions and 30,155 GECO gaze regions per model. GECO total is deferred as a secondary target in this phase, as permitted in the request. All serious variants, including negligible gains and losses, appear in the new registry. Training-fold distributions/correlations, fold and story/trial stability, and concentration of SSE gains among the largest 1% baseline errors are saved.

A **post-hoc** sensitivity, planned after the first large Natural Stories PCA result, adds current/previous punctuation, capitalization and digit flags, plus region/BPE position and original model position. It retains the primary inner-selected combinations and refits all preprocessing/tuning within folds; no new representation variants are searched. Primary results are unchanged. The sensitivity is a confound diagnostic, not a preregistered primary result.

All features use the current word. Gains may encode lexical identity, syntax, punctuation or position; they are not proof of anticipation, causal information propagation, or processing effort. Random saved word folds allow familiar word types and contexts across training/test; these results do not establish generalization to new texts. Story/trial diagnostics are descriptive. PCA retains a bounded subspace, so poor results do not rule out all hidden-state information. No conclusion about irreducible human variability is justified.

## Commands

Use `PYTHONDONTWRITEBYTECODE=1` and `python -B` to protect frozen bytecode. Every output command refuses to overwrite existing results; these commands document completed runs, not an instruction to regenerate frozen outputs.

```sh
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python -B -m experiments.gap_exploration.representation_v2.extract --model gpt2 --smoke --device mps
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 .venv/bin/python -B -m experiments.gap_exploration.representation_v2.extract --model gpt2 --device mps
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.gap_exploration.representation_v2.run --dataset natural_stories --model gpt2
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m experiments.gap_exploration.representation_v2.sensitivity --dataset natural_stories --model gpt2
```

Repeat the model-specific commands for `distilgpt2`, and regression/sensitivity for `--dataset geco`. Extraction processes both datasets. Cached model revisions are pinned to original surprisal metadata. In this environment MPS requires executing outside the filesystem sandbox. No downloads or package installs are needed.
