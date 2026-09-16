# Isolated gap exploration

All new code and outputs are confined to `src/gap_exploration/`,
`experiments/gap_exploration/`, `tests/gap_exploration/`, and
`results/gap_exploration/`. The established source, data, tests, documentation,
results and folds are read-only. Run Python with `-B` to preserve frozen bytecode.
The original-file manifest includes `.venv`, `.git`, existing bytecode and symlinks.

## Design

The pre-evaluation plan is in `results/gap_exploration/metadata/pre_evaluation_plan.json`.
Four fixed families are evaluated: pre-first-BPE distribution shape, lag-2
surprisal/entropy, nonlinear uncertainty readouts, and normalized trajectory-profile
shape. These are not cognitive effort, literal information flow, or causal measures.
Attention/head searches, arbitrary context lengths and uncalibrated intermediate
unembeddings were deferred and logged, rather than expanding the feature search.

The strong reference is the historically best saved **entropy-controlled,
trajectory-inclusive** configuration for each dataset/model. Its predictions are
numerically reproduced before results are accepted. The core model still includes
current/previous length, frequency, BPE count, surprisal and entropy. Natural Stories
uses its exact 10,216-row sample and saved folds. GECO gaze uses its exact 30,155-row
sample and saved folds. No completed total-RT regression existed, so the secondary
GECO total analysis explicitly inherits gaze rows, folds and architecture while
changing only the observed target. Every total RT is available on that sample.

All transformations requiring estimation, including historical spline knots, are
fitted within training folds. Ridge uses the original 25-alpha grid and five inner
folds. Within each outer-training set, a family qualifies for combination only if
it reduces pooled inner held-out error and wins at least four of five inner folds
against the strong reference. Best/combined choices never use outer outcomes.
The combined procedure can select different families in different outer folds.
Empty selection has an explicit reference fallback. All individual families remain
in the output regardless of performance. Historical reference choice is inherited
from prior work, not presented as a newly preregistered model choice.

The five-way comparison retains entropy controls, historical trajectory reference,
controls plus inner-selected best family, controls plus inner-selected combined
families, and trajectory reference plus those combined families. Primary candidate
gains are against the strongest reference, not against a weakened baseline.

## Important confounds and sensitivity

After observing lag-2 model predictions lead, a separately logged post-hoc sensitivity
adds lag-2 length, frequency and BPE counts. Missing frequency is encoded zero with
an explicit availability flag, preserving every original row. It compares adding
lag-2 model signals and, separately, the same primary inner-selected combined
families. It does not replace primary predictions or change their selection rule.
Results of these sensitivity analyses also appear in the exploration log.

Lag-2 unscored context is likewise encoded zero plus an availability flag. Boundary
influence is decomposed descriptively, without removing observations. A diagnostic
also records how much net squared-error gain comes from the largest 1% reference
errors; its fraction can exceed one when other observations lose. Distribution
shape is measured before the first BPE and thus can proxy first-piece predictability
rather than whole-region surprisal. All controls and the inherited limited GECO
frequency coverage remain in force. Word-level folds share contexts and readers.

## Execution

All required full feature extraction completed locally on MPS, with cached pinned
GPT-2/DistilGPT-2 revisions. MPS is inaccessible from the restricted tool sandbox,
but works in a normal local process. No new model or package download was used.
Float64 ridge/SVD fits run on CPU because MPS does not support float64.

For a fresh sandbox output tree, from the repository root:

```bash
python -B -m experiments.gap_exploration.extract --model gpt2 --dataset both --device mps --local-files-only
python -B -m experiments.gap_exploration.extract --model distilgpt2 --dataset both --device mps --local-files-only
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/hsp-gap-matplotlib OPENBLAS_NUM_THREADS=1 python -B -m experiments.gap_exploration.run
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -B -m experiments.gap_exploration.sensitivity
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 python -B -m experiments.gap_exploration.sensitivity --combined
MPLCONFIGDIR=/tmp/hsp-gap-matplotlib python -B -m experiments.gap_exploration.report
```

Outputs already exist in this workspace; extraction/evaluation refuses to replace
completed case files. GPU smoke extraction uses `--smoke-words 32` and separate
output paths. If reproducing on Colab, replace `--device mps --local-files-only`
with `--device cuda`; revisions remain pinned. No Colab extraction remains required.

Run the original and sandbox test suites together:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/hsp-gap-matplotlib OPENBLAS_NUM_THREADS=1 python -B -m unittest discover -s tests -v
python -B -m experiments.gap_exploration.verify
```

## Outputs

- `summary.tsv`, `summary.md`, `exploration_log.tsv`: complete primary and post-hoc comparisons.
- `predictions/`: identifiers, original folds, observed RT, predictions and signed residuals.
- `metadata/`: frozen manifest, per-case sample/reference descriptions, nested selection
  audits, training-only distributions/correlations, fold and story/trial stability,
  two explicitly post-hoc lexical sensitivity runs, preservation checks.
- `figures/`: RMSE, incremental R², fold stability, dataset/model and combined comparisons.
- `features/`: separately versioned smoke/full pre-first-BPE distribution descriptors.

No claim about irreducible human variability or architecture differences follows.
The overview highlights the observed best individual family only descriptively;
the best/combined prediction columns instead use proper inner-fold selection.
