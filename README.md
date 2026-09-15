# Transformer surprisal and human reading times

MSc seminar project testing whether information propagation inside Transformers
explains variation in human reading times remaining after surprisal and basic
lexical controls. This is a small Wilcox-style analysis, not an exact reproduction.

## Current pipeline

Natural Stories reading regions → GPT-2 / DistilGPT-2 word surprisal →
subject-averaged RTs plus lexical/spillover predictors → OLS baseline and held-out residual
analysis → layerwise representational-change analysis (implemented; full GPU extraction pending).

## Data

- `data/naturalstories/naturalstories_RTS/all_stories.tok`: authoritative displayed
  text; 10,256 regions indexed by `(item, zone)`, across 10 stories.
- `processed_RTs.tsv` in that directory: authoritative participant observations.
  Arithmetic mean RT in milliseconds is computed directly from all 848,875 rows;
  `n_observations` is retained. Duplicate participant/region keys, mismatched words,
  invalid RTs, or missing regions cause an error. No additional exclusions.
  Upstream filtering and alignment corrections are already applied; do not repeat them.
- `processed_wordinfo.tsv` is intentionally unused because its summaries disagree
  with the current RT observations.
- `data/naturalstories/freqs/freqs-1.tsv`: supplied Google Books counts, described
  by the corpus as summed from 1990 onward. No external frequency data is fetched.
  We join `.word` entries by region ID, check compacted `.whole` text against the
  displayed word, and use natural-log `log(1 + count)`. Zero counts remain zero;
  absent/empty forms or mismatches remain missing. Multi-part `.word` forms can
  represent phrase counts rather than unigrams; `frequency_form` and
  `frequency_is_multiword` expose this limitation. These are corpus lexical counts,
  not normalized probabilities. Word length counts displayed characters including
  punctuation. Ten regions have no usable lexical frequency form.

## Run

From the repository root, with Python 3.10+ and dependencies installed:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m experiments.extract_surprisal --model gpt2
python -m experiments.extract_surprisal --model distilgpt2
python -m experiments.build_analysis
```

Extraction may download the requested Hugging Face model on first use. Use
`--local-files-only` to require cached files. Tests use the cached GPT-2 tokenizer
and never download it. `test_tokenization.py` is an optional local mapping display.

For Colab with the repository/data available and a GPU runtime:

```bash
python -m experiments.extract_surprisal --model gpt2 --device cuda
python -m experiments.extract_surprisal --model distilgpt2 --device cuda
python -m experiments.build_analysis
```

To prepare human RTs and lexical controls before full model extraction:

```bash
python -m experiments.build_analysis --human-only
```

## Decisions and outputs

Each model loads once per extraction run. Automatic device selection prefers CUDA,
then MPS, then CPU; `--device` overrides it. Each story is reconstructed by joining
its exact reading regions with one space, without a leading space. It is tokenized
once. BPE character offsets are assigned by overlap with known region spans;
separator spaces are ignored for ownership, punctuation remains in its region,
and ambiguous mappings cause an error. Word surprisal sums its BPE surprisals in bits.

Logits at position `i-1` score token `i`. No BOS is inserted. The entire first word
of each story has `NaN` surprisal because its first BPE lacks preceding context;
its later BPEs are never reported as a partial word score. All other words must
have finite, nonnegative scores. Context resets only at story boundaries.

For GPT-2's 1,024-position window, the first window scores all available targets
except the first token. Later overlapping windows score at most 256 new tokens
(`--stride`), each exactly once. These targets have at least 768 preceding BPEs;
context length varies within a window. Position indices restart within each window.
This is a practical bounded-context approximation, not maximal-context scoring at
every position. No future token contributes to a prediction under causal masking.

Outputs live under `results/`:

- `{gpt2,distilgpt2}_surprisal.tsv`: `item, zone, word, surprisal, n_bpe`.
  JSON sidecars record model/revision, device, window/stride, boundary policy,
  row count, and multi-BPE count. Validation requires exact corpus coverage/text.
- `human_predictors.tsv`: all 10,256 word means and lexical predictors, without models.
- `analysis.tsv`: both models joined by region ID, with previous-word length,
  log frequency, and each model's surprisal. Previous-word values never cross stories.
  The builder requires both complete model files unless `--human-only` is explicit.
- `*_validation.json`: observation counts, exclusions, and missing-frequency counts.

Missing predictors are preserved, not dropped or imputed. First words lack lagged
predictors; second words have missing lagged surprisal. The regression stage below explicitly reports complete-case exclusions.
No interpretability method is implemented yet. Generated tables are not source code.


## Psychometric baseline and held-out mismatch

Run the analysis against the existing saved tables (does not regenerate surprisal):

```bash
python -m experiments.run_regression
```

The runner checks both model files, both validation reports, and the saved human
and analysis tables against means/lexical values reconstructed from corpus inputs.
For each model separately, fit unweighted OLS with an intercept:

```
mean_RT ~ surprisal + previous_surprisal + word_length + previous_word_length
          + log_frequency + previous_log_frequency
```

RT is the arithmetic subject mean in **milliseconds**, without log transformation
or observation-count weighting. Surprisal is in bits. Frequency already uses
`ln(1 + supplied count)`; it is not logged again. All six predictors are z-scored
using training-set means and population SDs (`ddof=0`). The outcome is not scaled.

Ten shuffled word-level folds use NumPy's `default_rng(2026)` on rows sorted by
`(item, zone)`. Both models have identical eligible rows and fold assignments.
Controls-only regressions use the same four lexical controls on those same rows
and folds. Each word receives exactly one prediction from the other nine folds.
Scaling is fitted independently within each training fold. Reported RMSE and R²
are pooled across all held-out predictions; R² uses the eligible outcome's overall
mean as its denominator reference, not an average of fold R² values.

The mismatch is **observed RT minus held-out predicted RT**. Positive values mean
slower-than-predicted human reading. It is never taken from the full-data fit.

Both models retain **10,216 of 10,256** regions. Forty unique exclusions comprise
10 first words, 10 second words whose previous surprisal is missing, 10 words
without usable frequency, and their 10 successors. First words also lack previous
predictors, so reason counts overlap. Exclusion TSVs identify every omitted row
and every applicable reason. No additional trimming, imputation, or outlier removal.

### Saved outputs

`results/regression/` contains, separately for each model:

- `*_heldout_residuals.tsv`: original columns plus observed/predicted RT, residual,
  current surprisal alias, fold (0–9), controls-only prediction and residual.
- `*_exclusions.tsv`: all excluded regions and explicit overlapping reasons.
- `*_coefficients.tsv`: full-data standardized OLS coefficients (ms per predictor
  SD), ordinary standard errors, t statistics, and two-sided Student-t p-values.
  The intercept is predicted RT at mean predictor values; residual df = N − 7.
- `*_fold_audit.json`: train/test sizes and training-only scaling parameters.
- `*_extremes.tsv`: 20 largest positive and 20 largest negative held-out residuals.
- `summary.json`: source validation, exclusions, held-out metrics, improvements,
  residual diagnostics, and validation checks.

`results/figures/` contains PNG and PDF plots for each model: observed versus
held-out predicted RT, residual histogram, and surprisal versus RT with an
unadjusted linear trend. `cv_performance` compares controls and full models.

### Interpretation limits

These random word folds estimate held-out-word prediction within this corpus;
training and test folds share stories and participants, and neighboring words
can fall in different folds. They do not estimate unseen-story generalization.
Ordinary OLS standard errors assume independent, homoscedastic errors; the reported
p-values are descriptive and do not adjust for story/participant dependence.
Residuals have a pronounced positive tail; extreme words are retained and listed
for inspection, without attributing their residuals to attention or information flow.
DistilGPT-2's small numerical performance advantage is not a significance test of
model differences. The supplied frequency-form limitations above still apply.


## Layerwise representational change

This stage tests associations between Transformer representational dynamics and
held-out human–Transformer mismatch. These distances are not cognitive effort,
processing difficulty, literal information flow, or causal evidence. No Attention
Flow, rollout, Integrated Gradients, or additional model is implemented.

### Extraction

Run from the repository root on the Colab GPU containing the same corpus and
saved surprisal/baseline results:

```bash
python -m pip install -r requirements.txt
python -m experiments.extract_representations --model gpt2 --device cuda
python -m experiments.extract_representations --model distilgpt2 --device cuda
python -m experiments.analyze_representations
```

The two model runs load weights once each. The extraction checks model revision
and context size against the corresponding surprisal metadata, reads its stride,
and uses the shared `context_windows` and `encode_regions` functions. Refactoring
the window iterator did not alter the previously validated surprisal schedule.
No surprisal or baseline-residual output is rewritten.

Use `--local-files-only --device cpu --smoke-words 32` for a cached local check.
Smoke outputs go exclusively to `results/representations/smoke/` and cannot be
used by the full analysis runner. Full CPU extraction was intentionally deferred.

Each forward uses `output_hidden_states=True`. There are 13 states for GPT-2
(embedding/position state plus 12 blocks), and 7 for DistilGPT-2 (plus 6 blocks).
**Final normalization convention:** native Hugging Face output includes `ln_f` in
the last returned state. A temporary pre-hook on `ln_f` captures the raw last
block output and substitutes it for that last state. Thus all transitions compare
raw residual-stream block outputs consistently, rather than conflating the last
block with a separate normalization operation. Tests compare the states against
hooks on every block for both architectures. Initial dropout is disabled in eval.

Representations are read at each BPE's **own position**, after it has been seen;
this differs conceptually from surprisal, which predicts that BPE from the previous
position. Input text, tokenization, windows, local position indices, and token
ownership are identical to surprisal. Token zero has a valid representation even
though its word has unscored surprisal. Its representation is extracted in the
initial window, but the established baseline exclusions remain in force.

At each depth, all constituent BPE states are arithmetic-mean pooled into one
word vector, **then** adjacent-layer distances are computed. If pieces of a word
span window assignments, each retains its own established surprisal-window
context before pooling; we do not move the entire word to a different window.
Only one window's hidden states and per-story pooled sums are held in memory;
full corpus hidden tensors are not saved.

### Features

For pooled vectors h[l], h[l+1]:

- `cosine_layer_k = 1 - cosine(h[k-1], h[k])` (primary).
- `relative_l2_layer_k = norm(h[k]-h[k-1]) / (norm(h[k-1]) + 1e-12)`.

Layer k is the **1-based destination block**, so layer 1 compares embeddings with
block 1. For both measures the compact summaries are total, mean, maximum,
peak-layer index (earliest on ties), early/middle/late means, and late-minus-early.
A zero-norm or nonfinite representation fails validation instead of inventing a
cosine value or silently dropping the word.

| Model | Early blocks | Middle blocks | Late blocks |
|---|---|---|---|
| GPT-2 | 1–4 | 5–8 | 9–12 |
| DistilGPT-2 | 1–2 | 3–4 | 5–6 |

`results/representations/{model}_trajectories.tsv` preserves identifiers, exact
text, BPE count, every adjacent cosine/L2 value, and all summaries. JSON sidecars
record model revision, state convention, context/pooling policies, definitions,
and the SHA-256 of the source surprisal table. Complete extraction requires all
10,256 regions with exact text/BPE agreement, unique keys, and finite features.

### Analyses

The runner joins to the existing 10,216-word held-out residual table, verifies
text and saved fold assignments, and permits no additional silent exclusions.
It fails if full feature tables are unavailable. Original residuals are unchanged.

1. Pearson and Spearman correlations at every layer with signed and absolute
   residuals. Benjamini–Hochberg FDR is applied **separately within model × distance
   measure × residual target × correlation method**, across that model's blocks.
   These are exploratory families, not a single correction across every analysis.
2. Standardized descriptive OLS with intercept: signed residual (and separately
   absolute residual) ~ early + middle + late mean cosine. Repeat with relative L2.
   Report ms per predictor SD, ordinary SE/t/p and in-sample R². No residual-regression
   fitted values replace the established mismatch variable.
3. Primary incremental test: RT ~ existing six baseline predictors + three third
   means, separately for cosine and relative L2. Read the **exact saved folds**;
   fit every predictor's scaling only on training folds. Verify the baseline
   predictions reproduce numerically, then compare the saved baseline with new
   held-out extended predictions using pooled RMSE and R². No feature selection
   is performed on held-out outcomes.
4. Pearson/Spearman confound diagnostics for summaries versus current surprisal,
   displayed length, ln(1+frequency), and BPE count. These p-values are unadjusted
   descriptive diagnostics. A constant summary (e.g. peak block) is explicitly
   marked `undefined_constant`, rather than assigned a spurious correlation.

Tables, source hashes, CV performance summaries, and training-fold scaling audits
are saved under `results/representations/analysis/`. Five figure sets per model
(PNG/PDF) go under `results/figures/representations/`: signed/absolute layer profiles
with FDR markers, third distributions, baseline/extended performance, and mean
trajectories for bottom/top absolute-residual quartiles. Quartile grouping is only
for visualization; ties at cutoffs are included, group sizes shown, and all
statistical analyses use the complete continuous eligible dataset.

### Limits and current execution status

Local real-model smoke extraction passed for 32 regions/model (3 multi-BPE words),
with exact cached revisions and 13/7 states. Eighteen automated tests passed,
including manual distance calculations, cross-window multi-BPE pooling, raw block
state checks, FDR, exact fold reuse, training-only scaling, and end-to-end synthetic
analysis/figure generation in temporary directories. Synthetic values are never
written as scientific results. Full extraction and real-data association/regression
figures remain pending the GPU commands above; no trajectory findings are claimed.

Word-level folds share stories and participants. Descriptive correlation and OLS
p-values assume independence that may not hold here, including dependence induced
by overlapping training sets for the baseline residuals. FDR does not remove these
limitations. Features are post-token representations, affected by token identity,
BPE pooling, local positions, and bounded context. Cross-model depths are relative
thirds, not matched learned transformations. Relative L2 is scale-sensitive; cosine
and relative L2 need not agree. The incremental held-out RT test is the primary
predictive check, and neither test identifies a causal mechanism.
