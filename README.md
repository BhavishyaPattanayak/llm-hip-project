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

## BPE-count control robustness

`python -m experiments.run_bpe_control` runs a separate robustness analysis using
saved representation joins and the exact fold IDs in the established residual
files. It never generates folds or changes primary outputs. It refuses to overwrite
an existing `results/representations/analysis/bpe_control/` directory.

The BPE-controlled baseline adds raw current and previous BPE counts to the six
original predictors. Previous BPE count is looked up at `(item, zone-1)` in the
**complete trajectory table**, including words excluded from regression; it never
means the previous retained analysis row or the end of the previous story.
Cosine and relative-L2 extensions separately add their three third-mean features.
All predictors use training-fold-only standardization; RT remains in milliseconds.

Outputs include model-specific summary JSON, aligned prediction TSVs, fold/story
cosine stability tables, and scaling audits. A SHA-256 preservation report verifies
that all pre-existing result files remain unchanged. Story stability summarizes
out-of-fold predictions within each story; it is not leave-one-story-out validation.
Positive RMSE improvement means the extension beats the BPE-controlled baseline.
Both models retain exactly 10,216 observations with no additional exclusions.

## Extensions: nested ridge, splines, entropy, and GECO

The completed Natural Stories baseline, trajectory, and BPE-control files remain
immutable. New scripts refuse to overwrite their output files/directories. New
analyses test prediction beyond the controls; they do not establish causality or
an architecture effect. DistilGPT-2 is a robustness model. No novelty claim is made.

### Local GECO input and human outcomes

Expected files: `data/geco/MonolingualReadingData.xlsx` and
`data/geco/EnglishMaterial.xlsx`. The **DATA** sheet of MonolingualReadingData is
authoritative. EnglishMaterial is not used to construct, filter, or reduce stimuli.
The loader checks WORD_ID uniqueness and participant agreement on WORD, PART,
TRIAL, and WORD_ID_WITHIN_TRIAL, and rejects duplicate participant–word observations.
It does not assume a balanced participant matrix.

Primary outcome: arithmetic subject mean of valid **WORD_GAZE_DURATION** in ms.
Secondary: mean valid **WORD_TOTAL_READING_TIME**. `.` is missing, never zero.
Nonpositive/nonfinite durations are separately counted and excluded from that DV's
mean; all regions remain in preprocessing. WORD_SKIP is recorded separately:
GECO has valid durations even when WORD_SKIP=1, so a valid observed duration is
retained regardless of that flag. Per-region participant counts, valid-DV counts,
gaze-observation proportion, skip proportion, and both means are saved.

Regions are ordered by PART, TRIAL, WORD_ID_WITHIN_TRIAL. Each (PART, TRIAL) is an
independent LM context. `item` is a deterministic trial index, and `zone` is a
contiguous ordinal of **observed** regions within that trial. Original WORD_ID and
WORD_ID_WITHIN_TRIAL are preserved. Position gaps are reported, not filled with
invented words. Previous-word predictors use the preceding observed reading region
and reset at trial boundaries. Every region in DATA is retained for extraction.

Some experimental reading regions contain internal spaces. A GECO-only option in
the existing offset-to-region aligner preserves the entire region and punctuation,
including internal spaces, rather than splitting it into multiple human words.
Natural Stories default alignment is unchanged. Some Excel WORD cells are numeric
or boolean; numeric cells use their decimal display text and boolean cells use
Excel's TRUE/FALSE display. These source-coerced regions and their IDs are reported;
original capitalization lost in a boolean cell cannot be recovered from DATA.

### Frequency controls and exclusions

GECO reuses the locally supplied Google Books `.word` counts from Natural Stories.
The lookup is exact and case-sensitive after stripping outer ASCII/curly quotation
punctuation (embedded punctuation is retained). Only unambiguous single-form entries
are used; multiple counts for the same form or unmatched words remain missing.
The transformation is **ln(1 + count)**, exactly the existing scale. This limited
vocabulary is not a comprehensive GECO frequency resource. Coverage and exclusions
are reported explicitly; novel-rare words can be disproportionately excluded. No
new frequency dataset, GECO-derived frequency scale, imputation, or zero filling is
introduced. Displayed word length includes punctuation and internal spaces.

All GECO variants use the same complete-case sample within each DV/model, including
availability of entropy and all-layer features; both models' sample keys/folds are
checked when both runs exist. Missing-current/previous predictors and missing DV
counts are reported separately with overlap and row-level exclusions. GECO uses
new deterministic shuffled word folds (seed 2026); Natural Stories reads its saved
fold IDs, even if an entropy exclusion were required. Trial/story stability uses
subsets of held-out predictions, not unseen-context validation.

### Contextual entropy and extraction

Entropy is Shannon entropy in **bits** of the next-token distribution immediately
**before the first BPE** of a reading region. It is not averaged across within-word
pieces. Compute `-sum(exp(log_softmax(logits))*log_softmax(logits))/ln(2)` at that
first BPE's preceding position in its assigned window. First context-region entropy
is NaN under no-BOS, like first-word surprisal. Current/previous entropy strengthen
the lexical + surprisal + BPE baseline.

Extraction pins each model and tokenizer to the existing Natural Stories revision,
reads the existing window size/stride, and reuses the exact target-window schedule.
Hidden-state, raw-final-block, and mean-pooling conventions are unchanged. The GECO
`--features all` mode loads each model once: one forward pass per window jointly
computes surprisal and entropy, and the established representation extractor makes
a second pass. Vocabulary distributions and full corpus hidden tensors are not
saved. Metadata records revision, tokenizer, context reset, stimulus checksum,
policies, row counts, and smoke/full status. Smoke mode repeats probability
extraction to verify determinism and writes only under separate `smoke/` paths.

### Nested ridge and fixed nonlinear model

Outer folds are untouched during tuning. Each outer-training set has deterministic
5-fold inner CV (`random_state = 2026 + outer_fold`). Alpha is selected by pooled
inner held-out squared error from **25 log-spaced values from 1e-4 through 1e4**;
ties choose the first/smallest alpha. Every inner fit estimates its own preprocessing,
then the selected model is refitted on the full outer-training set. Predictions
are generated once per outer-test row. No test-dependent feature selection occurs.

Ridge penalizes all standardized predictors, not the intercept. A centered SVD
implements the ridge alpha path efficiently and is tested against sklearn Ridge.
Per-outer-fold alphas, inner losses/splits, scaling parameters, and descriptive
standardized coefficients are saved. These coefficients are not inferential tests.
A separately tuned ridge-controls model is also reported to distinguish estimator
changes from added representation features; the primary reference remains the
explicitly named OLS baseline (saved BPE-control predictions for Natural Stories).

All-layer models use all 12/6 cosine changes, with equivalent relative-L2 robustness.
The fixed nonlinear configuration is `SplineTransformer(degree=3, n_knots=4,
knots="quantile", include_bias=False)`, default constant extrapolation. Only the
three early/middle/late trajectory variables are expanded. Baseline predictors stay
linear. Quantile knots are learned only inside each training fit, then the complete
design is standardized and ridge-regularized. Cosine is primary; relative-L2 splines
are also reported. No complexity search is performed.

GECO gaze and secondary total-RT runs include all specified lexical/surprisal,
BPE, and entropy baselines, linear-third, all-layer, and spline trajectory models,
plus relative-L2 robustness. All comparisons remain in outputs whether helpful or
harmful. Natural Stories without entropy reuses completed linear-third predictions;
the entropy stage fits new entropy-controlled models without changing residuals.

### Commands and output locations

Local preparation (already run; refuses to overwrite):

```bash
python -m experiments.prepare_geco
```

Dependencies: `python -m pip install -r requirements.txt`. Preparation/tests need
cached GPT-2/DistilGPT-2 tokenizers. The full local test suite is:

```bash
MPLCONFIGDIR=/tmp/hsp-matplotlib python -m unittest discover -s tests -v
```

Small CPU smoke commands (2–100 words, first context only):

```bash
python -m experiments.extract_extensions --dataset natural_stories --features entropy --model gpt2 --device cpu --smoke-words 32 --local-files-only
python -m experiments.extract_extensions --dataset natural_stories --features entropy --model distilgpt2 --device cpu --smoke-words 32 --local-files-only
python -m experiments.extract_extensions --dataset geco --features all --model gpt2 --device cpu --smoke-words 32 --local-files-only
python -m experiments.extract_extensions --dataset geco --features all --model distilgpt2 --device cpu --smoke-words 32 --local-files-only
```

On Colab T4, with the repository and current processed/saved results present:

```bash
python -m experiments.extract_extensions --dataset natural_stories --features entropy --model gpt2 --device cuda
python -m experiments.extract_extensions --dataset natural_stories --features entropy --model distilgpt2 --device cuda
python -m experiments.extract_extensions --dataset geco --features all --model gpt2 --device cuda
python -m experiments.extract_extensions --dataset geco --features all --model distilgpt2 --device cuda
```

Each GECO `all` command produces **surprisal, representations, and entropy** for
that model, avoiding three separate model loads. Individual `--features surprisal`,
`--features representations`, and `--features entropy` modes are also available.
Full GECO GPU extraction is never launched by tests or preprocessing.

Feasible Natural Stories analyses using existing features (already run):

```bash
python -m experiments.run_extensions --dataset natural_stories --stage trajectory
```

After full GPU outputs return to the project, run locally:

```bash
python -m experiments.run_extensions --dataset natural_stories --stage entropy
python -m experiments.run_extensions --dataset geco --dv gaze
python -m experiments.run_extensions --dataset geco --dv total
python -m experiments.report_extensions --output results/extensions/overview_final
```

New Natural Stories outputs: `results/extensions/natural_stories/trajectory/`,
`entropy/`, and `entropy_trajectory/`. GECO outputs: `results/geco/processed/`,
`surprisal/`, `representations/`, `entropy/`, and `extensions/{gaze,total}/`.
Each regression run contains explicit comparisons, row-level held-out predictions,
fold/context stability, exclusions, input provenance, and fitted-model audits.
Figures are in each run's `figures/` subdirectory; overview figures combine all
available models/datasets. They include percentage RMSE improvement, delta R²,
fold/context stability, and descriptive layer–RT relationships. Plot ranges are
not clipped to selected effects. No full entropy-controlled findings are claimed
before GPU extraction completes.

### Validated local extension results

Full GECO DATA validation found 774,015 participant rows, 14 participants, 56,410
unique regions and 588 trials. Both cached tokenizers align all regions with zero
failures and 15,084 multi-BPE regions. There are 472,151 valid gaze observations
and 472,151 valid total-RT observations. Total-RT aggregation excludes 965 reported
nonpositive values rather than treating them as observed positive reading times.
The explicit skip proportion is 51.632%; 97,776 valid-duration rows are also marked
skipped and are retained as observed durations. Frequency is missing for 14,856
regions (26.34%), before accounting for previous-frequency exclusions.

Input diagnostics retain 44 whitespace-containing regions, 22 numeric WORD cells,
7 boolean WORD cells, and 14 trials with noncontiguous original positions. Pure
separator-space BPEs attach to the following reading region; spaces inside a DATA
region remain part of that region. No human region is silently split or dropped.

Natural Stories runs retained 10,216 rows/model and preserved the original folds.
Against the completed BPE-controlled baseline, cosine all-layer ridge improved RMSE
by 1.415% (GPT-2) and 2.721% (DistilGPT-2); fixed spline thirds improved it by 1.492%
and 3.157%. All-layer relative-L2 improvements were 1.657% and 2.127%; relative-L2
spline improvements were 1.192% and 1.786%. Every specified comparison, including
ridge-controls and existing linear thirds, is saved. These numerical differences
are not architecture evidence. Alpha selections and descriptive coefficients remain
in the per-run files. Both models passed 32-word GECO all-feature and Natural Stories
entropy smoke extraction. Full GECO extraction and full entropy controls await GPU
execution; no second-dataset or entropy-controlled result is claimed yet.
