# Transformer surprisal and human reading times

MSc seminar project testing whether information propagation inside Transformers
explains variation in human reading times remaining after surprisal and basic
lexical controls. This is a small Wilcox-style analysis, not an exact reproduction.

## Current pipeline

Natural Stories reading regions → GPT-2 / DistilGPT-2 word surprisal →
subject-averaged RTs plus lexical/spillover predictors → regression and residual
analysis (not implemented yet) → information-flow analysis (future work).

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
predictors; second words have missing lagged surprisal. Regression transformations,
complete-case exclusions, frequency handling, and residual definition remain to be
chosen after inspecting the complete analysis table. No regression or interpretability
method is implemented yet. Generated tables are not source code.
