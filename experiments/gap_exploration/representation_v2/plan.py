"""Write the fixed plan once, before observing any new outcomes."""
from pathlib import Path
from src.extension_io import new_json
from src.gap_exploration.representation_v2.features import DEFINITIONS


if __name__ == '__main__':
    new_json(Path('results/gap_exploration/representation_v2/plan.json'), dict(
        families=DEFINITIONS, primary=['natural_stories/rt/gpt2', 'natural_stories/rt/distilgpt2',
        'geco/gaze/gpt2', 'geco/gaze/distilgpt2'],
        baseline='Current/previous lexical length, frequency, BPE, surprisal, entropy PLUS lag2 length/frequency/BPE/frequency-availability/surprisal/entropy/model-availability.',
        original_trajectory='Historically strongest trajectory architecture, refitted with the stronger nonrepresentation baseline.',
        pca='Fixed 16 components per layer, randomized SVD seed 2026, fitted separately on every inner and outer training subset. No full-data PCA. Float64 CPU.',
        typicality='Training mean; variance-shrinkage floor = 0.1 * mean retained PC variance. No outcome fitting or test references.',
        selection='Five inner folds seed 2026+outer. A family qualifies if pooled inner SSE improves baseline and wins >=4/5 inner folds. Best = lowest qualifying SSE; combined = all qualifiers. If layer_combination qualifies, omit individual layer PCAs from combined to avoid duplicate columns. No qualifying family => baseline. One fixed combined rule.',
        tuning='Existing 25 ridge alphas logspace(-4,4). Fit all preprocessing on inner training, refit on outer training. Exact frozen outer folds and samples.',
        context='Zero,16,64 prior BPE tokens, retaining complete current-word prefix and original full-window position IDs. Full-context token ownership unchanged. No retokenization, BOS, or position reset in truncated comparisons.',
        scope='Four primary cases. GECO total deferred as secondary to limit outcome search. No adaptive feature revisions based on outer scores.',
        interpretation='Own-token representations include lexical identity. Incremental prediction is not next-word anticipation or causal processing effort. Frozen folds are within-corpus, not unseen-story tests.'))
