"""Fixed candidate definitions and selection rule, written before new evaluation."""
FAMILIES={
 'distribution_shape': {'features':['target_log_rank','top10_mass','top_margin','collision'],
  'definition':'Pre-FIRST-BPE log(1+target rank), top-10 mass, p(top1)-p(top2), sum(p^2); current and previous region.',
  'motivation':'Distinguish competition among alternatives at comparable surprisal/entropy.',
  'confound':'Target rank describes the first BPE, while controlled surprisal sums all BPEs; inspect BPE correlations.'},
 'extended_spillover': {'features':['lag2_surprisal','lag2_entropy','lag2_available'],
  'definition':'Two-region-back surprisal and entropy from the complete context sequence; absent context encoded zero with an availability flag.',
  'motivation':'Test whether a longer history of model predictions captures delayed RT effects.',
  'confound':'May proxy omitted prior-word lexical effects or temporal autocorrelation; not evidence of retrieval.'},
 'nonlinear_uncertainty': {'features':['surprisal_sq','entropy_sq','surprisal_entropy','target_word_probability'],
  'definition':'Current and previous S^2, H^2, S*H, and 2^(-S). No outcome-dependent thresholds.',
  'motivation':'Test whether a linear readout discards predictive structure already in surprisal and uncertainty.',
  'confound':'Deterministic re-expression of existing predictors, not new Transformer information.'},
 'trajectory_shape': {'features':['cosine_roughness','cosine_center','cosine_peak_share','relative_l2_roughness','relative_l2_center','relative_l2_peak_share'],
  'definition':'For each distance profile: sum(abs(diff(d)))/sum(d), distance-weighted normalized layer depth, max(d)/sum(d).',
  'motivation':'Test nonlinear profile shape beyond the historical all-layer/third-based readouts.',
  'confound':'Derived from existing adjacent-distance scalars; not vector curvature or literal information flow.'}
}
SELECTION='Per outer training set: family improves pooled 5-fold inner MSE and wins at least 4/5 inner folds vs the strong reference. Best is lowest inner MSE among qualifying families; combine all qualifying families once. Empty selection falls back to reference.'
DEFERRED={'attention':'No compact validated attention artifacts; head/layer search would expand multiplicity substantially.',
 'truncated_context':'Requires additional matched short-context scoring; deferred rather than changing the context regime.',
 'logit_lens':'Intermediate unembedding requires calibration/normalization choices; not introduced in this first bounded exploration.',
 'vector_geometry':'Full hidden vectors are not saved; distance profiles cannot recover displacement or vector curvature.'}
