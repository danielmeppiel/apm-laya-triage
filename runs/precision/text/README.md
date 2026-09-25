# Cheap supervised text and hybrid baselines

**Development-only silver-label agreement, not adjudicated precision.**
The frozen primary policies use TF-IDF plus saved Laya scores. Pure text
accounts for most of the improvement; whether the smaller hybrid gain is
worth fresh Laya inference remains a parent-owned confirmation question.
No confirmation outcomes were accessed to build or select these policies.

## Frozen development results

All rows below use the same 214 development controls, with 565 positive
labels in observed dimensions. Missing whole dimensions are unscored,
not negative. Coverage means at least one proposed label in an observed
dimension. Mean labels counts *all* emitted dimensions, including unscored
ones.

| Frozen policy | Precision | Recall | F1 | Exact | Coverage | Mean labels |
|---|---:|---:|---:|---:|---:|---:|
| Primary: hybrid balanced | .7315 | .8053 | .7666 | .4907 | 1.0000 | 4.1822 |
| Primary: hybrid broad precision | .8517 | .5894 | .6967 | .4439 | .8738 | 2.4579 |
| Comparator: text balanced | .7827 | .7204 | .7502 | .5047 | .9953 | 3.3178 |
| Comparator: text broad precision | .8275 | .5434 | .6560 | .3785 | .8271 | 2.4813 |

The two primary policies use the same model: word unigrams/bigrams,
15,000-feature cap, `min_df=2`, sublinear TF, L2-normalized TF-IDF, title
repeated three times, and one-vs-rest logistic regression with
`C=4`, `solver=liblinear`, `tol=1e-6`, `max_iter=1000`, `random_state=0`.
Each label trains only on fit issues with an observed label in that
dimension. There is no class weighting. The hybrid appends all 25 saved
Laya scores, standardized using fit rows and multiplied by `0.10`.

The balanced threshold is `0.225`. Broad precision uses thresholds
`type=0.80`, `area=0.275`, `theme=0.60`. Each dimension separately had
development recall at least 0.50; this is not a guarantee on new issues.
The matched text-only references use the same lexical configuration,
with balanced threshold `0.30` and broad thresholds
`type=0.825`, `area=0.275`, `theme=0.575`.

| Dimension | Dev control issues | Positive support | Hybrid broad P | Hybrid broad R | Dimension coverage |
|---|---:|---:|---:|---:|---:|
| type | 212 | 258 | .9944 | .6899 | .6462 |
| area | 145 | 213 | .7152 | .5070 | .8828 |
| theme | 91 | 94 | .7705 | .5000 | .6703 |

Detailed per-label support, proposals, P/R/F1, exact agreement, label counts,
and per-dimension coverage are in each `summary.json`. Per-issue development
predictions for the four frozen policies are in `frozen/*-development.jsonl`.
The model scores are **not validated probabilities of label correctness**.

## Small-step experiment history

No feature/regularization grid or neural inference was run.

| Step | Model | Best pooled development P / R / F1 | Finding |
|---|---|---|---|
| 1 | Fit-only most-common label per dimension | .4911 / .3894 / .4344 | Meaningful no-reading comparator |
| 1 | Fit prevalence, pooled threshold | .4021 / .5345 / .4590 | More recall, many prior-driven proposals |
| 1 | Word TF-IDF, title weight 1 | .7279 / .7717 / .7491 | Large lexical/supervised gain |
| 2 | Title weight 3 | .7827 / .7204 / .7502 | Only +.0011 F1; negligible balanced improvement |
| 2 | Five nearest labeled texts, cosine-weighted | .7920 / .7009 / .7437 | Does not beat logistic; no neighbor sweep |
| 3 | All-25-score-only logistic | .7842 / .5982 / .6787 | Useful multivariate signal, weaker than text |
| 3 | Title-3 hybrid, score weight .25 | .7687 / .7469 / .7576 | Small balanced improvement |
| 4 | Title-3 hybrid, score weight .10 | .7315 / .8053 / .7666 | Final feature variant; no further sweep |
| Diagnostic | Title-3, remove leading bracketed type tags | .8028 / .6991 / .7474 | Only -.0028 F1; template tags are not the main gain |
| Diagnostic | Title-3, exact retained Laya input excerpts | .7765 / .7133 / .7435 | Only -.0067 F1; longer bodies are not the main gain |

The initial global precision-first policies were misleadingly narrow:
pure text reached .9314 precision at .5044 overall recall, but area recall
was only .1925. A hybrid global threshold reached .9630 precision, but
that operating point was **not frozen**. The parent requested one bounded
improvement: separate dimension thresholds with a recall floor in each
dimension. The final broad-precision policy is less flashy and more useful.

The final hybrid improves development F1 by .0164 over the matched text
balanced reference; broad precision improves by .0242 with recall also
increasing .0460. These are selected development differences, not
confirmation estimates or established statistical significance. The text
balanced reference has better precision and exact agreement than hybrid
balanced, so hybrid is not a uniform improvement.

### Robustness diagnostics

The marker ablation removes only leading bracketed tags such as `[BUG]`,
`[FEATURE]`, and `[DOCS]`; the body and ordinary semantic words are unchanged.
For the budget diagnostic, the prompt sibling generated exact
`prepare_state` baseline head/tail excerpts using tokenization only. It
removed the fixed Project prefix *after* shortening, without reallocating
tokens. All 891 fit/development input hashes and baseline state hashes were
checked. Of these, 825 were shortened. All retained `Title:` / `Body:`
delimiters survived, allowing the same title-weight-three model.
The deduplicated replay cache is `title3-excerpts-v1/excerpts.jsonl`;
it contains no labels or confirmation rows.

The excerpt broad-precision diagnostic reaches P=.7900/R=.5593/F1=.6549,
versus full-text P=.8275/R=.5434/F1=.6560. Diagnostics are explicitly
ineligible for finalist selection.

Fit/development metadata counts: 513/677 fit and 157/214 development titles
start with a bracket; only 6 fit texts and 0 development texts contain a
literal canonical taxonomy name. This is not a causal audit of label
provenance: current labels may reflect templates, prior automation or
human choices, and are not gold.

## Protocol and limitations

- Shared immutable protocol: `runs/precision/protocol.json`, content digest
  `dbc55b2ab2100d37167d36a631f7d1134f6cf473efeba9cd28829b802a74b8ea`.
  Fit uses exactly 677 control issues; development uses exactly 214.
  Parent owns the reserved 188 confirmation controls.
- Vocabulary, IDF, score standardization, logistic coefficients, fit priors,
  and nearest-neighbor index use fit rows only. Unknown entire dimensions
  do not contribute negative training examples. Constants and unsupported
  labels have explicit support/status, including all-zero abstention for
  no fit observations.
- Every threshold comes from the predeclared fixed 37-point grid
  `0.05, 0.075, ..., 0.95`. Coefficients are fit-only; pooled scalar
  thresholds and the three dimension scalars are development
  hyperparameter choices, as explicitly approved by the parent.
  Balanced maximizes F1, then precision, recall, threshold. Broad precision
  maximizes precision independently per dimension subject to recall
  >=.50, then F1, recall, threshold; the winning model maximizes aggregate
  precision, then F1. Infeasible policies are reported, never silently
  relaxed.
- Inputs whitelist title/body; attached labels, state, comments, URLs,
  timestamps, and issue numbers never enter lexical features. Issue
  numbers are join keys only; numeric issue references in text are removed.
  Hybrid adds only cached model responses, not their stored labels.
- These are supervised silver-reference models, not zero-shot Laya.
  Development is reused for selection and therefore optimistic.
  All-corpus baseline results had already been inspected historically;
  the reserved split is protected from these new experiments, not a
  pristine external holdout. Lexical grouping cannot remove all semantic
  duplication, and sparse labels/rare-class results remain uncertain.

## Latency and resource use

All fitting and prediction use one CPU thread and no neural/GPU inference.
The environment is separate from the production Laya environment.

| Model run | Fit seconds | Dev-214 batch seconds | Median single issue ms |
|---|---:|---:|---:|
| Text title-3 | .9273 | .0779 | 2.321 |
| Hybrid title-3, weight .10 | 1.0937 | .0660 | 2.596 |
| Score-only logistic | .0440 | .0029 | 1.728 |
| Exact-excerpt text | .2096 | .0168 | 1.837 |

Single-issue medians use the first 16 development rows in fixed protocol
order. Timings cover warm in-process preprocessing and classical model
work only; batch timing differences at this scale are noisy. They exclude
startup, disk I/O, and **Laya score generation**. The baseline's already
recorded development inference median was **7.578 seconds/issue**, mean
7.435 seconds, on its historical MPS run with 25 questions. This is not
a matched-hardware timing comparison, but demonstrates why cached hybrid
timing cannot be advertised as end-to-end new-issue latency.

## Isolated setup and replay

Use Python 3.12 and an isolated environment. Do not install these packages
into the production Laya environment.

```bash
python3.12 -m venv /tmp/apm-text-env
/tmp/apm-text-env/bin/python -m pip install -r experiments/requirements-text.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/tmp/apm-text-env/bin/python -m unittest discover -s experiments/tests -q
/tmp/apm-text-env/bin/python -m unittest discover -s tests -q
```

Frozen model artifact: `frozen/manifest.json`. It records exact source,
protocol, dependency and configuration hashes, two primary policies, and
two explicitly named comparator references. Models are deterministically
refit on fit IDs; no unsafe pickle loading is used.

Parent-owned prediction requires a JSONL input file containing **only**
`{"number": ..., "title": "...", "body": "..."}`. The parent selects
confirmation inputs only after freezing its full comparison registry.
Run these four keys separately, choosing new output paths:

```bash
/tmp/apm-text-env/bin/python -m experiments.precision_text \
  --predict-frozen runs/precision/text/frozen/manifest.json \
  --finalist balanced --inputs /path/to/clean-inputs.jsonl \
  --output /path/to/hybrid-balanced.jsonl
```

Other keys are `broad_precision`, `text_balanced`, and
`text_broad_precision`. The latter two are comparator references, not
additional selected finalists. The CLI verifies frozen source/dependency
hashes, fits only fit labels, then uses prediction-only inputs and cached
scores. All four deterministic refits replayed the frozen development
proposal files exactly. For fresh uncached issues the hybrid additionally
needs Laya scores; this offline command deliberately does not produce them.

Programmatic API:

```python
model, policy = refit_frozen(manifest_path, "balanced", repository_root)
cached = read_selected_scores(baseline_path, clean_inputs, model.labels)
rows = predict_records(model, clean_inputs, policy, cached)
```

Text-only comparators pass no cached scores. No `expected` labels are
accepted by the prediction policy. `TextBaseline.fit(...)` is the explicit
supervised boundary; `predict(...)` is separate.

To reproduce one development experiment, select a new output directory:

```bash
/tmp/apm-text-env/bin/python -m experiments.precision_text \
  --kind hybrid --title-weight 3 --score-weight .10 \
  --output /path/under/repository/new-hybrid-run
/tmp/apm-text-env/bin/python -m experiments.precision_text \
  --kind word --title-weight 3 \
  --text-excerpts runs/precision/text/title3-excerpts-v1/excerpts.jsonl \
  --output /path/under/repository/new-budget-diagnostic
```

Existing run directories and frozen files are immutable. `--help` exposes
the bounded dimension-policy and freeze commands. Tests use synthetic
data for masks, constants, metadata isolation, vocabulary/scaler leakage,
score joins, metric alignment, deterministic replay, title-tag handling,
excerpt integrity, recall floors and manifest tampering.
