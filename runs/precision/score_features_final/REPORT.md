# Cached-score feature experiments: frozen development findings

**No confirmation results are included.** All results below are silver-label
agreement on the fixed 214 development controls after model fitting on the 677
fit controls. The grouped protocol, missing-dimension scoring, previously
inspected-corpus caveat, and unlabelled-issue exclusions are unchanged.

## Bounded models and stopping evidence

After monotone per-label calibration plateaued, three models tested whether
cross-question scores carry additional signal. All models predict the same 25
labels. They receive cached probabilities only: no title, body, issue ID,
expected label, or observed-dimension metadata is a prediction feature.

| Model / diagnostic policy | Precision | Recall | F1 |
|---|---:|---:|---:|
| Fit-majority reference | .4911 | .3894 | .4344 |
| Earlier isotonic top-1 | .6071 | .4814 | .5370 |
| All 25 scores, logistic, balanced | .7842 | .5982 | .6787 |
| Type 8 scores only, logistic, balanced | .6855 | .5363 | .6018 |
| All 25 scores, conservative extra trees, balanced | .7825 | .5858 | .6700 |
| All 25 scores, logistic, global-precision diagnostic | .8865 | .5115 | .6487 |

The all-score logistic reference exactly reproduces the sibling text stream's
score-only result. Cross-label score interactions help substantially beyond
monotone marginal calibration. Reducing features to the eight type scores costs
7.69 points of F1. The single nonlinear alternative does not improve on logistic.
No further model or feature variants were tried.

Logistic fitting reuses `experiments.precision_text.TextBaseline`: fit-only
standardization, score weight .25, C=4, liblinear, tolerance 1e-6, maximum 1000
iterations, random seed 0, no class weights, and per-observed-dimension masks.
Type-only fitting zeroes non-type columns before this same implementation and
exports only the eight retained coefficients and scaling statistics.
The nonlinear alternative uses ExtraTrees with 128 trees, maximum depth 5,
minimum leaf size 10, all features, seed 0, and one CPU worker.

All three use the sibling's fixed 37 scalar thresholds, .05 through .95 in .025
steps, selected on development. The same development data select models and
operating points, so these are selection-biased estimates. Fit runtimes were
approximately .169 s (25-score logistic), .021 s (8-score logistic), and 2.330 s
(trees); these exclude the already-paid neural inference.

## Final two policies

The global-precision diagnostic hides weak dimension recall: type .7907,
area .2113, and theme .4255. It is **not** the final precision policy.
Without refitting or adding models, the same 37-point grid was applied
independently to three dimension thresholds, maximizing precision subject to
recall >= .50 in each observed dimension.

| Final policy | Precision | Recall | F1 | Exact agreement | Any-proposal issue coverage | Observed-dimension coverage | Mean labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Balanced: pooled threshold .425 | .7842 | .5982 | .6787 | .4439 | 1.0000 | .8304 | 2.7991 |
| Broad precision: type .775 / area .200 / theme .550 | .6195 | .5274 | .5698 | .2430 | 1.0000 | .7723 | 3.4206 |

| Dimension | Development controls | Positive-label support | Balanced P / R / F1 | Broad precision P / R / F1 |
|---|---:|---:|---|---|
| Type | 212 | 258 | .8943 / .8527 / .8730 | .9650 / .5349 / .6883 |
| Area | 145 | 213 | .6458 / .2911 / .4013 | .4201 / .5305 / .4689 |
| Theme | 91 | 94 | .6292 / .5957 / .6120 | .6812 / .5000 / .5767 |

**Balanced still has poor area recall.** Broad precision addresses recall, not
uniform per-dimension precision: area precision is only .4201, below the
fit-majority area's .5241. Do not describe either policy as uniformly accurate
across dimensions. Broad precision's observed-issue coverage is .9019 despite
any-proposal issue coverage of 1; some predictions land only in unscored
dimensions. Full supports and coverage definitions remain in `summary.json`.

The final primary candidates are these two 25-score logistic policies. Earlier
isotonic and global-precision artifacts remain diagnostic, not extra primary
confirmation comparisons. No reliable improvement is claimed before the
parent's frozen-plan, paired-group confirmation.

## Rare labels, serialization, and replay

The score models share the sibling's rare-class behavior: an unobserved or
all-negative label produces constant zero; an all-positive observed label
produces constant one. Otherwise labels are learned even at low positive
support, with no additional minimum-positive cutoff. `type/release` has only
one fit positive, so its learned score is particularly uncertain. Trees use the
specified minimum leaf size. Per-label supports are recorded for every model.

Frozen models contain numeric JSON coefficients and preprocessing statistics,
not Python pickle objects or training labels. Experimental tree export also
supports standard-library replay and matches sklearn's float32 feature
comparison semantics. All three exports were compared with native sklearn on
every fit and development row: maximum absolute errors were below 1e-15, with
identical decisions at all 37 thresholds.

The final balanced artifact is
`runs/precision/score_features_final/frozen_balanced.json`.
The final broad-precision artifact is
`runs/precision/score_features_final/frozen_broad_precision.json`.
Both record exact fit/development memberships, model configuration and numeric
model hash, source and protocol hashes, code hash, parent diagnostic artifact
hash, selection evidence, and a self-verifying artifact hash.

Parent-only confirmation replay, after all workstreams and the plan are frozen:

```bash
python3 -m experiments.precision_score_features \
  --replay runs/precision/score_features_final/frozen_balanced.json \
  --split confirmation --allow-confirmation \
  --output runs/precision/confirmation/score_balanced.jsonl

python3 -m experiments.precision_score_features \
  --replay runs/precision/score_features_final/frozen_broad_precision.json \
  --split confirmation --allow-confirmation \
  --output runs/precision/confirmation/score_broad_precision.jsonl
```

Replay needs only the standard library. It verifies provenance and outputs
`number`, `proposed`, and `artifact_sha256`. It never extracts expected labels
and never fits a model. Frozen development JSONL predictions are included and
replay byte-for-byte identically.

Python API: `load_frozen(path, protocol)`, followed by
`predict(scores_by_feature_label, frozen)` for labels, or
`predict_probabilities(scores_by_feature_label, frozen["model"])` for continuous
scores. For the final models, the required input features are all 25 labels.

To rerun training, use the pinned `experiments/requirements-text.txt`
environment and a **new** output directory:

```bash
python -m experiments.precision_score_features --output NEW_MODEL_RUN
python -m experiments.precision_score_features \
  --broaden NEW_MODEL_RUN --output NEW_FINAL_RUN
```

The second command changes decision policies only; no model is retrained. It
first reproduces the archived balanced development metrics exactly before
deriving new thresholds. Historical `runs/precision/score_features` artifacts
are preserved; their original module version is commit `ff34c43`.

## Cost limitations and remaining gains

The eight-feature experiment selects cached outputs originally produced with
all 25 questions. It does not prove an eight-question live run is equivalent or
68% faster. The prompt stream measured roughly 1.15-1.31 s means for altered
eight-question formulations; those are not measurements of an exact reduced
baseline question bank. Tiny batch-rounding changes could affect threshold
decisions. No new neural inference was run in this workstream.

The principal remaining opportunity is better area evidence, not another
global precision threshold or a larger sweep of these models. The sibling
text/hybrid experiments provide the more promising comparison. External,
adjudicated, distribution-shifted evaluation is still needed before treating
silver-label agreement as production correctness.
