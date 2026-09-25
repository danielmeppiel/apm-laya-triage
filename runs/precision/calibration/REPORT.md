# Saved-score calibration: development evidence

These are **silver-label agreement** results, not true precision against expert
gold labels. Confirmation has not been evaluated by this workstream. Previous
all-corpus results were already inspected; this protocol reserves confirmation
only from new experiments.

The frozen text-grouped protocol assigns 677 controls to fit, 214 to development,
and 188 to parent-owned confirmation. It records 249 unlabelled issues excluded
from fitting and scoring. Entire missing dimensions are unscored, not negatives.
Every prediction consumes only cached baseline probabilities; expected labels
and dimension-presence metadata never enter prediction inputs.

## Sequential bounded experiments

Round 1 tested eight configurations: original threshold, fit-majority reference,
raw top-1/top-2 per dimension, and pooled/per-label fit-optimal F1/F0.5 thresholds.
Round 2 added dimension-specific thresholds and six fixed isotonic floor/cap
configurations. Round 3 added three conditional-secondary-label configurations
and two fit-mean-cardinality variants. The 21 total configurations and all their
fit/development, per-dimension, per-label, support, exact-agreement, cardinality,
and coverage metrics are persisted in `round1.json` through `round3.json`.
Development results are selection-biased.

| Configuration | Precision | Recall | F1 | Exact | Issue coverage | Observed-dimension coverage | Mean labels |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original 0.5 | .1476 | .9628 | .2560 | .0047 | .9953 | .9933 | 23.02 |
| Fit-majority per dimension | .4911 | .3894 | .4344 | .2196 | 1.0000 | 1.0000 | 3.00 |
| Raw top-2 per dimension | .3415 | .5416 | .4189 | .0234 | 1.0000 | 1.0000 | 6.00 |
| Per-label fit F1 thresholds | .2890 | .6991 | .4089 | .0794 | .9860 | .9062 | 9.46 |
| Isotonic top-1: balanced finalist | .6071 | .4814 | .5370 | .2944 | 1.0000 | 1.0000 | 3.00 |
| Isotonic top-1 plus second at .25: recall-constrained finalist | .5064 | .5628 | .5331 | .1449 | 1.0000 | .9978 | 4.26 |
| Isotonic floor .5, cap 1: rejected low recall | .7650 | .2938 | .4246 | .1402 | .9907 | .4844 | 1.67 |

The balanced finalist gains 10.26 percentage points of development F1 over
fit-majority, but **does not meet recall >= .50**. The other finalist meets that
constraint, yet its precision is only 1.53 points above fit-majority: this is
not a confirmed precision win. Parent-owned paired group confirmation must
determine uncertainty and any claim of a reliable improvement.

## What helped and what did not

Raw probabilities have strong label-specific offsets. Per-label monotone
calibration makes cross-label ranking more useful than either raw top-k or
independently maximizing fit F-beta. Gains are concentrated in type: balanced
type P/R/F1 is .6651/.5465/.6000, compared with .4340/.3566/.3915 for fit-majority.
Area and theme gains are small. Full per-dimension supports are in every report.

Pooled and dimension thresholds failed to outperform fit-majority F1.
Independent label thresholds still overpredict. High calibrated floors raise
precision by discarding too much recall. Allowing a second label at .4 or .5
still fails the recall constraint; .25 helps recall but reduces precision.
Rounding fit mean cardinality reproduces top-1; ceiling it produces six labels
and lower F1. Further scalar sweeps are unlikely to change the recommendation
substantially; multivariate score interactions or supervised text features are
the remaining plausible larger gains, handled separately.

## Fit and replay contract

Per-label isotonic regressions use pooled-adjacent-violators on tied raw score
blocks in **fit only**, conditional on that dimension being observed. There is
no shrinkage and no cross-label pooling. Out-of-range scores use endpoint
probabilities; gaps use midpoint step boundaries. Labels with fewer than five
fit positives are suppressed. On this corpus only `type/release` is suppressed
(one fit positive); all other labels have at least eight fit positives.

The balanced finalist always proposes the highest calibrated supported label
per dimension. The recall-constrained finalist admits up to two labels per
dimension at calibrated probability >= .25 and retains the top positive
calibrated supported label if the floor selects none. It can abstain on an
unsupported dimension or one with no positive calibrated probability.

```bash
# Rerun finite development experiments; no inference and no confirmation.
python3 -m experiments.precision_calibration --round 3 --freeze

# Parent only, after the confirmation plan and all finalists are frozen:
python3 -m experiments.precision_calibration \
  --replay runs/precision/calibration/frozen_balanced.json \
  --split confirmation --allow-confirmation \
  --output runs/precision/confirmation/calibration_balanced.jsonl
```

Use `frozen_precision_first_recall050.json` for the recall-constrained finalist,
`reference_original_050.json` for the original baseline, and
`reference_most_common.json` for the fit-majority reference. The same command
supports `--split confirmation96` if the parent selects the shared subset
instead of full confirmation.

The frozen files record source byte hashes, protocol hash, exact fit/development
IDs, module hash, policy configuration, development metrics, source report hash,
and a self-verifying artifact hash. Changed policies cannot overwrite frozen
files. CLI replay verifies these identities, requires an explicit confirmation
flag, and writes only `number`, `proposed`, and `artifact_sha256` per JSONL row.
It never extracts expected labels.

Python API:

```python
from experiments.precision_calibration import load_frozen, metrics, predict

frozen = load_frozen(policy_path, protocol)
proposed = predict(scores_by_label, frozen["policy"])
# Expected labels are joined separately, only in the evaluation layer.
agreement = metrics(evaluation_records, sorted(protocol["taxonomy"]))
```

`evaluation_records` have `expected` and `proposed` label lists. Metrics reuse
`evaluation.aggregate`, `scored_sets`, and `label_metrics`. Coverage distinguishes
any proposal on the issue from proposals in its observed dimensions, preventing
unsupported-dimension proposals from disguising abstention.
