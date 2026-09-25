# Precision follow-up: cheap text is the strongest default

**A small supervised text classifier is the most useful default found in
this experiment.** On 188 reserved confirmation issues, it reached **80.2%
precision, 74.6% recall and 77.3% F1**, with exact agreement on **53.2%**
of issues. Its warm prediction time was about **2.3 ms per issue** in the
development timing sample, without running Laya.

These are agreement scores against existing APM labels, **not verified
real-world correctness**. The classifier learned from other issues'
labels; this is not a zero-shot Laya result. It is a promising advisory
baseline, not authorization to apply labels automatically.

Adding Laya is not a clear default upgrade. The balanced hybrid recovered
more labels but had worse precision and no demonstrated F1 improvement
over text alone. At the more selective operating point, the hybrid
improved recall and F1, but its small precision advantage remained
uncertain. It also needs neural inference for new, uncached issues.

The historical [zero-shot baseline report](REPORT.md), saved predictions
and [runtime measurements](PERFORMANCE.md) remain unchanged.

## What the confirmation comparison actually showed

All eight rows below use the same **188 issues, 180 text-duplicate groups,
and 472 existing labels in observed dimensions**. These numbers are not
comparable to the historical 1,079-control totals without accounting for
the different evaluation population.

| Frozen method | Precision | Recall | F1 | Exact agreement | Issue coverage | Labels per issue |
|---|---:|---:|---:|---:|---:|---:|
| Original Laya, threshold 0.50 | 14.2% | 97.5% | 24.8% | 0.0% | 100.0% | 23.23 |
| Most-common fit label per dimension, no text reading | 53.0% | 42.8% | 47.4% | 22.3% | 100.0% | 3.00 |
| Supervised Laya-score-only, balanced | 81.9% | 66.1% | 73.2% | 44.1% | 96.8% | 2.95 |
| Supervised Laya-score-only, broad precision | 65.0% | 57.8% | 61.2% | 21.8% | 89.9% | 3.32 |
| **Supervised text, balanced** | **80.2%** | **74.6%** | **77.3%** | **53.2%** | **100.0%** | **3.31** |
| Supervised text, broad precision | 83.2% | 58.9% | 69.0% | 37.2% | 84.0% | 2.55 |
| Supervised text + Laya, balanced | 73.3% | 79.7% | 76.3% | 47.9% | 100.0% | 4.13 |
| Supervised text + Laya, broad precision | 84.8% | 62.5% | 72.0% | 43.1% | 86.7% | 2.57 |

**Precision** asks how many proposed labels appeared in the controls.
**Recall** asks how many control labels were recovered. F1 balances the
two. **Exact agreement** requires matching every label in the issue's
observed dimensions. **Issue coverage** means at least one prediction
in an observed dimension; it does not mean every dimension was covered.
Labels per issue counts all proposals, including unscored dimensions.

The score-only model learned relationships among the 25 saved Laya
outputs. It did not retrain Laya, and it is not zero-shot. Text uses
title/body word features; the hybrid combines those features with the
same saved outputs. The no-reading prior is learned on fit issues only.

Source: [all-taxonomy confirmation evidence](runs/precision/confirmation/all-taxonomy-vs-prior.json).
The full JSON includes per-label support, errors and per-dimension results.

### A precision claim must not hide abandoned dimensions

Early development policies exceeded 93% overall precision while recovering
only about 19% of area labels. Those were not accepted as the final broad
precision policies. The replacement selected a separate threshold for
each dimension, requiring development recall of at least 50% in each.
That constraint also held for the final text and hybrid broad policies
on confirmation; it is not a guarantee on future data.

| Dimension | Confirmation issues / positive labels | Text balanced P / R | Text broad P / R | Hybrid broad P / R |
|---|---:|---:|---:|---:|
| type | 184 / 226 | 86.6% / 88.5% | 97.7% / 57.1% | 98.7% / 66.8% |
| area | 127 / 175 | 74.4% / 54.9% | 70.2% / 56.6% | 71.8% / 58.3% |
| theme | 70 / 71 | 70.9% / 78.9% | 82.0% / 70.4% | 79.2% / 59.2% |

Area remains the weakest dimension. Some rare labels are still missed
entirely: text balanced recovered none of the two `area/content-security`,
five `area/distribution`, four `area/enterprise`, or one `area/mcp-trust`
controls. There were no `type/release` positives in confirmation, so its
recall is undefined, not perfect.

## How convincing is the improvement?

The parent compared fixed predictions using **4,000 paired bootstrap
resamples of the 180 duplicate groups**, rather than treating every label
as independent. All methods used identical issue IDs and controls.

Against the no-reading prior, text balanced improved precision by
**27.2 percentage points** and F1 by **29.9 points**. The 95% intervals
adjusted across the seven comparisons to that prior were **+18.6 to +38.6
points** for precision and **+21.8 to +40.3** for F1. This is a clear
agreement improvement over that reference.

The more relevant cost question is whether Laya adds enough to text:

| Direct paired comparison | Precision difference, 95% interval | Recall difference, 95% interval | F1 difference, 95% interval |
|---|---:|---:|---:|
| Hybrid minus text, balanced policies | -6.89 points [-9.38, -4.48] | +5.08 [+2.91, +7.41] | -0.93 [-2.72, +0.94] |
| Hybrid minus text, broad policies | +1.54 points [-0.71, +3.78] | +3.60 [+0.64, +6.51] | +2.97 [+0.48, +5.44] |

At the balanced setting, hybrid precision is clearly lower and the F1
interval includes no change. At the broad setting, recall and F1 improve
in this paired comparison, but the precision interval includes no gain.
These compare separately development-selected operating points, not
identical thresholds or perfectly matched recall.

The direct-pair reports each contain one comparison. The prior comparison
uses its stated seven-candidate adjustment; none of these intervals is a
blanket correction over every metric and every exploratory analysis.
Group resampling also cannot remove all topic, template or time dependence.

Evidence: [balanced hybrid vs text](runs/precision/confirmation/hybrid-vs-text-balanced.json)
and [broad hybrid vs text](runs/precision/confirmation/hybrid-vs-text-broad.json).

## What was tried, and why we stopped

The work proceeded in small CPU/cached-score steps, with separate prompt
experiments. Model parameters and preprocessing were learned on **677 fit
controls**; finite model choices and thresholds used **214 development
controls**. The final registry was sealed before evaluating the **188
confirmation controls**. No policies were retuned on confirmation.

The main development findings were:

- Changing raw Laya cutoffs or picking its highest raw scores was not
  enough. Per-label isotonic calibration helped ranking, but the joint
  25-score logistic model was stronger.
- Cheap word TF-IDF plus one-vs-rest logistic regression provided the
  largest useful gain. Weighting the title three times added only about
  0.001 F1. Five-neighbor text retrieval did not beat logistic regression.
- Adding cached scores produced a small development gain; a final reduced
  score weight was frozen. The balanced hybrid gain did **not** become
  an established confirmation F1 advantage.
- A conservative ExtraTrees score model did not beat the score-only
  logistic model. Using only eight type-question scores reduced
  development F1 from about 0.679 to 0.602. Those variants were not
  promoted to more confirmation candidates.
- Removing leading title-template markers such as `[BUG]`, `[FEATURE]`
  and `[DOCS]` changed text development F1 from 0.750 to 0.747.
  Giving text only Laya's exact retained input excerpts changed it to
  0.744. The main gain therefore was not explained by trivial title tags
  or access to much longer bodies in these diagnostics.

The excerpt diagnostic used tokenizer-only exports for fit/development,
not new inference or confirmation data. It removed only the fixed project
prefix after the original 300-token head/tail truncation and added no
freed-budget text. The same lexical configuration was refit on those
excerpts. These diagnostic variants did not change the frozen finalists.

This was a bounded experiment, not a search for the best possible model.
Once the simple alternatives, useful improvements and requested robustness
checks were complete, the registry was frozen instead of expanding the
sweep. Further tuning would require a new evaluation plan and genuinely
new adjudicated data.

Detailed development evidence and configurations:
[text/hybrid research record](runs/precision/text/README.md).

### Zero-shot prompt and alternate-checkpoint follow-up

There was also a genuine improvement without training an APM classifier:
ask categorical questions for **one primary type and one primary theme**,
with explicit none/other options, rather than 25 binary membership
questions. Removing context alone had not helped in the initial screen;
categorical choice did. One additional, preselected checkpoint,
`convaiinnovations/laya-typed-decisions`, was then compared with the base
`convaiinnovations/laya` checkpoint using identical questions and retained
issue text.

Both frozen categorical methods were run on the same **96-issue subset**,
with **95 eligible type/theme controls and 91 duplicate groups**. The text
reference below is evaluated on exactly that subset and scope too.
**These are not all-taxonomy results: neither categorical method predicts
area labels.**

| Method, type + theme only | Precision | Recall | F1 | Exact agreement |
|---|---:|---:|---:|---:|
| Fit-only most-common reference | 54.2% | 45.5% | 49.5% | 36.8% |
| Original binary Laya, restricted to this scope | 19.5% | 98.1% | 32.6% | 0.0% |
| Base-checkpoint categorical | 74.4% | 57.7% | 65.0% | 45.3% |
| Typed-checkpoint categorical | 79.4% | 66.7% | 72.5% | 49.5% |
| Supervised text balanced, matched subset/scope | 85.1% | 87.8% | 86.4% | 72.6% |

Both categorical methods beat the prior on precision and F1 with the
report's four-comparison-adjusted intervals. Typed versus base categorical
improved F1 by **7.49 points**, with paired 95% interval **+1.23 to +13.66**.
Its precision improvement was not established: the interval was
**-2.22 to +12.15 points**. Supervised text still had the higher observed
scores on this matched scope; its training advantage must not be hidden.

The checkpoints were not fine-tuned on APM labels and no supervised APM
classifier sits behind these categorical predictions. However, the prompt
and checkpoint were selected using development results. "Zero-shot" here
does not mean an untouched, unselected or pristine evaluation.
Categorical top-one output also cannot express every multilabel control.

Median synchronized MPS inference was **0.379 seconds** for base and
**0.410 seconds** for typed, for **two questions only**. The complete
96-issue fresh-process invocations took **44.44 / 47.49 seconds**,
respectively; those are batch invocation times, not single-issue cold
latencies. Fewer questions and a narrower output contract explain why
these timings cannot replace the full 25-label latency.

Sources: [matched categorical comparison](runs/precision/confirmation/categorical-vs-prior.json),
[typed versus base](runs/precision/confirmation/typed-vs-base.json),
[prompt development record](runs/precision/prompts/REPORT.md), and
[checkpoint provenance](runs/precision/prompts/TYPED-CHECKPOINT.md).

## Four real examples, not adjudications

These cases were selected from the saved confirmation predictions to
illustrate behavior, not to estimate performance. Labels below are the
frozen snapshot controls, which can themselves be incomplete or debatable.

| Frozen issue | Existing controls | Text-balanced behavior | What it illustrates |
|---|---|---|---|
| [microsoft/apm#249](https://github.com/microsoft/apm/issues/249), CLI consistency report | `area/cli`, `type/automation`, `type/docs` | Recovered all three; also proposed `theme/portability`, which was unscored because theme had no controls | Original Laya proposed all 25 labels. Text greatly reduced over-labelling, but an exact observed-dimension match is not proof that every emitted label is correct. |
| [microsoft/apm#844](https://github.com/microsoft/apm/issues/844), Cursor receives Copilot's MCP schema | `area/mcp-config`, `theme/portability`, `type/bug` | Proposed exactly those three; original Laya proposed all 25 | A concrete cross-client configuration failure where text matched every observed dimension. |
| [microsoft/apm#248](https://github.com/microsoft/apm/issues/248), `apm init` differs from its documented structure | `type/docs` | Proposed `type/bug`, missing the docs control; area/theme proposals were unscored | A false positive and miss against the controls. The body itself says "Describe the bug", so the disagreement needs human interpretation rather than assuming either side is truth. |
| [microsoft/apm#650](https://github.com/microsoft/apm/issues/650), Copilot CLI user-scope instructions | `area/cli`, `area/docs-site`, `area/multi-target`, `theme/portability`, `type/feature` | Kept CLI, portability and feature; missed docs-site and multi-target | A single issue spans installation behavior, documentation and target compatibility. Good headline precision can still miss useful secondary area labels. |

## Cost: milliseconds do not include a hidden neural call

Text title-3's development sample had a **2.32 ms median warm single-issue
prediction**, with about **0.93 seconds to fit** the small model. The final
hybrid's classical portion was **2.60 ms**, with about **1.09 seconds to
fit**. Both used one CPU thread and a separate pinned experiment environment.

Those are warm in-process measurements over the first 16 development
issues, not fresh-process p95s or production service promises. The
confirmation replay took about **1.80 seconds for the entire 188-issue text
batch in a fresh process, including refitting**. That is a different
measurement, not 1.80 seconds per prediction.

**Every fresh text CLI invocation refits the frozen 677-control model
and pays import/setup costs.** Warm millisecond latency requires keeping
the fitted model in memory. The separately recorded live CLI example took
**3.84 seconds for its fresh process, excluding the GitHub fetch**; it is
not evidence for a 2 ms cold invocation.

For score-only and hybrid methods, saved Laya outputs were reused. Their
classical timings **exclude neural score generation**. The historical
development rows needed a median **7.58 seconds per issue** for the
25-question MPS Laya inference; hosted CPU measurements are slower in
[PERFORMANCE.md](PERFORMANCE.md). These are not matched-hardware timing
trials, but adding Laya certainly cannot be represented as a free 0.28 ms
increment for a new uncached issue.

## Try text-only suggestions on an issue

This route needs Python 3.12, the small experiment dependencies and `gh`
for the read-only fetch. **It needs no Torch, model weights, GPU or paid
API.** Run from the repository root:

```bash
python3.12 -m venv /tmp/apm-text-env
/tmp/apm-text-env/bin/python -m pip install -r experiments/requirements-text.txt
gh api repos/microsoft/apm/issues/3071 \
  --jq '{number,title,body:(.body // "")}|@json' > /tmp/apm-issue.jsonl
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/apm-text-env/bin/python -m experiments.precision_text \
  --predict-frozen runs/precision/text/frozen/manifest.json \
  --finalist text_balanced --inputs /tmp/apm-issue.jsonl \
  --output /tmp/apm-proposals.jsonl
```

`@json` produces one JSONL record, including when the issue body contains
newlines. Only number/title/body are retained; the live labels are not
prediction features. Replace the issue number as needed and choose a new
output path if it already exists. Text-only prediction accepts arbitrary
issue IDs and does not require a Laya score cache entry for that issue.
It still learns from the same frozen fit controls, not from the fetched
issue's current labels.

The parent exercised this command on
[microsoft/apm#3071](https://github.com/microsoft/apm/issues/3071), producing
`theme/portability` and `type/bug`, with no GitHub writes. The
[saved receipt](runs/precision/live-example/receipt.json) records the
3.84-second fresh invocation. This demonstrates usability, **not accuracy**:
the example may overlap fit/development data. Suggestions stay in a local
file; this command does not apply them to the source issue.

## Reproduce the evidence without calling a model

From the integrated repository, rebuild the saved confirmation comparison
using Python's standard library:

```bash
python3 -m experiments.precision_confirmation \
  --plan runs/precision/confirmation/all-taxonomy-vs-prior-plan.json \
  --output /tmp/apm-precision-confirmation-replay.json
```

For the direct comparisons, use `hybrid-vs-text-balanced-plan.json` or
`hybrid-vs-text-broad-plan.json` in the same directory and a different
output path. `categorical-vs-prior-plan.json` and `typed-vs-base-plan.json`
replay the narrower type/theme comparisons. The evaluator checks snapshot,
protocol, source/artifact and prediction hashes before computing metrics.
It does not train models,
perform inference, contact GitHub, or modify the saved evidence.

To deterministically refit the frozen **text-only** model and replay its
stored confirmation inputs, use a separate environment:

```bash
python3.12 -m venv /tmp/apm-text-env
/tmp/apm-text-env/bin/python -m pip install -r experiments/requirements-text.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/tmp/apm-text-env/bin/python -m experiments.precision_text \
  --predict-frozen runs/precision/text/frozen/manifest.json \
  --finalist text_balanced \
  --inputs runs/precision/confirmation/inputs.jsonl \
  --output /tmp/apm-text-balanced-replay.jsonl
```

This fits only the 677 fit controls, verifies pinned code/dependencies,
and accepts only number/title/body for prediction. Issue numbers are join
keys, not model features. It does not learn from confirmation labels.
Other frozen keys are `text_broad_precision`, `balanced` (hybrid), and
`broad_precision` (hybrid); use fresh output paths. Hybrid replay reads
existing score caches and does not generate scores for unseen live issues.

The [global registry](runs/precision/confirmation/registry.json) records
the pre-confirmation candidate freeze. The
[text/hybrid manifest](runs/precision/text/frozen/manifest.json) pins
configuration and source hashes. Per-issue predictions, evaluation plans,
support tables and paired intervals are under
[`runs/precision/confirmation/`](runs/precision/confirmation/).

## What this still does not establish

The controls are current labels with known legacy aliases, not a human
adjudication of every applicable label. Missing dimensions are excluded
from fitting and scoring, never treated as whole-dimension negatives.
Inside an observed dimension, an additional useful but unrecorded label
still counts as a disagreement.

The corpus contains recurring reports and potentially automated labels.
Exact/near-duplicate grouping reduces direct overlap but cannot ensure
independent topics or remove every template relationship. Historical
all-corpus baseline outcomes had already been inspected: confirmation
was reserved from these **new experiments**, not a pristine unseen
external test set. Results do not establish transfer to another repository
or later issues, and the classifier scores are not validated correctness
probabilities.

**Recommendation:** start with the frozen text-balanced model for local,
human-reviewed suggestions. It had the best observed all-taxonomy F1 and
exact agreement, roughly 80% agreement precision, and no per-issue neural
cost. Consider the broad hybrid only when its additional recall/F1 is
valuable enough to justify generating Laya scores; this experiment does
not establish a precision advantage over the matched text policy. Before
automatic application, collect new, independently adjudicated issues and
evaluate the operational costs of false additions and missed labels.

Nothing here authorizes status, priority, acceptance, assignments or
milestone decisions, and no source APM issues were changed.
