# Frozen base-checkpoint formulation experiments

## Result and scope

Categorical choice is a useful zero-shot formulation improvement on the shared
development subset. Context removal alone is not. These are silver-label
agreement measurements, not gold correctness estimates or confirmation results.
No issue was modified, no paid API was called, and no confirmation outcome was
read. The base checkpoint was already cached; these runs downloaded no weights.

| Shared development64 policy | Scope | Precision | Recall | Micro F1 |
|---|---|---:|---:|---:|
| Original binary, threshold 0.5 | type | .1641 | .9494 | .2799 |
| Most-common type from fit only | type | .3438 | .2785 | .3077 |
| Original binary scores, raw top1 | type | .3281 | .2658 | .2937 |
| New categorical top1 | type | **.6250** | **.5063** | **.5594** |
| New categorical fixed top2 diagnostic | type | .4262 | .6582 | .5174 |
| Original binary, threshold 0.5 | type + theme | .1853 | .9231 | .3087 |
| New categorical top1 per dimension | type + theme | **.6310** | **.5096** | **.5638** |

Type+theme has type P/R/F1 .6250/.5063/.5594 (64 control issues) and
theme .6500/.5200/.5778 (24 control issues). Type coverage is 100%; theme
proposal coverage is 65.625% across all 64 issues; combined issue coverage is
100%. Combined average proposals are 1.65625, not 25-label full-taxonomy output.
Missing control dimensions are unscored. The combined policy makes 106 total
proposals; 84 are in observed dimensions, with 53 true positives.

The calibration sibling independently reports fit-only isotonic top1 on these
same 64 type controls: P .578125, R .468354, F1 .517483 (37 TP).
Our categorical top1 has 40 TP. This is a modest advantage over a stronger,
supervised comparator, not a claim of dominance over the lexical/hybrid models.
The new prompt uses no fit labels. The lexical sibling's exact-excerpt diagnostic
also retained strong F1, so the short input budget alone does not explain the
neural gap.

## Controlled initial screen

| Shared screen24 arm | Type precision | Type recall | Type F1 | Mean proposals |
|---|---:|---:|---:|---:|
| Cached original | .1637 | .9655 | .2800 | 7.125 |
| Membership, remove context | .1618 | .9655 | .2772 | 7.208 |
| Concrete Noul, keep context | .2222 | .4828 | .3043 | 2.625 |
| Concrete Noul, remove context | .2500 | .4483 | .3210 | 2.167 |
| Categorical type8 + none | .5417 | .4483 | .4906 | 1.000 |

The three binary arms stopped after screen24. Choice alone expanded to64;
the categorical type+theme extension used the same24 before expanding to64.
Screen24 is nested in development64, so this is selection, not independent
validation. Fixed top2 was a finite postprocessing diagnostic, not a learned
cutoff: it improves recall but loses F1 and precision, so it is not a finalist.

Every arm starts from the exact original `prepare_state` output under its
300-token head/tail budget. Clean arms strip only the Project prefix *after*
shortening, preserving identical issue text. Freed space is never refilled.
The comparator's state/input hashes are checked. All 64 issue states matched;
60 were shortened. No controls or source label-presence metadata enter inference.

The original question asks abstract label membership with generic true/false
criteria. Concrete arms use a direct per-label predicate and omit criteria,
matching the pinned upstream SDK's simple Noul examples. Choice changes the task
to primary classification, allowing at most one type and one theme. It must not
be described as equivalent multi-label inference. Type has9 options and theme4,
including explicit none/other; area14 was not attempted.

## Runtime, provenance, cost

Pinned model/runtime/config/questions and code hashes are in every manifest.
The model is `convaiinnovations/laya` revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`; SDK source is
`23a17522aa4942da6cce53a995a275760320b691`.
All real runs used Apple MPS, FP32 weights/operations, two CPU threads, synchronized
timers, exclusive file locking, offline cache loading, strict probability/device
checks, and fsynced per-issue output. No prediction warnings or errors occurred.

The SDK emits a load warning for its unused `choice:11+` temperature bucket.
The actual active buckets (`choice:6-10`, `choice:3-5`, `noul:2`) are explicitly
checked for clamping and recorded. No instruction or option exceeded the SDK
head or per-option budgets; no hidden question truncation occurred.

Median inference: type8 binary about1.15-1.25s; categorical type about0.202s;
categorical type+theme about0.382s. Type64 total recorded inference16.27s,
type+theme64 24.68s, including reused screen rows and first-call overhead.
Development manifest wall segments cover only that invocation's40 new rows and
model setup, not the reused24. Across seven invocations, 200 unique issue/arm
calls and768 question rows were actually inferred, about130.15s synchronized
inference and192.32s harness wall time. No model ran concurrently.

Adding theme to the same batch preserved every type decision on64 issues;
maximum probability difference was0.0001 from batching/rounding. This does not
prove exact equivalence between an eight-question and25-question baseline run.

## Replay and frozen policies

Run from repository root using the existing pinned environment:

```sh
export HF_HOME=/path/to/existing/laya-cache
export USE_TF=0 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_TELEMETRY=1
PY=/path/to/laya-venv/bin/python

$PY -m experiments.precision_prompts run --partition screen24 \
  --variant choice-clean --dimensions type,theme \
  --output runs/precision/prompts/screen-choice-type-theme
$PY -m experiments.precision_prompts run --partition development64 \
  --variant choice-clean --dimensions type,theme \
  --reuse runs/precision/prompts/screen-choice-type-theme \
  --output runs/precision/prompts/development-choice-type-theme
$PY -m experiments.precision_prompts summarize \
  --run runs/precision/prompts/development-choice-type-theme
```

Completed identical runs are verified and returned without loading a model.
Change output directories to reproduce fresh inference. Frozen policies:

- `frozen-choice-type.json`: `47490874c502cb3f408e4244a9a1c341e50385b9b19b705f8245cfc631070a20`
- `frozen-choice-type-theme.json`: `cba66c0922802eb3cc1826132a82b24fbe5b132665c844e4241a8eb1a7aff96d`

The parent can authorize `run --partition confirmation96 --frozen <policy>
--output <new-directory>` after its explicit final gate. Subsequent authorized
prediction-only runs are recorded in `confirmation-gate.json`; neither was
scored in this stream. The summarizer refuses confirmation. Each JSONL row includes
`number`, `proposed`, `scores`, full response, exact state, hashes and timings.
The historic baseline saw all issues; confirmation is reserved from *new*
experiments only. No population-level success claim is warranted before it.

Official formulation reference:
<https://github.com/NandhaKishorM/laya/blob/23a17522aa4942da6cce53a995a275760320b691/README.md>.
