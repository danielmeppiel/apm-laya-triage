# APM issue triage: a private, read-only Laya experiment

Can an open model reproduce the classification labels already on
`microsoft/apm` issues? This repository runs real local inference, measures
agreement and latency, and keeps the evidence needed to repeat the experiment.

**It never changes APM.** There is no command to apply labels, post comments,
accept work, prioritize issues, assign people, or edit milestones. "Labelling"
here means writing **predicted labels to local files**.

Read [REPORT.md](REPORT.md) for the plain-English results and examples. The
baseline covers **all 1,328 issues** in the frozen snapshot: **168 open and
1,160 closed**, excluding pull requests. The initial full-corpus run is in
progress; its completed evidence and report will be committed to `main`.

## Quick start

Requirements: Python **3.12**, Git, the GitHub CLI (`gh`), and enough memory for
the 421M-parameter model. Allow several GB of RAM and disk for the runtime,
approximately **843 MB** for model weights, and additional working memory.
The recorded run uses an Apple M3 with 16 GiB RAM, MPS in full precision; CPU also works but
is slower. **No TypeSafe key, Hugging Face key, or paid model API is required.**

```bash
git clone https://github.com/danielmeppiel/apm-laya-triage.git
cd apm-laya-triage
python3.12 -m venv .venv
source .venv/bin/activate

# Linux only: install the CPU build first to avoid CUDA downloads.
# python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The SDK is pinned to the official Laya v0.3.20 source commit. This also handles
package mirrors that have not published that version yet. Dependencies and
weights are not committed. The exact model revision and weights SHA-256 are
in `config.json`; inference refuses a checksum mismatch. Safetensors and
tokenizer/config data are downloaded, not Python code from the model repository.

### See the saved results without running a model

```bash
python experiment.py report
```

This rebuilds `REPORT.md`, `runs/baseline/metrics.json`, and
`runs/baseline/proposals.csv` from the complete saved predictions. It requires
no model download, GPU, Hugging Face account, or GitHub login. It refuses to
present a partial run as a full-corpus report.

### Try ten real predictions

```bash
python experiment.py run --output runs/my-smoke --limit 10 --device auto
```

This is real inference, not a fixture mode. `auto` selects CUDA, then Apple
MPS, otherwise CPU. Use `--device cpu` or `--device mps` to select explicitly.
Unavailable requested devices fail rather than silently changing the measured
hardware. A smoke run is intentionally insufficient for a full-corpus report.

### Repeat the whole experiment

```bash
# Reuse the exact frozen issue text and controls, but run the model again.
python experiment.py run --output runs/my-repeat --device auto
python experiment.py report --run runs/my-repeat --output REPORT-repeat.md
```

**Expect a long run, potentially hours** on a laptop or CPU runner: there are
25 label questions for every issue. Per-issue predictions are flushed to disk,
and progress is printed every 25 issues. Interrupt with Ctrl-C and rerun the
same command to resume. The run fingerprint prevents mixing different input
snapshots, code, rubrics, thresholds, model revisions, dependency versions, or
devices in one result set. If a forceful process termination leaves an
incomplete final JSONL line, the reader fails explicitly; inspect/remove only
that incomplete last line before resuming. Never silently discard valid rows.

Do not overwrite the supplied `runs/baseline/` when changing configuration.
Choose a new run directory. The original measured run used:

```bash
python experiment.py run --output runs/baseline --device mps
```

### Refresh the current GitHub issues and labels

Only this step requires a GitHub login with read access:

```bash
gh auth login
python experiment.py snapshot --output data/snapshot-new.json
python experiment.py run --snapshot data/snapshot-new.json \
  --output runs/new-snapshot --device auto
python experiment.py report --snapshot data/snapshot-new.json \
  --run runs/new-snapshot --output REPORT-new-snapshot.md
```

The source is configured as `microsoft/apm`. Snapshot collection follows every
page of the issues endpoint with `state=all`, excludes PRs, and freezes current
labels, descriptions, issue text, and the authoritative APM alias contract.
The GitHub boundary hardcodes **GET**. No issue text is executed as shell code.
API pagination is not a transaction; edits during collection may be observed.
The current label state is the control, not the label history at issue creation.

## What gets classified

The canonical APM contract supplies **25 labels**: eight `type/`, fourteen
`area/`, and three `theme/` labels. Legacy names such as `bug`, `enhancement`,
and `documentation` are normalized using that contract, not guessed.

The real controls contain multiple labels even in the type dimension. For
example, one issue can be both a bug fix and automation work. Consequently,
this experiment asks **25 independent Noul (yes/no membership) questions**,
not a single forced-choice question. Every question has the label's description
and an explicit relevant/not-relevant rubric.

```text
Frozen issue title/body
    -> bounded local tokenization
    -> actual Laya forward pass, all 25 questions together
    -> probabilities
    -> fixed 0.50 threshold
    -> predicted label sets
    -> compare with existing labels outside the model
    -> JSON, CSV and a plain-English report
```

The model sees title/body and a project description. It does **not** receive
the issue's existing API labels, state, number, expected answer, comments, or
source code. Label names can still occur naturally in the original issue text;
the report includes a sensitivity check excluding such cases.

Never infer `status/*`, `priority/*`, acceptance, invitations, assignments,
milestones, or processing markers from this experiment. Those labels are
excluded from both predictions and scoring.

## Configuration

Edit `config.json` before a new run:

| Setting | Meaning |
|---|---|
| `source_repo` | Frozen source is `microsoft/apm`; the contract path is APM-specific. |
| `model_repo`, `model_revision`, `weights_sha256` | Exact publicly downloadable checkpoint and integrity pin. Update together after review. |
| `device` | Default `cpu`. Override with the CLI for MPS/CUDA/automatic selection. |
| `mixed_precision` | Kept `false`: the MPS mixed-precision pilot returned invalid probabilities. |
| `cpu_threads` | Two, to bound CPU use on a shared machine. |
| `state_token_budget` | 300. Long reports keep the start and a smaller tail, with an explicit omission marker. |
| `binary_threshold` | 0.50 for all labels, fixed before looking at corpus results. |
| `project_context` | Short description of APM included with each issue. |

This English checkpoint has a short context window. The report counts all
shortened inputs and compares shortened versus intact cases. It is **not**
full-discussion triage, duplicate detection, source-code bug verification, or
a multilingual production system.

`HF_HOME` can point to an existing Hugging Face cache; otherwise its normal
user cache is used. Downloads are anonymous and implicit Hugging Face tokens
and telemetry are disabled. Issue text is processed locally, not submitted to
a hosted model API.

## How to read "success rate"

Existing labels are a **silver reference**, not guaranteed human ground truth.
Some are incomplete, stale, overlapping, or automated.

- **Exact agreement:** all proposed labels match in dimensions with controls.
- **Precision:** how many proposed labels were present in those controls.
- **Recall:** how many existing control labels were recovered.
- **F1:** combines precision and recall so "add every label" cannot look good.
- **Coverage:** how many issues/dimensions had labels we could compare against.

An issue with a type label but no area labels is scored for type only. An
unlabelled dimension is not a negative example. Within a labelled dimension,
extra predictions count as disagreements, which can underestimate useful
suggestions when controls are only partially annotated.

Every issue receives predictions, including issues without controls. Those
unlabelled issues are listed in the CSV but **not included in success-rate
denominators**. The report breaks down open/closed issues, context shortening,
individual labels, and a naive most-common-label reference.

This is a zero-shot exploratory baseline, not a trained or held-out accuracy
claim. No training or threshold fitting occurs. If you tune after reading these
results, reserve new, human-adjudicated issues for an honest subsequent test.

## Files

| File | Purpose |
|---|---|
| `data/snapshot.json` | Frozen issues, current labels, label descriptions and alias contract |
| `runs/baseline/manifest.json` | Runtime, source/config fingerprint, model revision and completion status |
| `runs/baseline/predictions.jsonl` | Actual probabilities, per-issue latency, input hashes and shortening metadata |
| `runs/baseline/metrics.json` | Machine-readable aggregate and per-label results |
| `runs/baseline/proposals.csv` | All issues, existing labels and proposed labels; no source issue writes |
| `REPORT.md` | Results for readers without an ML background, including real examples and caveats |
| `experiment.py` | Snapshot, inference and report command-line entrypoint |
| `evaluation.py` | Coverage-aware scoring and report generation |
| `tests/` | Small offline tests for controls, denominators, input isolation and fail-closed behavior |

The dataset is public issue content, but this repository and its artifacts are
private by default. Review snapshots for sensitive information before sharing
or publishing them. Do not put tokens or model weights in Git.

## Optional GitHub Actions rerun

The manual **Laya experiment** workflow defaults to a 25-issue inference smoke
run. Set `limit` to **0** for the whole corpus. Set `refresh_snapshot` to true
to fetch current public APM issues first. No custom model secret is required.
The built-in GitHub token is used only by the snapshot read step.

Hosted runs use **CPU**, have a six-hour timeout, and upload raw results,
the run manifest, snapshot, and report as private Actions artifacts. They do
not commit results or modify APM. Private-repository Actions minutes still
count against the account's quota; no hosted model API is being billed.
Local measured latency must not be presented as GitHub-hosted CPU latency.
The full-corpus run and a hosted smoke run are separate experiments: a
successful small Actions run does not demonstrate full-corpus hosted latency.

## Upstream sources

- [Laya model and license](https://huggingface.co/convaiinnovations/laya)
- [Pinned runtime source](https://github.com/NandhaKishorM/laya/tree/23a17522aa4942da6cce53a995a275760320b691)
- [APM governance](https://github.com/microsoft/apm/blob/main/GOVERNANCE.md)

This is an independent experiment, not an official Microsoft, TypeSafe AI, or
Laya endorsement. It is not a benchmark of Jev.
