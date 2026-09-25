# One authorized checkpoint follow-up

The specialist checkpoint gives a modest additional development improvement,
not an improvement comparable to its unrelated upstream benchmark claim.
Development is now closed. `FINALISTS.json` names exactly two matched-scope
finalists: base categorical type+theme and typed categorical type+theme.
The earlier base type-only frozen policy remains a diagnostic, not a third
confirmation candidate.

| Checkpoint / subset | Type+theme precision | Recall | Micro F1 |
|---|---:|---:|---:|
| Base / screen24 | .5862 | .4595 | .5152 |
| Typed / screen24 | .6250 | .5405 | .5797 |
| Base / development64 | .6310 | .5096 | .5638 |
| Typed / development64 | **.6364** | **.5385** | **.5833** |

Typed type P/R/F1 is .65625/.53165/.58741 (42 TP,22 FP,37 FN;64 controls).
Typed theme is .58333/.56000/.57143 (14 TP,10 FP,11 FN;24 controls).
Base theme precision was .65, so the checkpoint trades some theme precision
for coverage. Typed proposes a theme for63/64 issues versus42/64 for base.
Both propose a type for every issue; typed makes127 total proposals, of which88
fall in observed dimensions. Missing theme dimensions remain unscored.
None/other is explicit but rarely selected by the specialist.

Questions, option order, decision rule, and retained issue text are identical.
Exact baseline excerpt hashes are checked before every alternate-model forward.
The different checkpoint has1024 context /256 head capacity, versus512/192 for
base, but **no additional text** is supplied and neither checkpoint truncates
these questions. Both use the same pinned SDK and dependency versions.

## Authorization and provenance

The coordinator explicitly authorized only this checkpoint plus its tokenizer
and config files after the base experiment was frozen:

- Repository: `convaiinnovations/laya-typed-decisions`.
- Revision: `1a793eb568e6718f15941d08f85432581df534e3`.
- Safetensors SHA256: `4fa56de72383a9d3efa9cfa78955733c81b9fc8067a587ca4beb82c78107a24e`.
- Weight size:842,609,220 bytes, verified against the downloaded file.
- Parameters:421,293,830 stored as FP16; loaded and verified as FP32 for MPS.
- License: Apache-2.0. No remote Python, new dependency, or paid API.
- Architecture: ModernBERT-large,28 encoder layers,1024 hidden dimensions,
  two decision-head layers. The pinned SDK explicitly supports this checkpoint.
- Download and verification:99.29s, separately recorded in `typed-download.json`.
  Only the five files listed there were requested, with `token=False`.

All real inference remained serialized, offline after download, two CPU threads,
FP32 MPS, with synchronized timers and strict response validation. Screen24 was
promising, so only40 additional development rows were inferred; no other
checkpoint or prompt variant was tried. Typed64 real inference totaled24.19s,
median0.364s/issue for the two categorical questions. Two model-load segments took
6.78s and5.08s; invocation wall times19.08s and23.27s include setup. The second
invocation reused24 records, so its wall time excludes their original inference.
No inference warnings, invalid probabilities, device fallbacks, or errors occurred.

The load warning about clamped `choice:11+` remains irrelevant to active
nine-option and four-option questions; the runner verifies the active buckets
are unchanged. Checkpoint confidence should still not be interpreted as an APM
calibrated probability.

## Frozen replay

The typed runner is separate so the base runner's frozen source hashes remain
unchanged. It reuses base preparation, hashing, loader, response checks,
metric aggregation and protocol validation rather than altering base evidence.

```sh
# Existing cache/environment, same settings as REPORT.md.
$PY -m experiments.precision_typed_checkpoint run \
  --partition screen24 --output runs/precision/prompts/typed-screen
$PY -m experiments.precision_typed_checkpoint run \
  --partition development64 --reuse runs/precision/prompts/typed-screen \
  --output runs/precision/prompts/typed-development
$PY -m experiments.precision_typed_checkpoint summarize \
  --run runs/precision/prompts/typed-development
```

Frozen typed policy: `frozen-typed-choice-type-theme.json`,
`efa1c51da2cc0d50dadec46df397ba7089cc5ab83a4e79a658f91f45fc357d87`.
Frozen base policy: `frozen-choice-type-theme.json`,
`cba66c0922802eb3cc1826132a82b24fbe5b132665c844e4241a8eb1a7aff96d`.

Only after the parent's global freeze gate, it may authorize these commands:

```sh
$PY -m experiments.precision_prompts run --partition confirmation96 \
  --frozen runs/precision/prompts/frozen-choice-type-theme.json \
  --output runs/precision/prompts/confirmation-base-type-theme
$PY -m experiments.precision_typed_checkpoint run --partition confirmation96 \
  --frozen runs/precision/prompts/frozen-typed-choice-type-theme.json \
  --output runs/precision/prompts/confirmation-typed-type-theme
```

They were **not run during this development experiment**. Scoring confirmation
is deliberately unavailable in either runner; the parent owns outcome evaluation
and grouped uncertainty estimates. After the parent's committed global freeze
and explicit authorization, both prediction-only runs completed; their gate,
row counts, hashes and non-overlapping execution timestamps are preserved in
`confirmation-gate.json`. No confirmation scoring or tuning occurred here.
Upstream reports0.766 versus0.362 on four
synthetic workflows, explicitly warns that transfer elsewhere may not improve,
and notes temperature calibration problems. Those numbers are not APM evidence.

Metadata sources, read before the authorized download:

- <https://huggingface.co/api/models/convaiinnovations/laya-typed-decisions?blobs=true>
- <https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/1a793eb568e6718f15941d08f85432581df534e3/README.md>
- <https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/1a793eb568e6718f15941d08f85432581df534e3/rl_agent_config.json>
- <https://huggingface.co/convaiinnovations/laya-typed-decisions/blob/1a793eb568e6718f15941d08f85432581df534e3/encoder/config.json>
