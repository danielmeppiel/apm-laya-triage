# Single-issue performance: measured, not estimated

Production triage handles one issue at a time. Whole-corpus batches are used only to measure quality and speed on enough cases. Every call retains all 25 label questions.

## Matched-hardware configuration pilot

Each of two runners executed all four configurations, in opposite orders. Six fixed issues were each repeated three times. That means six distinct accuracy cases, not 36 independent ones. These pilot timings are not a production p95 claim.
INT8 converts only encoder Linear layers; the decision head stays FP32.

| Configuration | Runner 0 median | Runner 1 median | Changed issues vs FP32/2, runner 0 / 1 | Maximum probability change |
|---|---:|---:|---:|---:|
| fp32-2 | 44.442s | 44.355s | 0/6 / 0/6 | 0.0000 |
| fp32-4 | 45.905s | 45.217s | 0/6 / 0/6 | 0.0000 |
| int8-2 | 29.369s | 29.046s | 1/6 / 1/6 | 0.3950 |
| int8-4 | 28.629s | 28.102s | 1/6 / 1/6 | 0.3950 |

The two speed-pilot machines were standard hosted runners; their exact CPU topology is saved in each replicate's cpu.txt. Compare modes within a runner: a four-thread result on another VM is not evidence that adding threads caused the difference. The separate one-issue dispatches below illustrate that variability.

Model loading and INT8 conversion are separate costs, recorded in summary.json. The pilot reuses downloaded weights on each runner after its first configuration. Do not present warm-loop timings as cold workflow latency.

## Actual one-issue workflow runs

| Run | Mode / threads | Runtime cache hit | Inference | Invocation including fetch and loading | Job | Dispatch to completion |
|---|---|---|---:|---:|---:|---:|
| [single-cold](https://github.com/danielmeppiel/apm-laya-triage/actions/runs/36119730300) | fp32 / 2 | False | 44.013s | 54.622s | 112s | 126s |
| [single-warm](https://github.com/danielmeppiel/apm-laya-triage/actions/runs/36120201858) | fp32 / 2 | True | 43.343s | 61.869s | 76s | 83s |
| [single-fp32-four](https://github.com/danielmeppiel/apm-laya-triage/actions/runs/36122096140) | fp32 / 4 | True | 24.596s | 39.533s | 51s | 54s |
| [single-int8](https://github.com/danielmeppiel/apm-laya-triage/actions/runs/36121729119) | int8 / 4 | True | 28.287s | 43.130s | 58s | 63s |

These are individual observed runs, not latency guarantees. Dispatch time includes queueing; job time includes setup, caching and artifact upload. A cache miss installs dependencies, while a hit restores the exact pinned virtualenv.

### Local GPU on the identical issue

| Hardware / device | Precision | Inference | Invocation | Fresh process total |
|---|---|---:|---:|---:|
| Apple M3 / mps | fp32 | 6.938s | 18.345s | 19.279s |

Local measurements use already-downloaded model weights and no competing bulk GPU worker. GPU calls synchronize before/after inference. A fresh process still imports dependencies, verifies weights and loads the model. This is an Apple GPU result, not a measurement of an NVIDIA CUDA fast path.

## Accuracy check on the full frozen control group

Both configurations classified all **1328 issues** against the same **1079 eligible reference issues**. Unlabelled dimensions are unscored; current labels are a silver reference, not guaranteed truth.

| Runtime | Exact agreement | Precision | Recall | F1 | Corpus median inference |
|---|---:|---:|---:|---:|---:|
| FP32 reference | 0.1% | 14.6% | 97.0% | 25.3% | 43.495s |
| Candidate | 0.0% | 14.1% | 99.1% | 24.6% | 16.034s |

The candidate changed the full proposed label set on **569/1328 issues** relative to FP32. This is output drift, not automatically an error: use the control metrics to distinguish improvement from disagreement.
No decision threshold was fitted to these results. Rare labels remain under-supported, and choosing a runtime from this experiment does not turn the same corpus into an untouched future test set.
The full-corpus CPU runs used different pools of hosted machines. Their median-time ratio is not a controlled optimization speedup; use the matched-hardware pilot above for that comparison.

Candidate F1 differs from FP32 by **-0.68 percentage points**. This is measured reference agreement, not proof of true accuracy or calibrated probabilities. Faster INT8 is explicitly selectable, not silently substituted for the FP32 default.

## Complete GPU-versus-CPU replication

The GPU and hosted FP32 runs independently classified all **1328 issues**. Prepared input hashes matched, and **1328/1328 full proposed label sets were identical**. This confirms cross-device reproducibility of this classifier, not its correctness.

| Device | Full-corpus median inference | p95 |
|---|---:|---:|
| Local mps GPU | 7.548s | 8.370s |
| Hosted FP32 CPU pool | 43.495s | 54.238s |

Maximum reported probability difference: **0.0001**. The laptop is a shared workstation, not an isolated benchmark appliance.
