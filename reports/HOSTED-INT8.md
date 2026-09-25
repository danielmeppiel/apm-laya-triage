# APM issue labelling experiment: the plain-English report

## The short answer

We ran **real GitHub Actions CPU Laya inference on all 1,328 issues**, including 168 open and 1,160 closed issues. Pull requests were excluded.
**No APM issue, label, comment, status, assignment, or milestone was changed.**

On the 1,079 issues with usable existing classification labels, the model matched every known dimension exactly on **0.0%** (0/1,079).
The 95% Wilson interval for exact agreement is **0.0% to 0.4%**.
Label precision was **14.1%**, recall **99.1%**, and the combined F1 score **24.6%**.
Median model time was **16.034 seconds per issue** for all 25 label decisions together.

**This measures agreement with today's labels, not whether the model understood or fixed a bug.** The controls may be incomplete, debatable, or assigned by automation. Do not read this as a production accuracy guarantee or permission to auto-label.
**Engineering verdict: inference and the workflow work; this classifier does not demonstrate useful triage quality.** Its F1 is below the no-reading, most-common-label reference.
The model proposed an average of **24.64 of 25 labels per issue**, including every allowed label on **1094 issues**. Proposing almost everything can produce impressive recall while being unusable for labelling; precision and exact agreement expose that failure.

## What we asked the model to do

- Read the issue title and body, plus a short description of APM.
- Answer one yes/no membership question for every allowed classification label.
- Return a probability for each label. Propose it when that probability is at least 0.50. Multiple types, areas and themes can apply.
- Write the proposals to local JSON/CSV files. Never call a GitHub write API.

The model never received the issue's current API label list, closed/open state, number, expected answers, comments, or repository source code. The label names and their definitions are the shared classification rubric, not per-issue answers.
All labels were evaluated with the same fixed threshold. There was no training, fine-tuning, threshold fitting, or post-result prompt selection for this baseline.

## What the numbers mean

| Measure | In ordinary language |
|---|---|
| Exact agreement | Did the entire proposed label set match, in every dimension with a control? |
| Precision | Of the labels proposed in scorable dimensions, what share already appeared in the controls? |
| Recall | Of the control labels, what share did the model recover? |
| F1 | A combined precision/recall score; high recall alone is not enough if many extra labels are proposed. |
| Unscored | We generated predictions, but there are no relevant existing labels to check them against. |

There are **249 unscored issues**. Their predictions are included in the output, but excluded from success-rate denominators.
If an issue has a type label but no area labels, we score its type only. Missing an entire dimension is **not** treated as proof that every label in it should be absent. Within an observed dimension, additional proposed labels count as disagreements; some could be reasonable labels that maintainers never added.
The intervals assume independent, representative issues. They do not correct wrong reference labels, correlated issue clusters, or future topic drift; this frozen corpus is not a random sample of every future issue.
Labels with fewer than 20 positive controls are flagged as low-support, not treated as proven classes: `area/mcp-trust`, `type/release`. Per-label precision/recall intervals are included in metrics.json.

## Results by kind of label

| Dimension | Control issues | Exact agreement | Precision | Recall | F1 | Controls with multiple labels |
|---|---:|---:|---:|---:|---:|---:|
| type | 1067 | 0.0% | 15.1% | 98.9% | 26.2% | 212 |
| area | 741 | 0.0% | 10.5% | 99.4% | 19.0% | 297 |
| theme | 453 | 0.4% | 35.0% | 98.9% | 51.8% | 22 |

A deliberately simple reference proposes the single most common label in each dimension without reading the issue. Its F1 is **43.2%**, versus **24.6%** for Laya. That reference uses this snapshot's label prevalence; it is a descriptive sanity check, not a separately trained or held-out baseline. Laya is 18.6 percentage points below the no-reading reference on F1. This baseline does not demonstrate better F1 than simply using label prevalence.

## Every individual label

| Label | Existing positives | Model proposals in scorable issues | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| `area/audit-policy` | 46 | 729 | 6.3% | 100.0% | 11.9% |
| `area/ci-cd` | 44 | 731 | 6.0% | 100.0% | 11.4% |
| `area/cli` | 352 | 735 | 47.9% | 100.0% | 64.8% |
| `area/content-security` | 22 | 732 | 3.0% | 100.0% | 5.8% |
| `area/distribution` | 68 | 733 | 9.1% | 98.5% | 16.7% |
| `area/docs-site` | 102 | 740 | 13.8% | 100.0% | 24.2% |
| `area/enterprise` | 34 | 732 | 4.5% | 97.1% | 8.6% |
| `area/lockfile` | 81 | 732 | 10.9% | 98.8% | 19.7% |
| `area/marketplace` | 67 | 738 | 9.1% | 100.0% | 16.6% |
| `area/mcp-config` | 61 | 740 | 8.2% | 100.0% | 15.2% |
| `area/mcp-trust` | 11 | 736 | 1.5% | 100.0% | 2.9% |
| `area/multi-target` | 93 | 737 | 12.6% | 100.0% | 22.4% |
| `area/package-authoring` | 50 | 735 | 6.5% | 96.0% | 12.2% |
| `area/testing` | 54 | 718 | 7.2% | 96.3% | 13.5% |
| `theme/governance` | 86 | 442 | 19.0% | 97.7% | 31.8% |
| `theme/portability` | 285 | 448 | 62.9% | 98.9% | 76.9% |
| `theme/security` | 104 | 451 | 23.1% | 100.0% | 37.5% |
| `type/architecture` | 27 | 1046 | 2.6% | 100.0% | 5.0% |
| `type/automation` | 227 | 1048 | 21.5% | 99.1% | 35.3% |
| `type/bug` | 466 | 1053 | 43.8% | 98.9% | 60.7% |
| `type/docs` | 153 | 1059 | 14.4% | 100.0% | 25.2% |
| `type/feature` | 319 | 1047 | 29.8% | 97.8% | 45.7% |
| `type/performance` | 65 | 1056 | 6.2% | 100.0% | 11.6% |
| `type/refactor` | 26 | 1056 | 2.5% | 100.0% | 4.8% |
| `type/release` | 1 | 1054 | 0.1% | 100.0% | 0.2% |

## Real examples, including disagreements

These are the first examples by issue number, not hand-picked best or worst cases. An extra prediction in an entirely unlabelled dimension is shown but not scored.

### Matches in observed dimensions

No examples in this category.

### Disagreements

**[microsoft/apm#5](https://github.com/microsoft/apm/issues/5): apm install &lt;my-apm-package-repo&gt;**

- Existing classification: `type/feature`.
- Model proposal: `area/audit-policy`, `area/ci-cd`, `area/cli`, `area/content-security`, `area/distribution`, `area/docs-site`, `area/enterprise`, `area/lockfile`, `area/marketplace`, `area/mcp-config`, `area/mcp-trust`, `area/multi-target`, `area/package-authoring`, `area/testing`, `theme/governance`, `theme/portability`, `theme/security`, `type/architecture`, `type/automation`, `type/bug`, `type/docs`, `type/feature`, `type/performance`, `type/refactor`, `type/release`.
- Context shortened: no.

**[microsoft/apm#6](https://github.com/microsoft/apm/issues/6): Add support for GitHub EU**

- Existing classification: `type/feature`.
- Model proposal: `area/audit-policy`, `area/ci-cd`, `area/cli`, `area/content-security`, `area/distribution`, `area/docs-site`, `area/enterprise`, `area/lockfile`, `area/marketplace`, `area/mcp-config`, `area/mcp-trust`, `area/multi-target`, `area/package-authoring`, `area/testing`, `theme/governance`, `theme/portability`, `theme/security`, `type/architecture`, `type/automation`, `type/bug`, `type/docs`, `type/feature`, `type/performance`, `type/refactor`, `type/release`.
- Context shortened: no.

**[microsoft/apm#12](https://github.com/microsoft/apm/issues/12): \[BUG\] Fork PR Testing does not work (no access to secrets)**

- Existing classification: `type/bug`.
- Model proposal: `area/audit-policy`, `area/ci-cd`, `area/content-security`, `area/distribution`, `area/docs-site`, `area/enterprise`, `area/lockfile`, `area/marketplace`, `area/mcp-trust`, `area/multi-target`, `area/package-authoring`, `area/testing`, `theme/security`, `type/architecture`, `type/automation`, `type/bug`, `type/feature`, `type/performance`, `type/refactor`, `type/release`.
- Context shortened: no.

**[microsoft/apm#13](https://github.com/microsoft/apm/issues/13): Refactor \`apm init\` to minimal-only mode (breaking change)**

- Existing classification: `type/feature`.
- Model proposal: `area/audit-policy`, `area/ci-cd`, `area/cli`, `area/content-security`, `area/distribution`, `area/docs-site`, `area/enterprise`, `area/lockfile`, `area/marketplace`, `area/mcp-config`, `area/mcp-trust`, `area/multi-target`, `area/package-authoring`, `area/testing`, `theme/governance`, `theme/portability`, `theme/security`, `type/architecture`, `type/automation`, `type/bug`, `type/docs`, `type/feature`, `type/performance`, `type/refactor`, `type/release`.
- Context shortened: yes.

**[microsoft/apm#14](https://github.com/microsoft/apm/issues/14): Add auto-bootstrap in \`apm install\` when no apm.yml exists**

- Existing classification: `type/feature`.
- Model proposal: `area/audit-policy`, `area/ci-cd`, `area/cli`, `area/content-security`, `area/distribution`, `area/docs-site`, `area/enterprise`, `area/lockfile`, `area/marketplace`, `area/mcp-config`, `area/mcp-trust`, `area/multi-target`, `area/package-authoring`, `area/testing`, `theme/governance`, `theme/portability`, `theme/security`, `type/architecture`, `type/automation`, `type/bug`, `type/docs`, `type/feature`, `type/performance`, `type/refactor`, `type/release`.
- Context shortened: yes.

The model does not generate explanations. A disagreement is not evidence of a verified bug in the control labels or a known reasoning process inside the model.
A successful installation or small inference smoke test only proves that the software runs. It does not establish accuracy for this full, multi-label taxonomy. These results apply to this checkpoint, rubric, context policy and threshold; they do not prove that every possible Laya-based classifier would fail.

## Speed and what machine did the work

- Actual inference device: **cpu**, **INT8 encoder Linear layers with an FP32 decision head** (mixed precision: False). CPU thread limit: 4.
- Platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`.
- Median / p95 / p99: **16.034s / 29.479s / 29.913s** per issue.
- Mean / maximum: **20.857s / 30.800s**.
- Total measured inference: **461.6 minutes**.
- Amortized throughput: **2.9 issues/minute**.
- Inference shards: **8**. Load/checksum figures below are summed across shards, not parallel wall time.
- Recorded model load: **25.85s**; download/cache lookup plus checksum: **52.97s**.
- Runtime optimization: **20.14s**, summed across shards.

GPU timings synchronize the device before and after each call. They are not merely the time taken to queue GPU work. All label questions run together; multiplying the per-issue latency by the label count would double-count work. Installation, GitHub snapshot download, and report generation are not included in model latency. This run may reuse cached weights; a cold installation also downloads an approximately 843 MB model per runner.

These are actual GitHub-hosted CPU measurements. The sum of inference times is total work across runners, not elapsed workflow time. Shards ran in parallel, and their original predictions and fingerprints are preserved unchanged.

## Important limitations

1. **Short context:** 1,235/1,328 issues were shortened to at most 300 state tokens. The beginning and a smaller tail are retained, with an explicit omission marker. Useful evidence in the middle may be lost. This is not full-discussion triage.
2. **Silver controls, not gold truth:** labels reflect the repository snapshot, including legacy aliases and potentially automated decisions. We did not reconstruct who applied each label or the labels that existed when an issue was originally opened.
3. **Historical and language drift:** the English checkpoint sees old and new issues through today's taxonomy. Non-English issues are not routed to a multilingual checkpoint.
4. **Possible text hints:** 8 issues mention canonical label names in their original title/body. We did not strip that prose; the sensitivity table below also reports the corpus without those cases.
5. **Confidence is not authority:** probabilities and the SDK's action head are not permission to accept, prioritize, assign, implement, or label an issue.
6. **Runtime behavior:** the initial MPS mixed-precision pilot produced non-finite probabilities. Validation rejected them. This baseline disables mixed precision. The SDK also warns about a clamped temperature for 11-plus-option choices; this experiment uses only two-option Noul questions.
7. **Not a held-out production claim:** this is a zero-shot exploratory benchmark against a frozen public corpus. Future tuning must be evaluated on separately reserved data.

### Sensitivity checks

| Subset | Control issues | Exact agreement | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| open | 167 | 0.0% | 14.2% | 99.6% | 24.9% |
| closed | 912 | 0.0% | 14.0% | 99.0% | 24.6% |
| full_context | 78 | 0.0% | 13.3% | 95.1% | 23.3% |
| shortened_context | 1001 | 0.0% | 14.1% | 99.3% | 24.7% |
| without_label_names_in_text | 1072 | 0.0% | 14.1% | 99.1% | 24.6% |

## Provenance and reproducibility

- Snapshot collection: `2026-09-25T08:36:05.650721+00:00` to `2026-09-25T08:36:42.433889+00:00`.
- Source repository revision for the label contract: `c5fea9e1b51bf69e98256be4e4fe9c9ac66e0be9`.
- Model: `convaiinnovations/laya` at `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`.
- Safetensors SHA-256: `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
- Run fingerprint: `da923e25e142818d536476f66cd4bab7b32b921d19ebc3755a4e3738f529dd4a`.
- Run completed: `2026-09-25T11:27:16.597528+00:00`.
- Current labels, their descriptions, and the canonical alias contract are frozen with the issue snapshot.
- Model weights are data-only Safetensors. Python code is installed from the pinned official SDK source, not executed from the model repository.
- Raw per-issue probabilities and timing: [../runs/hosted-int8/predictions.jsonl](../runs/hosted-int8/predictions.jsonl).
- Machine-readable metrics and all proposals: [../runs/hosted-int8/metrics.json](../runs/hosted-int8/metrics.json) and [../runs/hosted-int8/proposals.csv](../runs/hosted-int8/proposals.csv).
- Exact installation, refresh, resume, and rerun commands are in [README.md](../README.md).

## Recommendation

**Keep this as a read-only experiment, not an automatic labelling bot.** Review disagreements, improve the rubric or context strategy, and use a separately held-out, human-adjudicated sample before authorizing any real label writes. This repository has no apply-labels command.
