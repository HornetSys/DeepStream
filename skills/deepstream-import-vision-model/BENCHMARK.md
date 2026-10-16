# Skill Benchmark: deepstream-import-vision-model

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `deepstream-import-vision-model`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 19 evaluation tasks (13 positive, 6 negative)
- Dataset digest: `sha256:cb100e32809dfeed6266f96b61903f14f006c9f678edb445c813466a2e4103f0` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 83.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 81.6% → 100.0% (+18.4 points) |
| Correctness | Not available | 79.0% → 99.0% (+20.0 points) |
| Discoverability | Not available | 27.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 75.7% → 94.0% (+18.3 points) |
| Efficiency | Not available | 98.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,412,476 | 5,144,456 | -2,731,980 | -53.11% | skill 19/19; base 19/19 |
| claude-code | 1 | 124,143 | 30,693 | +93,450 | +304.47% | skill 1/1; base 1/1 |
| claude-code | 2 | 68,897 | 31,855 | +37,042 | +116.28% | skill 1/1; base 1/1 |
| claude-code | 3 | 122,924 | 30,250 | +92,674 | +306.36% | skill 1/1; base 1/1 |
| claude-code | 4 | 107,879 | 31,644 | +76,235 | +240.91% | skill 1/1; base 1/1 |
| claude-code | 5 | 110,483 | 121,893 | -11,410 | -9.36% | skill 1/1; base 1/1 |
| claude-code | 6 | 204,022 | 33,243 | +170,779 | +513.73% | skill 1/1; base 1/1 |
| claude-code | 7 | 67,725 | 60,876 | +6,849 | +11.25% | skill 1/1; base 1/1 |
| claude-code | disco-hf-onnx-no-skill-name | 107,176 | 30,582 | +76,594 | +250.45% | skill 1/1; base 1/1 |
| claude-code | disco-ngc-trafficcamnet-no-skill-name | 197,835 | 31,080 | +166,755 | +536.53% | skill 1/1; base 1/1 |
| claude-code | disco-safetensors-export-no-skill-name | 180,723 | 31,380 | +149,343 | +475.92% | skill 1/1; base 1/1 |
| claude-code | import-2d-pose-bodypose | 160,566 | 30,983 | +129,583 | +418.24% | skill 1/1; base 1/1 |
| claude-code | import-classification-resnet50 | 149,915 | 30,815 | +119,100 | +386.50% | skill 1/1; base 1/1 |
| claude-code | import-monocular-depth-midas | 160,102 | 31,552 | +128,550 | +407.42% | skill 1/1; base 1/1 |
| claude-code | import-semantic-segmentation-unet | 107,362 | 31,234 | +76,128 | +243.73% | skill 1/1; base 1/1 |
| claude-code | non-activation-audio-classification | 269,503 | 215,849 | +53,654 | +24.86% | skill 1/1; base 1/1 |
| claude-code | non-activation-general-install-question | 30,343 | 30,251 | +92 | +0.30% | skill 1/1; base 1/1 |
| claude-code | non-activation-optical-flow | 67,617 | 3,268,299 | -3,200,682 | -97.93% | skill 1/1; base 1/1 |
| claude-code | non-activation-sam2-promptable-segmentation | 67,414 | 157,554 | -90,140 | -57.21% | skill 1/1; base 1/1 |
| claude-code | non-activation-stereo-depth | 107,847 | 914,423 | -806,576 | -88.21% | skill 1/1; base 1/1 |
| codex | All cases | 860,944 | 1,938,619 | -1,077,675 | -55.59% | skill 19/19; base 19/19 |
| codex | 1 | 14,592 | 14,623 | -31 | -0.21% | skill 1/1; base 1/1 |
| codex | 2 | 48,381 | 14,363 | +34,018 | +236.84% | skill 1/1; base 1/1 |
| codex | 3 | 14,204 | 14,282 | -78 | -0.55% | skill 1/1; base 1/1 |
| codex | 4 | 14,908 | 14,340 | +568 | +3.96% | skill 1/1; base 1/1 |
| codex | 5 | 14,176 | 13,995 | +181 | +1.29% | skill 1/1; base 1/1 |
| codex | 6 | 14,920 | 14,538 | +382 | +2.63% | skill 1/1; base 1/1 |
| codex | 7 | 46,890 | 29,834 | +17,056 | +57.17% | skill 1/1; base 1/1 |
| codex | disco-hf-onnx-no-skill-name | 47,981 | 14,321 | +33,660 | +235.04% | skill 1/1; base 1/1 |
| codex | disco-ngc-trafficcamnet-no-skill-name | 14,708 | 14,360 | +348 | +2.42% | skill 1/1; base 1/1 |
| codex | disco-safetensors-export-no-skill-name | 48,354 | 14,503 | +33,851 | +233.41% | skill 1/1; base 1/1 |
| codex | import-2d-pose-bodypose | 14,734 | 14,292 | +442 | +3.09% | skill 1/1; base 1/1 |
| codex | import-classification-resnet50 | 14,272 | 14,041 | +231 | +1.65% | skill 1/1; base 1/1 |
| codex | import-monocular-depth-midas | 48,129 | 13,961 | +34,168 | +244.74% | skill 1/1; base 1/1 |
| codex | import-semantic-segmentation-unet | 14,387 | 14,338 | +49 | +0.34% | skill 1/1; base 1/1 |
| codex | non-activation-audio-classification | 149,166 | 265,391 | -116,225 | -43.79% | skill 1/1; base 1/1 |
| codex | non-activation-general-install-question | 42,638 | 55,579 | -12,941 | -23.28% | skill 1/1; base 1/1 |
| codex | non-activation-optical-flow | 111,043 | 524,429 | -413,386 | -78.83% | skill 1/1; base 1/1 |
| codex | non-activation-sam2-promptable-segmentation | 48,838 | 621,781 | -572,943 | -92.15% | skill 1/1; base 1/1 |
| codex | non-activation-stereo-depth | 138,623 | 255,648 | -117,025 | -45.78% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,273,420 | 7,083,075 | -3,809,655 | -53.79% | skill 38/38; base 38/38 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 192 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** | 2 validator(s); 1 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 19 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **CRITICAL** CONTENT_DEDUP/chunk_count_limit: Tier 2 produced more than 512 content chunks. (`tests/test_capacity_runner_contract.py`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/deepstream-import-vision-model/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in adapter-development.md (`skills/deepstream-import-vision-model/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/deepstream-import-vision-model/SKILL.md (`BENCHMARK.md:115`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/deepstream-import-vision-model/SKILL.md (`BENCHMARK.md:116`)
- 188 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
