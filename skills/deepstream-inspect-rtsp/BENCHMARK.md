# Skill Benchmark: deepstream-inspect-rtsp

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `deepstream-inspect-rtsp`
- Evaluation date: 2026-09-04
- Evaluator version: `1.5.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:629076504c3af596a269d7076a76f1e54b9bdee3bacf205cc708545f92a37d61` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 88.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 85.7% → 100.0% (+14.3 points) |
| Correctness | Not available | 65.7% → 100.0% (+34.3 points) |
| Discoverability | Not available | 47.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 66.9% → 97.1% (+30.2 points) |
| Efficiency | Not available | 97.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 709,349 | 2,234,470 | -1,525,121 | -68.25% | skill 7/7; base 7/7 |
| claude-code | 1 | 109,280 | 126,162 | -16,882 | -13.38% | skill 1/1; base 1/1 |
| claude-code | 2 | 107,207 | 99,223 | +7,984 | +8.05% | skill 1/1; base 1/1 |
| claude-code | 3 | 107,209 | 124,299 | -17,090 | -13.75% | skill 1/1; base 1/1 |
| claude-code | 4 | 103,164 | 122,312 | -19,148 | -15.66% | skill 1/1; base 1/1 |
| claude-code | 5 | 64,171 | 1,694,670 | -1,630,499 | -96.21% | skill 1/1; base 1/1 |
| claude-code | 6 | 106,264 | 31,199 | +75,065 | +240.60% | skill 1/1; base 1/1 |
| claude-code | 7 | 112,054 | 36,605 | +75,449 | +206.12% | skill 1/1; base 1/1 |
| codex | All cases | 330,525 | 614,827 | -284,302 | -46.24% | skill 7/7; base 7/7 |
| codex | 1 | 36,138 | 40,130 | -3,992 | -9.95% | skill 1/1; base 1/1 |
| codex | 2 | 56,167 | 37,963 | +18,204 | +47.95% | skill 1/1; base 1/1 |
| codex | 3 | 55,870 | 36,448 | +19,422 | +53.29% | skill 1/1; base 1/1 |
| codex | 4 | 35,080 | 13,814 | +21,266 | +153.95% | skill 1/1; base 1/1 |
| codex | 5 | 34,690 | 436,940 | -402,250 | -92.06% | skill 1/1; base 1/1 |
| codex | 6 | 56,174 | 19,749 | +36,425 | +184.44% | skill 1/1; base 1/1 |
| codex | 7 | 56,406 | 29,783 | +26,623 | +89.39% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,039,874 | 2,849,297 | -1,809,423 | -63.50% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED** | 1 validator(s); 0 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- Schema & Repository Governance: Found skill manifest: SKILL.md
- AGENT_EVAL: Tier 3 evaluation complete: verdict PASS; best agent codex

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
