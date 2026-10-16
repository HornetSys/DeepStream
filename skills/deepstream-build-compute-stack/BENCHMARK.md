# Skill Benchmark: deepstream-build-compute-stack

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `deepstream-build-compute-stack`
- Evaluation date: 2026-09-16
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:c206fcfde545c3d83e6cee4a069891b79e9deb83ec3e2f92577523503270bec5` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 3
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
| Overall | 74.4% — baseline ran, but no comparable score was available; uplift unavailable | 63.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 90.0% → 75.0% (-15.0 points) | 41.7% → 42.9% (+1.2 points) |
| Correctness | 16.0% → 95.0% (+79.0 points) | 25.0% → 88.6% (+63.6 points) |
| Discoverability | 73.8% — baseline ran, but no comparable score was available; uplift unavailable | 85.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 19.6% → 50.5% (+30.9 points) | 19.6% → 40.3% (+20.7 points) |
| Efficiency | 77.8% — baseline ran, but no comparable score was available; uplift unavailable | 61.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,368,278 | 1,059,148 | N/A | N/A | skill 4/4; base 10/10 |
| claude-code | deepstream-build-compute-stack-cudnn-only-no-toolkit-004 | 705,173 | 788,827 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | deepstream-build-compute-stack-full-stack-001 | 277,605 | 150,079 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | deepstream-build-compute-stack-jetson-out-of-scope-003 | 30,124 | 29,741 | +383 | +1.29% | skill 1/1; base 1/1 |
| claude-code | deepstream-build-compute-stack-single-component-002 | 355,376 | 90,501 | N/A | N/A | skill 1/1; base 3/3 |
| codex | All cases | 2,214,810 | 2,489,402 | N/A | N/A | skill 7/7; base 12/12 |
| codex | deepstream-build-compute-stack-cudnn-only-no-toolkit-004 | 766,155 | 877,842 | N/A | N/A | skill 2/2; base 3/3 |
| codex | deepstream-build-compute-stack-full-stack-001 | 177,007 | 226,446 | N/A | N/A | skill 1/1; base 3/3 |
| codex | deepstream-build-compute-stack-jetson-out-of-scope-003 | 35,033 | 543,957 | N/A | N/A | skill 1/1; base 3/3 |
| codex | deepstream-build-compute-stack-single-component-002 | 1,236,615 | 841,157 | +395,458 | +47.01% | skill 3/3; base 3/3 |
| ALL AGENTS | Dataset aggregate | 3,583,088 | 3,548,550 | N/A | N/A | skill 11/11; base 22/22 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 15 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Resource-based skill lacks documentation of available resources (`skills/deepstream-build-compute-stack/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/deepstream-build-compute-stack/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/deepstream-build-compute-stack/SKILL.md (`BENCHMARK.md:82`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/deepstream-build-compute-stack/SKILL.md (`BENCHMARK.md:83`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/deepstream-build-compute-stack/SKILL.md (`BENCHMARK.md:84`)
- 10 additional finding(s) are available in the full evaluation artifacts.

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
