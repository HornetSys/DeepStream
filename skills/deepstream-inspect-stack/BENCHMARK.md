# Skill Benchmark: deepstream-inspect-stack

> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**

One or more required evaluation tiers did not complete, so this benchmark is not publication-complete.

## Evaluation Metadata

- Skill: `deepstream-inspect-stack`
- Evaluation date: 2026-09-04
- Evaluator version: `1.5.4`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:9e96bdd8be53bed8adbdf84fef7620d9884146309a40ccd9c8e37c09bb17fb9a` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 94.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 85.7% → 100.0% (+14.3 points) |
| Correctness | Not available | 62.9% → 100.0% (+37.1 points) |
| Discoverability | Not available | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 48.3% → 93.3% (+45.0 points) |
| Efficiency | Not available | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 631,659 | 2,177,034 | -1,545,375 | -70.99% | skill 7/7; base 7/7 |
| claude-code | 1 | 107,119 | 32,140 | +74,979 | +233.29% | skill 1/1; base 1/1 |
| claude-code | 2 | 101,720 | 31,208 | +70,512 | +225.94% | skill 1/1; base 1/1 |
| claude-code | 3 | 96,329 | 30,256 | +66,073 | +218.38% | skill 1/1; base 1/1 |
| claude-code | 4 | 96,626 | 30,497 | +66,129 | +216.84% | skill 1/1; base 1/1 |
| claude-code | 5 | 105,901 | 64,388 | +41,513 | +64.47% | skill 1/1; base 1/1 |
| claude-code | 6 | 62,146 | 134,388 | -72,242 | -53.76% | skill 1/1; base 1/1 |
| claude-code | 7 | 61,818 | 1,854,157 | -1,792,339 | -96.67% | skill 1/1; base 1/1 |
| codex | All cases | 435,953 | 575,828 | -139,875 | -24.29% | skill 7/7; base 7/7 |
| codex | 1 | 110,600 | 52,947 | +57,653 | +108.89% | skill 1/1; base 1/1 |
| codex | 2 | 83,917 | 32,107 | +51,810 | +161.37% | skill 1/1; base 1/1 |
| codex | 3 | 46,795 | 39,905 | +6,890 | +17.27% | skill 1/1; base 1/1 |
| codex | 4 | 66,844 | 32,825 | +34,019 | +103.64% | skill 1/1; base 1/1 |
| codex | 5 | 69,658 | 50,846 | +18,812 | +37.00% | skill 1/1; base 1/1 |
| codex | 6 | 28,946 | 160,119 | -131,173 | -81.92% | skill 1/1; base 1/1 |
| codex | 7 | 29,193 | 207,079 | -177,886 | -85.90% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 1,067,612 | 2,752,862 | -1,685,250 | -61.22% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 1 validator(s); 1 finding(s) |
| Tier 2 | Semantic deduplication | **NOT RUN** | No result was recorded |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCHEMA/frontmatter_field_placement: Root field 'version' is ignored; use 'metadata.version' (`skills/deepstream-inspect-stack/SKILL.md`)

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
