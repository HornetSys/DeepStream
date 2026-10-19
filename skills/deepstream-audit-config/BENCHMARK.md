# Skill Benchmark: deepstream-audit-config

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `deepstream-audit-config`
- Evaluation date: 2026-09-29
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (4 positive, 1 negative)
- Dataset digest: `sha256:53d152f6e4d2ba85ca3236614ccfd54f934a600a92340682dcc2f858c203dd4d` (skill-evaluator-dataset-snapshot/1)
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
| Overall | Not available | 92.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 100.0% → 90.0% (-10.0 points) |
| Correctness | Not available | 80.0% → 100.0% (+20.0 points) |
| Discoverability | Not available | 92.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 68.8% → 87.5% (+18.7 points) |
| Efficiency | Not available | 92.2% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,005,160 | 660,971 | N/A | N/A | skill 6/15; base 5/5 |
| claude-code | deepstream-audit-config-001 | 170,682 | 90,323 | +80,359 | +88.97% | skill 1/1; base 1/1 |
| claude-code | deepstream-audit-config-002 | 160,024 | 91,888 | +68,136 | +74.15% | skill 1/1; base 1/1 |
| claude-code | deepstream-audit-config-003 | 132,771 | 150,846 | -18,075 | -11.98% | skill 1/1; base 1/1 |
| claude-code | deepstream-audit-config-004 | 204,448 | 228,047 | -23,599 | -10.35% | skill 1/1; base 1/1 |
| claude-code | deepstream-audit-config-005 | 1,337,235 | 99,867 | N/A | N/A | skill 2/2; base 1/1 |
| codex | All cases | 680,991 | 514,318 | +166,673 | +32.41% | skill 5/5; base 5/5 |
| codex | deepstream-audit-config-001 | 84,147 | 58,244 | +25,903 | +44.47% | skill 1/1; base 1/1 |
| codex | deepstream-audit-config-002 | 83,643 | 56,171 | +27,472 | +48.91% | skill 1/1; base 1/1 |
| codex | deepstream-audit-config-003 | 80,997 | 60,440 | +20,557 | +34.01% | skill 1/1; base 1/1 |
| codex | deepstream-audit-config-004 | 296,610 | 185,796 | +110,814 | +59.64% | skill 1/1; base 1/1 |
| codex | deepstream-audit-config-005 | 135,594 | 153,667 | -18,073 | -11.76% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,686,151 | 1,175,289 | N/A | N/A | skill 11/20; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 38 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/deepstream-audit-config/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Instructions lack clear action verbs (`skills/deepstream-audit-config/SKILL.md`)
- **MEDIUM** QUALITY/quality_efficiency: Deeply nested references in config_edits.md (`skills/deepstream-audit-config/SKILL.md`)
- **MEDIUM** SCRIPT_LINT/deep_nesting: validate_config.py has deeply nested code (depth 8, max 6) (`skills/deepstream-audit-config/scripts/validate_config.py`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/deepstream-audit-config/SKILL.md (`BENCHMARK.md:86`)
- 33 additional finding(s) are available in the full evaluation artifacts.

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
