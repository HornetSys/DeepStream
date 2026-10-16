# Skill Benchmark: deepstream-generate-config

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `deepstream-generate-config`
- Evaluation date: 2026-10-01
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (4 positive, 3 negative)
- Dataset digest: `sha256:3de8d5641dee45423330038601fb21f952a905914fbc649688439fbe7d0eb357` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 90.1% — baseline ran, but no comparable score was available; uplift unavailable | 89.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 92.9% → 92.9% (±0.0 points) |
| Correctness | 74.3% → 91.4% (+17.1 points) | 68.6% → 85.7% (+17.1 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 90.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 76.1% → 77.1% (+1.0 points) | 66.6% → 91.1% (+24.5 points) |
| Efficiency | 81.7% — baseline ran, but no comparable score was available; uplift unavailable | 89.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,890,121 | 947,277 | +942,844 | +99.53% | skill 7/7; base 7/7 |
| claude-code | invalid-profile-rejected | 96,939 | 131,854 | -34,915 | -26.48% | skill 1/1; base 1/1 |
| claude-code | missing-source-details-clarification | 499,314 | 96,476 | +402,838 | +417.55% | skill 1/1; base 1/1 |
| claude-code | negative-audit-existing-config | 118,496 | 178,356 | -59,860 | -33.56% | skill 1/1; base 1/1 |
| claude-code | negative-nsys-profile | 159,483 | 159,768 | -285 | -0.18% | skill 1/1; base 1/1 |
| claude-code | negative-nvbug-investigation | 195,313 | 187,002 | +8,311 | +4.44% | skill 1/1; base 1/1 |
| claude-code | quickstart-default-config | 166,400 | 64,783 | +101,617 | +156.86% | skill 1/1; base 1/1 |
| claude-code | rtsp-low-latency-config | 654,176 | 129,038 | +525,138 | +406.96% | skill 1/1; base 1/1 |
| codex | All cases | 1,627,641 | 785,391 | +842,250 | +107.24% | skill 7/7; base 7/7 |
| codex | invalid-profile-rejected | 29,437 | 115,242 | -85,805 | -74.46% | skill 1/1; base 1/1 |
| codex | missing-source-details-clarification | 247,958 | 122,560 | +125,398 | +102.32% | skill 1/1; base 1/1 |
| codex | negative-audit-existing-config | 54,751 | 55,214 | -463 | -0.84% | skill 1/1; base 1/1 |
| codex | negative-nsys-profile | 100,180 | 114,746 | -14,566 | -12.69% | skill 1/1; base 1/1 |
| codex | negative-nvbug-investigation | 951,025 | 310,476 | +640,549 | +206.31% | skill 1/1; base 1/1 |
| codex | quickstart-default-config | 62,290 | 38,062 | +24,228 | +63.65% | skill 1/1; base 1/1 |
| codex | rtsp-low-latency-config | 182,000 | 29,091 | +152,909 | +525.62% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,517,762 | 1,732,668 | +1,785,094 | +103.03% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 9 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SCRIPT_LINT/deep_nesting: render.py has deeply nested code (depth 9, max 6) (`skills/deepstream-generate-config/scripts/render.py`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: Without declared permissions the skill's intent is opaque and cannot be validated. (`SKILL.md:1`)
- **MEDIUM** SECURITY/Unknown (LP3): MCP Least Privilege: The skill explicitly performs file read and file write operations (reading spec/YAML files, writing generated config fil (`SKILL.md:1`)
- **MEDIUM** SECURITY/Unknown (SQP-2): The template hardcodes `http-ip=localhost` and `http-port=9000` for the nvmultiurisrcbin REST control endpoint without a (`assets/templates/source-list-multiuri.txt:37`)
- **LOW** SCRIPT_LINT/magic_numbers: render.py contains magic numbers (`skills/deepstream-generate-config/scripts/render.py`)
- 4 additional finding(s) are available in the full evaluation artifacts.

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
