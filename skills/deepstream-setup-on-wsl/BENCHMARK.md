# Skill Benchmark: deepstream-setup-on-wsl

> ❌ **Overall verdict: FAIL — Publication blocked**

The skill should be reviewed before publication. Address the blocking findings below, then rerun Skill Evaluator.

## Evaluation Metadata

- Skill: `deepstream-setup-on-wsl`
- Evaluation date: 2026-09-17
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 4 evaluation tasks (4 positive)
- Dataset digest: `sha256:83fdb79851b04a9cf07a5f19468648089c74d51214a32e1554dd37120556cfc9` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 84.1% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 50.0% → 100.0% (+50.0 points) | Not available |
| Correctness | 53.3% → 80.0% (+26.7 points) | Not available |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 62.2% → 79.2% (+17.0 points) | Not available |
| Efficiency | 76.5% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 787,401 | 2,403,267 | N/A | N/A | skill 4/4; base 6/6 |
| claude-code | 1 | 399,291 | 795,142 | N/A | N/A | skill 1/1; base 3/3 |
| claude-code | 2 | 63,296 | 1,042,017 | -978,721 | -93.93% | skill 1/1; base 1/1 |
| claude-code | 3 | 151,575 | 150,938 | +637 | +0.42% | skill 1/1; base 1/1 |
| claude-code | 4 | 173,239 | 415,170 | -241,931 | -58.27% | skill 1/1; base 1/1 |
| codex | All cases | 334,809 | 1,490,639 | N/A | N/A | skill 6/6; base 10/12 |
| codex | 1 | 133,741 | 378,362 | N/A | N/A | skill 1/1; base 3/3 |
| codex | 2 | 41,882 | 50,771 | -8,889 | -17.51% | skill 1/1; base 1/1 |
| codex | 3 | 80,927 | 80,317 | +610 | +0.76% | skill 3/3; base 3/3 |
| codex | 4 | 78,259 | 981,189 | N/A | N/A | skill 1/1; base 3/3 |
| ALL AGENTS | Dataset aggregate | 1,122,210 | 3,893,906 | N/A | N/A | skill 10/10; base 16/18 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **FAILED** | 11 validator(s); 213 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 4 task(s) |

## Blocking Findings

- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/deepstream-setup-on-wsl/scripts/run_verification.py:88`)

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** BANDIT/B108:hardcoded_tmp_directory: Probable insecure usage of temp file/directory. (CWE-377) (`skills/deepstream-setup-on-wsl/scripts/run_verification.py:88`)
- **MEDIUM** QUALITY/quality_correctness: No documented scripts in table format (`skills/deepstream-setup-on-wsl/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: Instructions don't mention 'run_script' (`skills/deepstream-setup-on-wsl/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/deepstream-setup-on-wsl/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/deepstream-setup-on-wsl/SKILL.md`)
- 208 additional finding(s) are available in the full evaluation artifacts.

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
