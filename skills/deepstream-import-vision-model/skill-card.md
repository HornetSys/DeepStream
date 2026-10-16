## Description: <br>
Import detection, classification, segmentation, pose, and depth models into DeepStream with task routing and evidence-gated validation. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to import supported visual models into NVIDIA DeepStream and validate their outputs through task routing, engine build, pipeline execution, and evidence-gated promotion. Detection has the established end-to-end path; classification, semantic segmentation, 2D pose, and monocular depth require model-and-fixture-specific runtime, semantic, and rendered-output evidence before claiming readiness. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Optional] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>
- [references/model-choice.md](references/model-choice.md) <br>
- [references/model-acquire.md](references/model-acquire.md) <br>
- [references/runtime-environment.md](references/runtime-environment.md) <br>
- [references/engine-approval.md](references/engine-approval.md) <br>
- [references/engine-build.md](references/engine-build.md) <br>
- [references/pipeline-run.md](references/pipeline-run.md) <br>
- [references/multitask/task-registry.md](references/multitask/task-registry.md) <br>
- [references/multitask/pipeline-graph.md](references/multitask/pipeline-graph.md) <br>
- [references/multitask/validation.md](references/multitask/validation.md) <br>
- [references/adapter-development.md](references/adapter-development.md) <br>
- [references/adapter-readiness.md](references/adapter-readiness.md) <br>
- [references/semantic-oracles.md](references/semantic-oracles.md) <br>
- [references/benchmark-evidence.md](references/benchmark-evidence.md) <br>
- [references/report-generation.md](references/report-generation.md) <br>
- [references/artifact-layout.md](references/artifact-layout.md) <br>
- [references/script-index.md](references/script-index.md) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Configuration instructions, Files] <br>
**Output Format:** [JSON result object with Markdown/HTML/PDF reports] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [KITTI detections for detection runs, category-specific structured outputs, rendered frames/video, and hashed evidence bundles] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
19 evaluation tasks (13 positive, 6 negative), 1 attempt per task, each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Equal-weight mean of `skill_efficiency` and `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 83.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 81.6% → 100.0% (+18.4 points) |
| Correctness | Not available | 79.0% → 99.0% (+20.0 points) |
| Discoverability | Not available | 27.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 75.7% → 94.0% (+18.3 points) |
| Efficiency | Not available | 98.1% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.6.0 (source: frontmatter, changelog, released 2026-09-16) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
