## Description: <br>
Generates and repairs validated deepstream-app and deepstream-test5-app configuration files from a bounded requirements spec, with explicit output-path safeguards. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers building DeepStream video analytics pipelines who need to generate or repair deepstream-app and deepstream-test5-app configuration files for RTSP/GStreamer sources, streammux, inference, tracking, or message-broker settings. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>
- [references/reference.md](references/reference.md) <br>
- [references/examples.md](references/examples.md) <br>
- [references/profiles.md](references/profiles.md) <br>
- [references/interactive_qa.md](references/interactive_qa.md) <br>
- [references/natural_language_mapping.md](references/natural_language_mapping.md) <br>
- [references/validator_pitfalls.md](references/validator_pitfalls.md) <br>


## Skill Output: <br>
**Output Type(s):** [Configuration files, Shell commands] <br>
**Output Format:** [DeepStream INI-style configuration files with inline bash commands] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (4 positive, 3 negative), 1 attempt per task, each in an isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Final-answer correctness against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and the workflow executed. <br>
- Effectiveness: Goal completion (50%) plus expected workflow adherence (50%). <br>
- Efficiency: Tool-call productivity (50%) plus token efficiency (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys avoided, and workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 90.1% | 89.8% |
| Security | 100.0% → 100.0% (±0.0 pts) | 92.9% → 92.9% (±0.0 pts) |
| Correctness | 74.3% → 91.4% (+17.1 pts) | 68.6% → 85.7% (+17.1 pts) |
| Discoverability | 100.0% | 90.0% |
| Effectiveness | 76.1% → 77.1% (+1.0 pts) | 66.6% → 91.1% (+24.5 pts) |
| Efficiency | 81.7% | 89.1% |

## Skill Version(s): <br>
1.0.10 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
