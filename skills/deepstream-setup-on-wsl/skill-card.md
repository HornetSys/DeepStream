## Description: <br>
Set up and verify NVIDIA DeepStream in Docker inside WSL2 on a local or remote Windows machine. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers use this skill to install WSL2, Ubuntu, Docker, NVIDIA Container Toolkit, and a requested DeepStream image on a local or remote Windows machine, validate a supported WDDM workstation GPU, and run built-in or user-supplied DeepStream pipelines. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [Other [Basic auth], Other [SSH key]] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [WSL2 DeepStream Setup Steps](references/wsl2-deepstream-steps.md) <br>
- [WinRM HTTPS Setup](references/winrm-https-setup.md) <br>
- [Version Matrix](references/version-matrix.md) <br>
- [WSL2 FAQ and Known Issues](references/wsl2-faq-and-known-issues.md) <br>
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
4 evaluation tasks (4 positive), 3 attempts per task, in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use, checking for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- Effectiveness: Whether the skill helped complete the user's goal (50% goal accuracy + 50% expected workflow adherence). <br>
- Efficiency: Whether wasted tool calls and token usage were avoided (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity measurement. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 84.1% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Security | 50.0% → 100.0% (+50.0 points) | Not available |
| Correctness | 53.3% → 80.0% (+26.7 points) | Not available |
| Discoverability | 85.0% — baseline ran, but no comparable score was available; uplift unavailable | Not available |
| Effectiveness | 62.2% → 79.2% (+17.0 points) | Not available |
| Efficiency | 76.5% — baseline ran, but no comparable score was available; uplift unavailable | Not available |

## Skill Version(s): <br>
2.1.1 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
