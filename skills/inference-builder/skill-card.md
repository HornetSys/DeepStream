## Description: <br>
Generate deployable Vision AI pipelines with high-performance video and streaming capabilities on NVIDIA GPUs. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache 2.0 <br>
## Use Case: <br>
Developers and engineers creating GPU-accelerated inference microservices or standalone applications for vision, video, and streaming workloads on NVIDIA hardware. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk) <br>
- [Inference Builder Schemas Reference](../../tools/inference_builder/schemas/README.md) <br>
- [Pipeline Configuration Schema](../../tools/inference_builder/schemas/config.schema.json) <br>
- [Platform Guide](../../tools/inference_builder/doc/platform-guide.md) <br>
- [Builder Samples Index](../../tools/inference_builder/builder/samples/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks and YAML configs] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
Evaluated against 3 evaluation tasks (3 positive) in isolated k8s-sandbox pods. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use (unsafe operations, secret leakage, unauthorized access). <br>
- Correctness: Whether the answer produced is correct against the reference answer. <br>
- Discoverability: Whether the right skill was found and activated when needed. <br>
- Effectiveness: Whether the skill helped complete the user's goal and expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 55% → 84% (+29 points) | 53% → 79% (+26 points) |
| Security | 100% → 100% (±0 points) | 67% → 67% (±0 points) |
| Correctness | 7% → 100% (+93 points) | 67% → 67% (±0 points) |
| Discoverability | 60% → 67% (+6 points) | 42% → 93% (+51 points) |
| Effectiveness | 47% → 98% (+51 points) | 52% → 81% (+29 points) |
| Efficiency | 60% → 56% (-4 points) | 37% → 87% (+50 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
