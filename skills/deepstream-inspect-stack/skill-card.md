## Description: <br>
Use when diagnosing NVIDIA DeepStream SDK installation, version compatibility, container tag, JetPack, CUDA, TensorRT, driver, GPU/Jetson, WSL2, pyservicemaker, or pyds issues; also use to reroute DeepStream pipeline-building or model-import requests to sibling skills without producing code. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
CC-BY-4.0 AND Apache-2.0 <br>
## Use Case: <br>
Developers and engineers diagnosing NVIDIA DeepStream SDK installation and version-compatibility issues across discrete GPU, Jetson, container, and WSL2 deployments, and routing pipeline-building or model-import requests to the correct sibling skill. <br>

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
- [DeepStream Compatibility Matrix](references/compatibility-matrix.md) <br>
- [Known-Good Setups](references/known-good-setups.md) <br>
- [NGC Container Tags](references/container-tags.md) <br>
- [Python Bindings Troubleshooting](references/python-bindings.md) <br>
- [Verify Commands](references/verify-commands.md) <br>
- [Troubleshooting](references/troubleshooting.md) <br>
- [DeepStream Developer Guide](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2) <br>
- [DeepStream Installation Guide](https://nvidia-deepstream.docs.buildwithfern.com/deepstream/9.2/text/DS_Installation) <br>
- [NGC DeepStream Container Catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/deepstream) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Five-section diagnostic report: Risk, Most-likely mismatch, Verify commands, Known-good setup, Next step] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), each attempt in an isolated sandbox pod. Evaluator version 1.5.4. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the expected skill was selected and decoys were avoided. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and excessive token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion token usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | Not available | 94.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | Not available | 85.7% → 100.0% (+14.3 points) |
| Correctness | Not available | 62.9% → 100.0% (+37.1 points) |
| Discoverability | Not available | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | Not available | 48.3% → 93.3% (+45.0 points) |
| Efficiency | Not available | 84.2% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
