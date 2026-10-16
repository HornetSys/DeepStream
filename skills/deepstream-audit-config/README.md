# deepstream-audit-config

Human-oriented index for the `deepstream-audit-config` agent skill.

- **Agents** — follow [SKILL.md](SKILL.md) for workflow, probes, and reporting style.
- **Validation rules** — [references/rules.md](references/rules.md)
- **Probe flags** — [references/probe_flags.md](references/probe_flags.md)
- **RTSP/NTP policy** — [references/rtp_ntp_policy.md](references/rtp_ntp_policy.md)
- **Example findings** — [references/examples.md](references/examples.md)

## Layout

| Path | Role |
|------|------|
| `SKILL.md` | Agent instructions |
| `requirements.txt` | Optional PyYAML dependency for `.yml` / `.yaml` validation |
| `scripts/validate_config.py` | Config validator + approval-gated runtime probes |
| `scripts/ntp_probe.py` | Bounded RTSP NTP / RTCP SR probe (invoked by validator) |
| `references/rules.md` | Detailed validation rules |
| `evals/evals.json` | Skill evaluation cases |
| `evals/config.yml` | Canonical Tier 3 Harbor task-source declaration |
| `evals/files/` | Checked-in config fixtures used by file-backed audit evals |
| `tests/test_evals_contract.py` | Local regression checks for fixtures, activation scope, and approval gates |

## Relationship to config generator

`deepstream-generate-config` owns creation and its render-time validation. This
skill audits a generated config only when the user explicitly requests an
independent audit or supplies it later for review. Do not activate the auditor or
run `validate_config.py` during a creation-only turn. If a generated config fails
an independent audit, fix the spec/generator and re-render — do not hand-edit
generated output in a loop.
