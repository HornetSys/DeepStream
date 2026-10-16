# Applying config updates after validation

After validation, if the script emitted actionable `SUGGEST [...]` lines:

1. Ask whether to apply safe config updates (1–3 bullet summary).
2. Do **not** modify the config until the user explicitly permits it.
3. Edit only the supplied config path; preserve comments and unrelated keys.
4. Prefer minimal key updates over regenerating the whole file.
5. Re-run `validate_config.py` once and give a new short recommendation.

Each approval authorizes at most 1 edit/revalidation retry: 1 minimal edit pass
and 1 static re-validation. After that re-validation, abort further edits and
report any remaining findings; do not request another approval in the same turn.

Default prompt:

```text
I can apply the safe config updates for you (for example: <top fixes>). Should I update the config file and re-run validation?
```

Ambiguous choices (e.g. `attach-sys-ts-as-ntp=1` vs `0`) require user confirmation.
Do not fix runtime/environment issues in the config unless a valid config alternative
exists (headless EglSink → ask about fakesink; do not silently swap display samples).

If the user pasted a snippet, save to a temp file and validate that file.
For design-only review, use [rules.md](rules.md) directly.
