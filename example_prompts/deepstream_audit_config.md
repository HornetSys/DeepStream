# Audit an existing DeepStream config

Audit the intentionally invalid example config at
`@skills/deepstream-audit-config/evals/files/app_rtsp_bad.txt` for four
production RTSP cameras. A nonzero validator exit is expected when it identifies
the seeded error.

Use static/offline validation first and pass the use-case intent. Report the
verdict, counts, every error, and concrete fixes. Do not run runtime or network
probes; ask for explicit approval first.
