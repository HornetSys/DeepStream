# deepstream-setup-on-wsl

Set up and verify NVIDIA DeepStream in Docker inside WSL2 on a local or remote
Windows 11 machine.

The public core provides:

- local, guide-mode, WinRM HTTPS, and SSH target paths;
- private target profiles so passwords never enter chat, arguments, or logs;
- a read-only prerequisite check followed by confirmed installation;
- a hard WDDM workstation-GPU gate;
- WSL2, Ubuntu, Docker, NVIDIA Container Toolkit, and DeepStream provisioning;
- optional fixed and constrained custom-pipeline verification.

SSH targets must already have a host key verified through a trusted channel and
stored in `known_hosts`; the helper never auto-trusts a first connection. Before
passing `--yes`, display the exact remote command or payload plan and obtain
explicit user confirmation. Never put credentials in the command or payload
arguments.

The root `SKILL.md` is authoritative.

## License

Skill prose and documentation are licensed under CC-BY-4.0; executable and
configuration source files are licensed under Apache-2.0.
