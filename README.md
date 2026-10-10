# Agent DS

**An AI-assisted penetration testing framework with a terminal-first workflow.**

[![GitHub stars](https://img.shields.io/github/stars/DSCYBERS/AGENT-DS?style=flat-square)](https://github.com/DSCYBERS/AGENT-DS/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/DSCYBERS/AGENT-DS?style=flat-square)](https://github.com/DSCYBERS/AGENT-DS/network/members)
[![GitHub issues](https://img.shields.io/github/issues/DSCYBERS/AGENT-DS?style=flat-square)](https://github.com/DSCYBERS/AGENT-DS/issues)
[![Last commit](https://img.shields.io/github/last-commit/DSCYBERS/AGENT-DS?style=flat-square)](https://github.com/DSCYBERS/AGENT-DS/commits/main)
[![License](https://img.shields.io/github/license/DSCYBERS/AGENT-DS?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Status](https://img.shields.io/badge/status-alpha-orange?style=flat-square)](#project-status)

All badges above are live — they pull from the GitHub API rather than being hand-typed, so they stay accurate as the repository changes.

---

## Table of contents

- [Overview](#overview)
- [Project status](#project-status)
- [Legal notice](#legal-notice)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology phases](#methodology-phases)
- [Project structure](#project-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Developer](#developer)
- [License](#license)

## Overview

Agent DS is a Python CLI that walks through a structured, ten-phase penetration-testing methodology — reconnaissance, vulnerability analysis, web and database exploitation, admin-access testing, AI-adaptive attacks, privilege escalation, persistence testing, and reporting — presented through a live terminal dashboard built on [Rich](https://github.com/Textualize/rich).

It is organized in two layers:

| Layer | Role |
|---|---|
| `core/` | The framework itself — an AI coordination manager, a module enhancer, and the attack modules they orchestrate (SQL injection, XSS, auth bypass, reconnaissance, etc.). |
| `main_agent_ds.py` | The primary CLI entry point — argument parsing, target/scope validation, authorization gating, and the phase-by-phase dashboard. |

## Project status

This is an early-stage, actively developed framework, not a finished product. Based on a direct read of the current source:

**Working today**

- Target validation (URL, domain, IP, CIDR) and an authorization/scope-guard flow that must be explicitly confirmed, with an allowed-domain allowlist checked before a run starts.
- `core/attacks/complete_ai_integration.py` — a real coordination layer (mission context, phased execution, module enhancement), not a stub.
- CLI argument handling for non-interactive, scripted runs.

**Known gaps**

- `main_agent_ds.py` imports `core.attacks.interactive_cli`, which does not exist at that path in the repository (a top-level `interactive_cli.py` exists instead). Interactive mode (`--interactive`) will fail to import and fall back to demo mode until this is corrected.
- No `requirements.txt` or `setup.py` ships with the repository, despite both being referenced in earlier documentation. The auto-installer in `main_agent_ds.py` only covers `rich`, `requests`, `beautifulsoup4`, and `colorama` — the AI/coordination modules under `core/` will likely need more.
- The phase-by-phase execution in `main_agent_ds.py` currently runs on timed, scripted output (`asyncio.sleep()` plus fixed status lines and counter increments) rather than invoking `core/`'s coordination manager per phase. In its current wiring, the dashboard narrates a run rather than reporting live findings.

None of this is unusual for a framework mid-build — it's documented here so the CLI's current behavior isn't mistaken for live scan results. See [Roadmap](#roadmap) for the suggested path to closing these gaps.

## Legal notice

This project is intended **for authorized security testing only** — systems you own, or have explicit written permission to test. The CLI enforces this at startup: it will not proceed without confirmed authorization, an operator and company name, and an allowed-domain list that the target is checked against. Testing systems without authorization is illegal in most jurisdictions.

## Architecture

```
            ┌───────────────────────┐
            │   main_agent_ds.py    │   CLI entry point, dashboard,
            │   (entry point)       │   scope/auth validation
            └──────────┬────────────┘
                       │
                       ▼
            ┌───────────────────────┐
            │  core/attacks/        │   AI coordination manager +
            │  complete_ai_          │   module enhancer
            │  integration.py       │
            └──────────┬────────────┘
                       │
                       ▼
            ┌───────────────────────┐
            │  Attack modules        │   Web, database, auth, recon
            │  (core/)               │   modules the coordinator
            │                        │   orchestrates
            └───────────────────────┘
```

## Installation

```bash
git clone https://github.com/DSCYBERS/AGENT-DS.git
cd AGENT-DS

# No requirements.txt ships yet — this installs what main_agent_ds.py
# actually imports at startup:
pip install rich requests beautifulsoup4 colorama

# Exercising the core/ AI modules directly may surface additional
# ImportErrors — check core/attacks/ and core/ai/ as they come up.
```

## Usage

Flags below are taken directly from the `argparse` configuration in `main_agent_ds.py`.

```bash
# Guided run — prompts for company name, operator, allowed domains,
# explicit authorization, and an optional OTX API key if not passed as flags
python main_agent_ds.py https://example.com

# Non-interactive, fully-flagged run
python main_agent_ds.py https://example.com \
  --authorized \
  --company-name "Acme Corp" \
  --operator-name "Your Name" \
  --allowed-domains "example.com"

# Interactive terminal UI (see Known gaps — currently broken)
python main_agent_ds.py --interactive

# Dependency / system status check
python main_agent_ds.py --status
```

| Flag | Description |
|---|---|
| `target` | URL, domain, or IP address (positional) |
| `--interactive`, `-i` | Launch the interactive terminal UI |
| `--demo` | Run in demonstration mode |
| `--status` | Print system/dependency status and exit |
| `--authorized` | Confirm the target is explicitly authorized |
| `--allowed-domains` | Comma-separated list of in-scope domains |
| `--company-name` | Company name recorded for the mission |
| `--operator-name` | Operator name recorded for the mission |
| `--otx-api-key` | AlienVault OTX API key for threat-intel enrichment |

## Methodology phases

| # | Phase | Covers |
|---|---|---|
| 1 | Initialization | Startup, AI engine load, target validation |
| 2 | Reconnaissance | Port, service, and DNS enumeration |
| 3 | Vulnerability Analysis | CVE lookups, AI-assisted assessment |
| 4 | Web Exploitation | SQL injection, XSS, file-upload testing |
| 5 | Database Exploitation | Fingerprinting, SQLi-based extraction |
| 6 | Admin Access Testing | Panel discovery, credential testing |
| 7 | AI-Adaptive Attacks | Payload generation, evasion techniques |
| 8 | Privilege Escalation | Local privesc, kernel/service checks |
| 9 | Persistence Testing | Backdoor and persistence-mechanism checks |
| 10 | Reporting | HTML report generation |

## Project structure

```
AGENT-DS/
├── main_agent_ds.py        # Main CLI entry point (dashboard + phase walkthrough)
├── agent_ds.py              # Secondary entry point
├── demo_agent_ds.py         # Demo-mode runner
├── interactive_cli.py       # Interactive CLI (top-level — see Known gaps)
├── launcher_v2.py           # Launcher
├── integration_test.py
├── test_integration.py
├── core/
│   ├── attacks/
│   │   ├── complete_ai_integration.py   # AI coordination + module enhancement
│   │   ├── terminal_theme.py
│   │   └── terminal_interface.py
│   └── ai/                  # Referenced by the integration layer
├── agent-ds/
├── USER_GUIDE.md
├── README_DEMO.md
└── LICENSE                  # MIT
```

## Roadmap

Near-term items that would close the gap between the one-command pitch and current behavior:

- [ ] Fix the `core.attacks.interactive_cli` import path so `--interactive` runs the real interactive UI.
- [ ] Add a `requirements.txt` covering every dependency actually imported across `core/`.
- [ ] Route each dashboard phase through the `core` coordination manager instead of the current timed/scripted stand-ins, so phase output reflects live findings.
- [ ] Add automated tests around the scope-guard and authorization flow, since it's the project's main safety control.

## Contributing

Issues and pull requests are welcome, particularly around the roadmap items above. If you're proposing a new attack module, please include tests and keep it behind the same authorization/scope-guard flow the rest of the CLI uses.

## Developer

**Kashyap Divyansh** — Founder, DSCYBER | Cyber Security Researcher

This credit is carried over directly from the project's own source — the same name and title appear in the startup banner inside `main_agent_ds.py`.

## License

Released under the [MIT License](LICENSE).

---

<sub>A note on this README: an earlier version of this document included fabricated award badges, an unattributed testimonial, and "live" global usage statistics that don't correspond to anything this local CLI tracks or reports. Those have been removed in favor of information that can be verified against the current source. A demo link, hosted docs site, and Discord invite referenced previously could not be reached from this environment — confirm they're live before re-adding them.</sub>
