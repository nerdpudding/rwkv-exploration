# AI Instructions — rwkv-exploration

Hands-on exploration of RWKV-7 "Goose" (G1) to understand how pure RNN inference differs from Transformer-based LLMs. The developer has extensive experience with Transformer tooling (llama.cpp, ollama, vLLM) but is new to RNN architectures. The goal is learning, not production deployment.

## Principles

- **SOLID, DRY, KISS** — default yes, adapt to project needs. This is an exploration project so simplicity wins.
- **One source of truth** — no duplicate information. If it exists in one place, reference it from others.
- **Never delete, always archive** — move outdated content to `archive/` with date prefix.
- **Modularity & flexibility** — keep scripts and configs modular so experiments are easy to reproduce.
- **ALL code, docs, comments, plans, and commit messages MUST be in English** — always, no exceptions. The user often communicates in Dutch, but everything written to files must be English.
- **Keep up to date** — after any change, verify that docs, agent instructions, and config files still reflect reality. Stale docs are worse than no docs.
- **Learn from mistakes** — when an approach fails or wastes effort, document it in `docs/lessons_learned.md`. This file is persistent context for AI assistants to avoid repeating the same mistakes.
- **Build on existing work** — the RWKV-LM repo contains working demo scripts. Adapt and extend rather than rewriting from scratch.
- **Use agents when their role fits** — check the agents table below before starting specialized tasks. Update agent instructions after project changes.
- **Local-first** — everything runs on local hardware (RTX 4090 + RTX 5070 Ti).
- **Docker where possible** — but only after understanding the raw setup. Conda/uv first to learn, Docker later for reproducibility.

## Workflow

1. Plan (use plan mode) → ask approval → implement → test → iterate → clean up
2. For experiments: document the setup, run the experiment, record findings in lessons_learned.md
3. For benchmarks: record hardware config, model size, settings, and results

## Project Hierarchy

```
rwkv-exploration/                          # Project root
├── AI_INSTRUCTIONS.md                     # THIS FILE — project rules, hierarchy, agents
├── README.md                              # Overview + status
├── roadmap.md                             # Sprint plan and status tracking
├── todo_<date>.md                         # Daily task tracker (temp, moves to archive/)
│
├── Concepts/                              # Initial concept and design thinking
│   ├── concept.md                         # Vision, diagrams, technical decisions
│   └── project_concept.md                 # Original project description (origin doc)
│
├── docs/                                  # Guides, specs, detailed documentation
│   └── lessons_learned.md                 # Ongoing log of what worked and what didn't
│
├── RWKV-LM/                              # [CLONED REPO, GITIGNORED] Do not modify — reference only
│   └── RWKV-v7/                           # V7 demo scripts, CUDA kernels, tokenizer
│       ├── rwkv_v7_demo.py                # GPT-mode inference demo (generic, references g1a weights)
│       ├── rwkv_v7_demo_rnn.py            # RNN-mode inference demo (no custom CUDA kernel)
│       ├── rwkv_v7_demo_fast.py           # Hybrid mode (GPT prefill + RNN generation)
│       ├── rwkv_v7a_demo.py               # GPT+RNN demo for g1a/g1b weight variants
│       ├── rwkv_v7b_demo.py               # GPT+RNN demo for g1b weight variants
│       ├── rwkv_v7_numpy.py               # Numpy reference implementation
│       ├── rwkv_v8_rc00_demo.py           # V8 RC demo (no v8 weights available, ignore)
│       ├── rwkv_v8_rc00_hybrid_demo.py    # V8 RC hybrid demo (ignore)
│       ├── rwkv_vocab_v20230424.txt        # Tokenizer vocabulary (65K tokens)
│       ├── misc/                           # LAMBADA test data (auto-evaluated by demos)
│       └── cuda/                           # Custom CUDA kernels (wkv7, wkv7s)
│
├── Models/                                # [GITIGNORED] Model weights — do not modify
│   └── rwkv7-g1/                          # RWKV-7 G1 "GooseOne" weights (.pth)
│       ├── rwkv7-g1d-0.1b-20260129-ctx8192.pth   # 0.1B g1d (~382 MB) ← primary 0.1B
│       ├── rwkv7a-g1d-0.1b-20260212-ctx8192.pth  # 0.1B g1d "a" variant (~2.0 GB, larger!)
│       ├── rwkv7b-g1b-0.1b-20250822-ctx4096.pth  # 0.1B g1b variant (~3.7 GB, ctx4096, for rwkv_v7b_demo.py)
│       ├── rwkv7-g1d-0.4b-20260210-ctx8192.pth   # 0.4B g1d (~902 MB)
│       ├── rwkv7-g1d-1.5b-20260212-ctx8192.pth   # 1.5B g1d (~3.1 GB)
│       ├── rwkv7-g1d-2.9b-20260131-ctx8192.pth   # 2.9B g1d (~5.9 GB)
│       ├── rwkv7-g1d-7.2b-20260131-ctx8192.pth   # 7.2B g1d (~14.4 GB)
│       └── rwkv7-g1d-13.3b-20260131-ctx8192.pth  # 13.3B g1d (~26.5 GB)
│
├── scripts/                               # Our own inference scripts (based on demos)
├── claude_plans/                          # Active plans from plan mode
├── archive/                               # Archived docs, plans, task trackers
│
└── .claude/
    ├── settings.json                      # Project-level Claude Code settings
    ├── settings.local.json                # Local tool permissions (nvidia-smi, conda, etc.)
    └── agents/
        ├── doc-keeper.md                  # Documentation audit agent
        ├── repo-researcher.md             # RWKV-LM codebase exploration agent
        └── environment-setup.md           # Python/CUDA environment setup agent
```

## Agents

| Agent | When to Use |
|-------|-------------|
| `doc-keeper` | After changes to verify docs still reflect reality. Documentation audits, consistency checks, broken references. |
| `repo-researcher` | To explore and understand the RWKV-LM codebase (read-only). Architecture details, demo script analysis, CUDA kernel questions. |
| `environment-setup` | Setting up conda/uv/Docker environments. Installing PyTorch + CUDA. Troubleshooting CUDA kernel compilation. GPU verification. |

## Plan Rules

- Plans go in `claude_plans/` folder
- After exiting plan mode, immediately rename the plan file to `PLAN_<topic>.md`
- After completing a plan: move to `archive/` with date prefix
- Update progress in the roadmap (one place, not duplicated)

## Archive Rules

- Completed plans → `archive/` with date prefix (e.g., `2026-02-22_PLAN_setup.md`)
- Old daily task trackers → `archive/` with date prefix
- Superseded docs → `archive/` with date prefix
- Never delete anything

## Git Commits

- No AI attribution (no "Co-Authored-By: Claude" or similar)
- Only commit when explicitly asked
- English commit messages

## After Compaction

Read order:
1. This file (`AI_INSTRUCTIONS.md`)
2. `README.md`
3. `docs/lessons_learned.md`
4. Current task tracker (`todo_<date>.md`)
5. Active plans in `claude_plans/`
6. `Concepts/concept.md`
7. List contents of `claude_plans/`, `docs/`, `archive/`
8. Continue with the task
