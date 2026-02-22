# AI Instructions — rwkv-exploration

Hands-on exploration of RWKV-7 "Goose" (G1) to understand how pure RNN inference differs from Transformer-based LLMs. The developer has extensive experience with Transformer tooling (llama.cpp, ollama, vLLM) but is new to RNN architectures. The goal is learning, not production deployment. Sprint 1 tested native PyTorch inference (all modes, all model sizes). Sprint 2 tested GGUF via llama.cpp in Docker — confirmed it works, but RWKV-Runner was dropped (no value over llama.cpp directly). First impressions documented, more investigation needed (training/fine-tuning, edge use cases, RWKV-8 ROSA).

## Principles

- **SOLID, DRY, KISS** — default yes, adapt to project needs. This is an exploration project so simplicity wins.
- **One source of truth** — no duplicate information. If it exists in one place, reference it from others.
- **Never delete, always archive** — move outdated content to `archive/` with date prefix.
- **Modularity & flexibility** — keep scripts and configs modular so experiments are easy to reproduce.
- **ALL code, docs, comments, plans, and commit messages MUST be in English** — always, no exceptions. The user often communicates in Dutch, but everything written to files must be English.
- **Keep up to date** — after any change, verify that docs, agent instructions, and config files still reflect reality. Stale docs are worse than no docs.
- **Learn from mistakes** — when an AI approach fails, a wrong assumption is made, or effort is wasted due to a misunderstanding, document it in `docs/lessons_learned.md`. This file is exclusively for AI-facing lessons (mistakes to avoid, gotchas, corrections). It is NOT for experiment results, benchmark data, or general observations — those go in `docs/inference_results.md`.
- **Build on existing work** — the RWKV-LM repo contains working demo scripts. Adapt and extend rather than rewriting from scratch.
- **Use agents when their role fits** — check the agents table below before starting specialized tasks. Update agent instructions after project changes.
- **Local-first** — everything runs on local hardware (RTX 4090 + RTX 5070 Ti).
- **Docker where possible** — but only after understanding the raw setup. Conda/uv first to learn, Docker later for reproducibility. Note: Sprint 2's RWKV-Runner Docker setup was removed because the wrapper was useless, not because Docker is off the table. Docker may return for other purposes (frontends, APIs, etc.) but is currently not on the roadmap.

## Workflow

1. Plan (use plan mode) → ask approval → implement → test → iterate → clean up
2. For experiments and benchmarks: record setup, hardware config, model size, settings, and results in `docs/inference_results.md`
3. If something went wrong or a mistake was made: document the lesson in `docs/lessons_learned.md` (AI-facing, mistakes to avoid)

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
│   ├── inference_guide.md                 # How RWKV inference works: modes, Transformer comparison, benchmarks
│   ├── setup_guide.md                     # Full environment setup, GPU config, troubleshooting
│   ├── inference_results.md               # Raw benchmark data, VRAM usage, speed measurements
│   └── lessons_learned.md                 # AI-facing: mistakes to avoid, gotchas, corrections (NOT for results)
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
│       ├── cuda/                           # Custom CUDA kernels (wkv7, wkv7s)
│       ├── train_temp/                    # Training scripts (train.py — for fine-tuning)
│       ├── mmlu_dev_dataset/              # MMLU evaluation dataset (dev)
│       ├── mmlu_test_dataset/             # MMLU evaluation dataset (test)
│       └── rwkv_mmlu_eval.py             # MMLU evaluation script
│
├── Models/                                # [GITIGNORED] Model weights — do not modify
│   └── rwkv7-g1/                          # RWKV-7 G1 "GooseOne" weights
│       ├── *.pth                          # Native weights (fp16), various sizes (0.1B–13.3B)
│       └── GGUF/                          # GGUF quantized models (for llama.cpp)
│
├── scripts/                               # Our own inference scripts (based on demos)
│   ├── run_rnn.py                         # RNN-mode (no CUDA kernel, torch.compile)
│   ├── run_gpt.py                         # GPT-mode (wkv7 CUDA kernel, batch processing)
│   ├── run_hybrid.py                      # Hybrid (wkv7s CUDA kernel, GPT prefill + RNN gen)
│   ├── run_pipeline.py                    # rwkv pip package test (PIPELINE API, repetition penalties)
│   └── run_all_models.py                  # Benchmark all g1d models (0.1B–13.3B), speed + VRAM
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
