# rwkv-exploration

Hands-on exploration of RWKV-7 "Goose" — a pure RNN language model — to understand how RNN inference differs from Transformer-based LLMs.

## Table of Contents

- [Goal](#goal)
- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Scripts](#scripts)
- [Use Cases](#use-cases)
- [Key Technical Choices](#key-technical-choices)
- [Resources](#resources)
- [Hardware](#hardware)
- [Development Approach](#development-approach)
- [Project Structure & Agents](#project-structure--agents)
- [Documentation](#documentation)

## Goal

Learn how RNN-based language models differ from Transformer-based LLMs through hands-on experimentation with RWKV-7. Key questions: How does constant memory / no KV-cache change the inference story? Where does RWKV excel, where does it fall short?

## Architecture Overview

```
  Prompt ──> [Tokenizer] ──> [RWKV-7 Model] ──> [Sampler] ──> Generated Text
                                  |   ^
                                  v   |
                               [State]       <-- Fixed size, no KV-cache
```

RWKV-7 has three inference modes:
- **GPT-mode**: Parallel processing (fast prefill, like a Transformer)
- **RNN-mode**: Sequential token-by-token (constant memory generation)
- **Hybrid**: GPT for prefill, RNN for generation (best of both)

## Quick Start

Prerequisites: NVIDIA GPU with CUDA 12.6+, conda, git.

```bash
# 1. Clone external repos (not in git — these are gitignored)
git clone https://github.com/BlinkDL/RWKV-LM.git
# Download model weights into Models/rwkv7-g1/ (see HuggingFace: BlinkDL/rwkv7-g1)

# 2. Create environment
conda create -n rwkv7 python=3.12 -y
conda activate rwkv7
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy rwkv

# 3. Run inference (simplest mode first — no CUDA kernel compilation needed)
python scripts/run_rnn.py
```

For the full setup guide with GPU targeting, environment variables, and troubleshooting, see [docs/setup_guide.md](docs/setup_guide.md).

## Scripts

This project has two approaches to running RWKV-7 inference:

### Our scripts (`scripts/`) — for learning

Low-level scripts adapted from the upstream demo scripts in `RWKV-LM/`. They show step by step what the model does: weight loading, state management, the WKV kernel, token sampling. Each script runs one inference mode with hardcoded architecture parameters.

| Script | Mode | CUDA kernel | Based on |
|--------|------|-------------|----------|
| `scripts/run_rnn.py` | RNN (sequential) | none — uses torch.compile | `rwkv_v7_demo_rnn.py` |
| `scripts/run_gpt.py` | GPT (parallel) | wkv7 (JIT compiled) | `rwkv_v7_demo.py` |
| `scripts/run_hybrid.py` | Hybrid (GPT prefill + RNN gen) | wkv7s (JIT compiled) | `rwkv_v7_demo_fast.py` |

All scripts target the 0.1B model by default. To switch model size, update `MODEL_NAME`/`MODEL_PATH` and the architecture parameters (`n_layer`, `n_embd`) — see [docs/setup_guide.md](docs/setup_guide.md) for the parameter table.

Note: `run_rnn.py` and `run_hybrid.py` generate tokens one by one (streaming output + speed measurement). `run_gpt.py` processes the full sequence in one forward pass (batch mode) — it runs LAMBADA evaluation but does not stream token-by-token output.

### `rwkv` pip package — for convenience

The official `rwkv` pip package wraps the same math in a cleaner API. It adds a `PIPELINE` class with tokenizer management, chunked prefill, and built-in repetition penalties (frequency/presence) — features our raw scripts don't have. Useful for longer generation and chat experiments.

Requires `RWKV_V7_ON=1` environment variable before import. See [docs/setup_guide.md](docs/setup_guide.md) for usage examples.

## Use Cases

- Run RWKV-7 inference locally, understand each step
- Compare RNN-mode vs GPT-mode trade-offs
- Benchmark speed and VRAM usage vs context length
- Evaluate chat quality at different parameter counts
- Compare against Transformer models of similar size

## Key Technical Choices

See [Concepts/concept.md](Concepts/concept.md) for full technical decisions and rationale.

## Resources

| Resource | Location | In git? |
|----------|----------|---------|
| Our inference scripts | `scripts/` | Yes |
| RWKV-LM source + demos | `RWKV-LM/` (cloned, gitignored) | No |
| Model weights (0.1B–13.3B) | `Models/rwkv7-g1/` (gitignored) | No |
| `rwkv` pip package | [PyPI](https://pypi.org/project/rwkv/) | — |
| Albatross inference engine | [GitHub](https://github.com/BlinkDL/Albatross) | — |
| Architecture comparison (Transformers vs MoE vs RNN vs Hybrid) | [YouTube](https://youtu.be/mx9gsRgo8b8) | — |

## Hardware

| Component | Spec |
|-----------|------|
| GPU 1 | RTX 4090 (24 GB VRAM) |
| GPU 2 | RTX 5070 Ti (16 GB, ~12 GB avail) |
| CPU | AMD 5800X3D |
| RAM | 64 GB |
| OS | Ubuntu Desktop |

## Development Approach

Iterative, learning-first. Start small (0.1B), scale up, understand before wrapping.

## Project Structure & Agents

See [AI_INSTRUCTIONS.md](AI_INSTRUCTIONS.md) for the full project hierarchy and agent descriptions.

## Documentation

- [Concept Document](Concepts/concept.md) — vision, architecture, technical decisions
- [AI Instructions](AI_INSTRUCTIONS.md) — project rules, hierarchy, agents
- [Setup Guide](docs/setup_guide.md) — full environment setup, GPU config, troubleshooting
- [Roadmap](roadmap.md) — sprint plan and status
- [Lessons Learned](docs/lessons_learned.md) — ongoing log of findings
- [Inference Results](docs/inference_results.md) — benchmark data, VRAM usage, speed measurements
