# rwkv-exploration

Hands-on exploration of RWKV-7 "Goose" — a pure RNN language model — to understand how RNN inference differs from Transformer-based LLMs.

## Table of Contents

- [Goal](#goal)
- [Architecture Overview](#architecture-overview)
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

RWKV-7 has two inference modes:
- **GPT-mode**: Parallel processing (fast prefill, like a Transformer)
- **RNN-mode**: Sequential token-by-token (constant memory generation)
- **Hybrid**: GPT for prefill, RNN for generation (best of both)

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
- [Roadmap](roadmap.md) — sprint plan and status
- [Lessons Learned](docs/lessons_learned.md) — ongoing log of findings
