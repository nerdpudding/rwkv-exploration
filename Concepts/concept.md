# RWKV-Exploration: Concept Document

## Vision

Develop deep practical understanding of RNN-based language model inference by running, comparing, and benchmarking RWKV-7 "Goose" against familiar Transformer-based LLMs. Build the knowledge to evaluate when an RNN architecture is the better tool for a job.

## Origin

This project started from a conversation exploring the differences between RNN and Transformer-based LLMs. The key insight: while Transformers won NLP via parallel self-attention and long context windows, modern RNNs like RWKV-7 bridge the gap — training in parallel (GPT-mode) but running inference as a pure RNN with constant memory and constant time per token.

RWKV-7 introduces "meta-in-context learning" where the model performs mini gradient-descent on its own state at each token, adapting to context during inference. The architecture recognizes all regular languages and performs state tracking — capabilities standard Transformers cannot match under equivalent complexity assumptions.

## Core Idea

```
                    RWKV-7 Inference Pipeline
                    ========================

  Prompt Text                                         Generated Text
      |                                                     ^
      v                                                     |
  [Tokenizer]                                          [Tokenizer]
      |                                                     ^
      v                                                     |
  Token IDs ──> [RWKV Model] ──> Logits ──> [Sampling] ──> Token
                     |   ^
                     v   |
                  [State]          <-- Fixed size, never grows
                  (compact)            No KV-cache!

  Two inference modes:
  ┌─────────────────────────────────────────────────────────────┐
  │ GPT-mode: Process all tokens in parallel (fast prefill)     │
  │ RNN-mode: Process one token at a time (fast generation)     │
  │ Fast/Hybrid: GPT for prefill, then switch to RNN for gen   │
  └─────────────────────────────────────────────────────────────┘
```

## System Context (C4 Level 1)

```
  ┌──────────────┐        ┌──────────────────────────────┐
  │              │        │      RWKV-7 Inference        │
  │  Developer   │──CLI──>│                              │
  │  (learning)  │<─text──│  - Model loading             │
  │              │        │  - Token processing          │
  └──────────────┘        │  - Text generation           │
                          │  - Benchmarking              │
                          └──────────────┬───────────────┘
                                         │
                          ┌──────────────┴───────────────┐
                          │        GPU (CUDA)            │
                          │  RTX 4090 / RTX 5070 Ti      │
                          └──────────────────────────────┘
```

## Container Diagram (C4 Level 2)

```
  ┌─────────────────────────────────────────────────────────┐
  │                 RWKV-7 Inference System                  │
  │                                                         │
  │  ┌─────────────┐   ┌──────────────┐   ┌─────────────┐  │
  │  │  Tokenizer   │──>│  RWKV Model  │──>│  Sampler    │  │
  │  │  (vocab 64K) │   │  (PyTorch)   │   │  (top-p,    │  │
  │  └─────────────┘   │              │   │   temp)     │  │
  │                    │  ┌─────────┐  │   └─────────────┘  │
  │                    │  │  CUDA   │  │                    │
  │                    │  │ Kernels │  │                    │
  │                    │  │ (wkv7)  │  │                    │
  │                    │  └─────────┘  │                    │
  │                    └──────────────┘                    │
  │                                                         │
  │  ┌─────────────────────────────────────────────────┐    │
  │  │              Model Weights (.pth)                │    │
  │  │  0.1B | 0.4B | 1.5B | 2.9B | 7.2B | 13.3B     │    │
  │  └─────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────┘
```

## Input/Output Design

| Phase | Input | Output |
|-------|-------|--------|
| **MVP (Sprint 1)** | Text prompt (CLI) | Generated text + tokens/sec |
| **Sprint 2** | Chat conversation | Multi-turn chat + VRAM metrics |
| **Later** | Benchmark suite | Comparison table RNN vs Transformer |

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Model | RWKV-7 G1 (.pth) | SOTA pure RNN, native PyTorch format |
| Runtime | PyTorch + CUDA kernels | Direct from source, full visibility into internals |
| Package | `rwkv` pip (v0.8.32+) | Official package, supports v7, simple API |
| Environment | conda or uv first | Understand mechanics before containerizing |
| Starting size | 0.1B or 0.4B | Fast iteration while learning the pipeline |
| Target size | 2.9B (fits 4090 easily) | Best quality/VRAM trade-off for learning |
| Stretch size | 7.2B (~14.4GB fp16) | Still fits 4090, real quality benchmark |

## Hardware & Constraints

| Resource | Spec | Notes |
|----------|------|-------|
| GPU 1 | RTX 4090, 24 GB VRAM | Primary inference GPU, fully available |
| GPU 2 | RTX 5070 Ti, 16 GB (~12 GB avail) | Secondary, 4 GB reserved for OS/display |
| CPU | AMD 5800X3D | 8 cores, good single-thread |
| RAM | 64 GB | More than enough for model loading |
| OS | Ubuntu Desktop (Linux) | CUDA native support |

RWKV state size is constant regardless of context length. For a 2.9B model in fp16, expect ~5.9 GB VRAM with negligible growth during generation. This is fundamentally different from Transformer models where KV-cache can consume additional GBs.

## Available Resources

| Resource | Location | Purpose |
|----------|----------|---------|
| RWKV-LM repo | `RWKV-LM/` | Demo scripts, CUDA kernels, tokenizer |
| G1 model weights | `Models/rwkv7-g1/` | 0.1B to 13.3B .pth files |
| `rwkv` pip package | PyPI | Official inference package (v0.8.32+) |
| Albatross | GitHub (not cloned) | Optimized inference engine (later) |

## Use Cases

### Primary
1. Run RWKV-7 inference locally and understand what happens at each step
2. Compare RNN-mode vs GPT-mode inference characteristics
3. Benchmark token generation speed and VRAM usage vs context length
4. Chat with the model and evaluate quality at different parameter counts

### Secondary
5. Compare chat quality against a Transformer model of similar size
6. Understand multi-GPU options (if applicable for RNN architecture)
7. Build a Docker container for reproducible inference setup

## Development Approach

Iterative, learning-first:
- Start with smallest model to verify setup and understand mechanics
- Scale up incrementally for quality evaluation
- Document findings and comparisons as we go
- SOLID/DRY/KISS where applicable (this is exploration, not production)
