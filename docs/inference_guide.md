# Inference Guide

A practical overview of how RWKV-7 inference works, how it compares to Transformer-based models, and what the three inference modes mean.

## Transformers vs RNNs vs RWKV

### Transformer models (GPT, Llama, ChatGPT)

Transformer models process all tokens in parallel using "attention" — every token looks at every other token. This is powerful but expensive: the model needs to store a key-value cache (KV-cache) that grows with every token in the conversation.

**Strengths:** Excellent at recalling specific details from anywhere in the conversation. Mature ecosystem (tool calling, RAG, fine-tuning infrastructure).

**Weaknesses:** Memory usage grows with conversation length. A long chat session uses progressively more VRAM and eventually hits a limit (the context window). Processing long prompts is compute-intensive.

### Traditional RNNs (LSTM, GRU)

Traditional RNNs process tokens one at a time, compressing the entire history into a fixed-size "state." Memory stays constant regardless of conversation length — but they can only run sequentially, making them slow to train and difficult to scale.

**Strengths:** Constant memory, no context window limit in theory.

**Weaknesses:** Slow to train (no parallelism), hard to scale, older context fades from the compressed state.

### RWKV-7: the hybrid

RWKV stands for Receptance Weighted Key Value. It uses "linear attention" — a mathematical formulation that can be computed **both** in parallel (like a Transformer) **and** sequentially (like an RNN). Same model, same weights, two different execution paths.

This means RWKV gets the training efficiency of Transformers (parallel processing) with the inference efficiency of RNNs (constant memory during generation). It's not a compromise — it's architecturally designed to support both.

**Strengths:** Constant memory during generation (no KV-cache growth), fast training (parallel), efficient long-context handling.

**Weaknesses:** Lossy context compression — the fixed-size state gradually loses detail from earlier in the conversation. Less mature ecosystem than Transformers (fewer tools, less community infrastructure for tool calling, agents, etc.).

## The three inference modes

RWKV's dual nature means you can run it in three ways. These are not different models — they are different execution strategies for the **same** model with the **same** weights.

### RNN-mode (sequential)

```
Token 1 → [Model + State] → Token 2 → [Model + State] → Token 3 → ...
```

Processes one token at a time. Each token updates a fixed-size state that carries the context forward. The simplest mode — no special CUDA kernel required (uses `torch.compile`).

| Aspect | Detail |
|--------|--------|
| Best for | Simple generation, testing, environments without CUDA kernel support |
| Prompt processing | Slow (one token at a time) |
| Token generation | Fast (this is the native generation path) |
| Memory | Constant — state is a few MB regardless of context length |
| CUDA kernel | None needed |

### GPT-mode (parallel)

```
[All tokens] → [Model (parallel)] → [All outputs at once]
```

Processes the entire sequence in one parallel pass, like a Transformer. Cannot generate tokens one by one — it processes a fixed input and produces scores for all positions simultaneously.

| Aspect | Detail |
|--------|--------|
| Best for | Benchmarking, evaluation (e.g., LAMBADA), batch scoring |
| Prompt processing | Fast (parallel) |
| Token generation | Not supported (batch-only) |
| Memory | Higher (processes all tokens at once) |
| CUDA kernel | Requires `wkv7` kernel (JIT compiled on first run) |

### Hybrid-mode (parallel prefill + sequential generation)

```
[Prompt tokens] → [Model (parallel)] → State → Token 1 → [Model + State] → Token 2 → ...
                   \___ GPT-mode ___/            \___________ RNN-mode ___________/
```

Uses GPT-mode to process the prompt in parallel (fast), then switches to RNN-mode for generating new tokens one at a time (streaming). This is the practical mode for real applications.

| Aspect | Detail |
|--------|--------|
| Best for | Chat, interactive use, any application that needs streaming output |
| Prompt processing | Fast (parallel) |
| Token generation | Fast (sequential, streaming) |
| Memory | Constant during generation, efficient prompt processing |
| CUDA kernel | Requires `wkv7s` kernel (JIT compiled on first run) |

### Which mode should I use?

For almost all practical purposes: **hybrid** (or the `rwkv` pip package, which uses a similar approach internally). The only reasons to use the other modes:

- **RNN-mode**: When you can't compile CUDA kernels, or to understand how sequential inference works.
- **GPT-mode**: When you need to evaluate/score a dataset in batch, not generate text.

## Benchmark results: all models on RTX 4090

Tested with the `rwkv` pip package (`cuda fp16` strategy) on an RTX 4090 (24 GB VRAM). Each model ran three generation tests: factual completion (100 tokens), chat with reasoning (200 tokens), and a throughput test (500 tokens).

| Model | VRAM | % of 24 GB | Speed (tok/s) | Quality | Fits RTX 4090? |
|-------|------|------------|---------------|---------|----------------|
| 0.1B | 364 MB | 1.5% | 80–148 | Poor — makes factual errors | Yes, plenty of room |
| 0.4B | 868 MB | 3.6% | 57–80 | Basic — correct facts, flat style | Yes, plenty of room |
| 1.5B | 2.9 GB | 12% | 62–70 | Decent — structured, mostly accurate | Yes |
| 2.9B | 5.6 GB | 23% | 52–54 | Good — nuanced, references Newton by name | Yes |
| 7.2B | 13.7 GB | 57% | 51 | Best — names original Eiffel Tower designers correctly | Yes, 10 GB headroom |
| 13.3B | — | >100% | — | — | No — needs 25 GB, OOM |

**Key takeaways:**

- **VRAM = model file size.** No surprises. State overhead is negligible (2–3 MB).
- **Speed flattens above 2.9B.** The 7.2B model is barely slower than 2.9B (~51 tok/s both). At this point memory bandwidth is the bottleneck, not compute.
- **51 tokens/second ≈ 35 words/second.** That's roughly 10x faster than reading speed — you can't read the output as fast as it's generated.
- **Quality scales with size.** The jump from 0.1B to 0.4B fixes basic factual errors. The jump from 1.5B to 2.9B adds nuance. 7.2B is notably more knowledgeable.
- **7.2B is the sweet spot for RTX 4090.** Best quality available, still fast, and uses just over half the VRAM.

For the full benchmark data with per-test breakdowns, see [inference_results.md](inference_results.md).

## GGUF inference via llama.cpp

RWKV-7 models are available in GGUF format, which means they can run via llama.cpp — the same tool used for quantized Transformer models. This was tested in Sprint 2 with the 13.3B model in both Q8_0 and FP16 quantizations.

**How it works:** llama.cpp implements `llama_memory_recurrent` specifically for RWKV models. Instead of allocating a KV-cache (as it does for Transformers), it maintains the fixed-size recurrent state. The GGUF file contains embedded chat templates and metadata.

**What this enables:**
- Quantized inference (Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16) — same quantization options as Transformer GGUFs
- Multi-GPU layer offloading — layers distributed across GPUs automatically
- Standard llama.cpp tooling (CLI, server, Python bindings via llama-cpp-python)

**State degradation** is a known architectural property of RWKV: the fixed-size state uses lossy compression, so detail from older context gradually fades. This is fundamentally different from a Transformer's KV-cache, which retains all tokens within the context window exactly. The point at which recall actually breaks down has not been measured in this project — it is an open question for future testing.

For Sprint 2 test results (VRAM, layer splits, chat quality), see [inference_results.md](inference_results.md).

## When to use RWKV — first impressions

> **These are preliminary impressions based on limited testing and architectural reasoning. RWKV could be an interesting alternative in certain scenarios. More investigation is needed before drawing firm conclusions.**

**What was actually tested:**
- GGUF inference via llama.cpp works — quantization and multi-GPU layer offloading are functional
- Recurrent state confirmed at 63 MiB (13.3B model), constant regardless of context
- For GGUF inference, llama.cpp directly (with own scripts or a simple wrapper) is the better path — more control than RWKV-Runner over GPU placement, sampling, and configuration

**Potential niches (discussed, NOT tested):**
- **Tasks without exact recall needs** (summarization, style continuation, sentiment tagging) — lossy state compression may be acceptable, and the simplicity is an advantage
- **Low-budget single GPU with many concurrent users** — no KV-cache means more requests fit in VRAM. Small difference, but real
- **Weak CPU + long sequences + tiny model** — RWKV's linear complexity vs Transformer's quadratic attention could matter here. Worth testing for edge use cases
- **Constrained hardware (low RAM, CPU-only)** — a 256K Transformer KV-cache needs gigabytes of RAM. RWKV uses constant RAM regardless of sequence length
- **Large-batch inference** — same KV-cache argument from a throughput angle. Counterarguments: short RAG contexts make KV-cache small anyway; MoE is already efficient; PagedAttention shrinks the problem

**General impression:** On hardware with a modern GPU or decent ARM chip with enough RAM, a small quantized Transformer likely wins on most practical metrics. But the edge/constrained hardware niche is untested and potentially interesting.

## Not yet explored: training and fine-tuning

All testing so far has focused on inference. Training and fine-tuning have not been tested at all. Research (not own testing) indicates RWKV may have real advantages here:

- **Linear memory during training** — Transformers need O(n^2) memory for attention. RWKV is O(n). At 128K tokens, RWKV reportedly achieves 1.37x speedup over Flash-Attention v3, and the gap widens with sequence length.
- **infctx training mode** — unique to RWKV. Splits long sequences into chunks, passes hidden state between them. Fine-tuning on 128K+ tokens with LoRA on a single 24GB GPU is reportedly feasible (~2MB VRAM per 1024-2048 tokens for 7B). Transformers fundamentally cannot replicate this trick because attention requires all tokens to attend to all others.
- **State tuning** — unique to RWKV. Fine-tune only the initial state (tiny parameter count: hidden_dim x n_layers). Extremely cheap, reportedly effective for alignment. No Transformer equivalent.
- **RWKV-PEFT** project exists with LoRA, PiSSA, Bone, State Tuning, INT8/NF4 quantization, DeepSpeed support.

These are claims from RWKV documentation and research papers — not verified through own testing. If the training advantages hold up, this could be a more compelling case for RWKV than the inference story.

## Watch: RWKV-8 "Heron" and ROSA

As of February 2026, RWKV-8 is experimental and under active development. Its most significant feature is **ROSA** (Rapid Online Suffix Automaton) — designed to directly solve the lossy state problem, which is the main argument against RWKV:

- Introduces a discrete bottleneck (VQ-VAE codebooks) to convert hidden vectors to discrete codes
- A suffix automaton (classical CS data structure) indexes these codes on CPU in parallel with GPU computation
- When the model needs distant context, the suffix automaton performs exact pattern matching on the code history
- This is "neurosymbolic" — bridges neural generalization with symbolic precision

Early results (small scale only, as of Feb 2026):
- 1M param model: 99% accuracy on 40-digit addition/subtraction (requires exact recall)
- 40K param model: 99.8% accuracy on 1-60 digit reversal (requires exact recall)

If ROSA works at scale, it would eliminate the main argument against RWKV (lossy state = no real infinite context). But: only tested on tiny models at time of writing, not independently validated, unclear how the CPU suffix automaton scales.

Paper: [arXiv:2602.02499](https://arxiv.org/abs/2602.02499) (ROSA-Tuning, evaluated on Qwen3-Base-1.7B)
