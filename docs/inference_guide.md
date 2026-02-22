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
