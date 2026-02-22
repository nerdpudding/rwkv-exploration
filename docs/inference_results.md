# Inference Results

Test results from running RWKV-7 inference on local hardware. Each section documents a specific test run with hardware config, model, settings, and findings.

---

## 2026-02-22 — First inference: 0.1B g1d, all three modes

### Setup

| Component | Value |
|-----------|-------|
| Model | `rwkv7-g1d-0.1b-20260129-ctx8192` (382 MB on disk) |
| GPU | RTX 4090 (24 GB VRAM), CUDA_VISIBLE_DEVICES=0 |
| PyTorch | 2.10.0+cu126 |
| Python | 3.12.12 (conda env `rwkv7`) |
| DTYPE | float16 |
| Prompt | `"User: simulate SpaceX mars landing using python\n\nAssistant: <think"` |
| Generation | 500 tokens, temperature=1.0, top_p=0.0 (greedy) |

### Architecture params (verified from weight file)

| Parameter | Value |
|-----------|-------|
| n_layer | 12 |
| n_embd | 768 |
| n_head | 12 |
| head_size | 64 |
| vocab_size | 65536 |
| D_DECAY_LORA | 64 |
| D_AAA_LORA | 64 |
| D_MV_LORA | 32 |
| D_GATE_LORA | 128 |

LoRA dimensions are identical to the demo script defaults — no adjustment needed for g1d.

### Generation speed

| Mode | Script | tok/s (real) | tok/s (pure model) | Total time (500 tok) |
|------|--------|-------------|-------------------|---------------------|
| RNN | `scripts/run_rnn.py` | 171 | 181 | 3.0s |
| GPT | `scripts/run_gpt.py` | n/a (batch only) | n/a | n/a |
| Hybrid | `scripts/run_hybrid.py` | 166 | 175 | 3.4s |

- "real" includes sampling and tokenizer overhead. "pure model" is forward pass only.
- RNN and hybrid generate at similar speed because both use the RNN path for token-by-token generation. The hybrid advantage is in prefill (parallel GPT-mode for the prompt), which matters more for longer prompts.

### CUDA kernel compilation

| Kernel | Mode | Registers | Spill | Shared mem | Status |
|--------|------|-----------|-------|------------|--------|
| none (torch.compile) | RNN | n/a | n/a | n/a | OK |
| wkv7 | GPT | 128 | 0 bytes | 1280 bytes | OK, sm_89 |
| wkv7s | Hybrid | 130 | 0 bytes | 1280 bytes | OK, sm_89 |

Both CUDA kernels compiled cleanly for sm_89 (Ada) with zero register spill — optimal.

### VRAM usage

| Metric | Value |
|--------|-------|
| Model weights (steady state) | 371 MB |
| RNN state (all layers) | 2.3 MB |
| Total allocated | 373 MB |
| Peak during loading | 725 MB |
| % of RTX 4090 (24 GB) | 1.5% |

The state is fixed at 2.3 MB regardless of context length — this is the core RNN advantage over Transformers where KV-cache grows linearly.

### LAMBADA benchmark

| Mode | Samples | Perplexity | Accuracy |
|------|---------|-----------|----------|
| RNN | 50 (partial) | ~22 | ~36% |
| GPT | 2600 (partial) | 14.9 | 45.5% |
| Hybrid | 1600 (partial) | 15.0 | 45.7% |

The RNN partial result (50 samples) shows higher perplexity / lower accuracy due to small sample size. GPT and hybrid results converge to the same scores with more samples (same model, same weights). GPT-mode evaluates LAMBADA much faster because it processes entire sequences in one forward pass.

### Output quality

At 0.1B parameters, the model generates coherent English but hallucinates heavily. The "think" prompt triggers a reasoning chain that loops and contradicts itself (e.g., "SpaceX landed in 2011" — incorrect). This is expected behavior for a model this small. Quality evaluation should wait for 2.9B+ models.

### Next prediction test (GPT-mode)

Prompt: "The Eiffel tower is in the city of"

| Token | Probability |
|-------|------------|
| Paris | 36.4% |
| E | 15.2% |
| Lyon | 1.6% |
| New | 1.1% |
| Mars | 1.0% |

The model correctly predicts "Paris" as the most likely continuation.

---

## 2026-02-22 — All models benchmark: 0.1B–7.2B via rwkv pip package

### Setup

| Component | Value |
|-----------|-------|
| Models | All g1d models: 0.1B, 0.4B, 1.5B, 2.9B, 7.2B, 13.3B |
| GPU | RTX 4090 (24,081 MB VRAM), CUDA_VISIBLE_DEVICES=0 |
| PyTorch | 2.10.0+cu126 |
| Python | 3.12.12 (conda env `rwkv7`) |
| Strategy | `cuda fp16` via `rwkv` pip package |
| Sampling | temperature=1.0, top_p=0.85, alpha_frequency=0.2, alpha_presence=0.2 |
| Script | `scripts/run_all_models.py` |

### Summary table

| Model | Disk (MB) | VRAM (MB) | Peak (MB) | Load (s) | Factual (tok/s) | Chat (tok/s) | Speed (tok/s) |
|-------|-----------|-----------|-----------|----------|-----------------|--------------|---------------|
| 0.1B  | 365       | 364       | 378       | 1.7      | 79.3            | 118.2        | 147.6         |
| 0.4B  | 860       | 868       | 881       | 1.9      | 57.4            | 80.1         | 77.0          |
| 1.5B  | 2,914     | 2,921     | 2,948     | 3.8      | 61.7            | 69.6         | 67.0          |
| 2.9B  | 5,623     | 5,630     | 5,675     | 5.3      | 53.5            | 52.7         | 51.9          |
| 7.2B  | 13,733    | 13,739    | 13,811    | 13.6     | 50.7            | 50.9         | 50.7          |
| 13.3B | 25,311    | —         | —         | —        | —               | —            | OOM           |

### Tests per model

- **Factual** (100 tokens): `"The Eiffel tower is in the city of"`
- **Chat + think** (200 tokens): `"User: Explain gravity in one paragraph.\n\nAssistant: <think>"`
- **Speed** (500 tokens, silent): `"Once upon a time in a land far away,"` — pure throughput, no print overhead

### Observations

**VRAM scales linearly with disk size.** Steady-state VRAM ≈ disk size in every case. Peak during generation is only 1–4% above steady state. Peak during loading is higher (up to 26% for 0.1B) but still modest — no 2x spike like some frameworks.

**Speed flattens at 2.9B+.** Token generation rate drops from 148 tok/s (0.1B) to ~51 tok/s (2.9B), then stays flat through 7.2B. The bottleneck shifts from compute to memory bandwidth at larger sizes.

**13.3B does not fit on RTX 4090.** At 25.3 GB on disk, it exceeds the 24 GB VRAM. Would need quantization or multi-GPU to run.

**7.2B uses 57% of RTX 4090 VRAM** (13.7 GB of 24 GB), leaving plenty of headroom for longer sequences or batching.

**Quality improves noticeably with scale:**
- 0.1B: Says the Eiffel Tower is in "Lyons" — wrong. Gravity explanation includes made-up formulas.
- 0.4B: Correctly says Paris. Mentions 1889 construction (correct). Gravity explanation is coherent.
- 1.5B: Correct, more structured (bullet-point style). Mentions 312m height (actual: 330m, close).
- 2.9B: Correct, mentions 1889, "tallest building in the world" at the time (correct). Gravity explanation references Newton's formula.
- 7.2B: Best quality. Correctly names the original designers (Maurice Koechlin and Émile Nouguier). Gravity explanation is clean and complete.

**0.1B speed test generated only 328 of 500 tokens** — hit a stop token early. All other models completed the full token count.

---

## 2026-02-22 — Sprint 2: GGUF via RWKV-Runner Docker

### Setup

| Component | Value |
|-----------|-------|
| Container | Custom Docker image |
| Base image | `nvidia/cuda:13.0.0-devel-ubuntu24.04` |
| Backend | llama-cpp-python (GGUF via llama.cpp) |
| CUDA architectures | sm_89 (Ada) + sm_120 (Blackwell) |
| GPU 0 | RTX 4090 (24 GB VRAM) |
| GPU 1 | RTX 5070 Ti (16 GB VRAM) |
| Frontend | RWKV-Runner in web mode |
| Model | `rwkv7-g1d-13.3b` GGUF (Q8_0 and FP16) |

### Q8_0 GGUF (single-GPU target)

| Metric | Value |
|--------|-------|
| Model file | `rwkv7-g1d-13.3b-Q8_0.gguf` (~14 GB) |
| Total VRAM | 9,741 MiB |
| GPU placement | Split across both GPUs (unintended — no control via RWKV-Runner) |
| Recurrent state | 63 MiB fixed (no KV cache) |
| Memory type | `llama_memory_recurrent` confirmed in logs |

The Q8 model should fit on a single RTX 4090 but was split across both GPUs because RWKV-Runner provides no GPU placement control.

### FP16 GGUF (dual-GPU)

| Metric | Value |
|--------|-------|
| Model file | `rwkv7-g1d-13.3b-20260131-ctx8192-FP16.gguf` (~25 GB) |
| CUDA0 (RTX 4090) | 41 layers, 16,454 MiB |
| CUDA1 (RTX 5070 Ti) | 20 layers, 8,538 MiB |
| CPU offload | 512 MiB |
| Graph splits | 8 |
| Recurrent state | 63 MiB total (42 MiB CUDA0 + 21 MiB CUDA1) |

The FP16 model auto-split across both GPUs. The recurrent state is constant at 63 MiB regardless of context length — confirmed by `llama_memory_recurrent` in llama.cpp logs.

### Chat test

The model generates coherent responses but:
- No system prompt configured — the model identifies as ChatGPT or Qwen in some responses (likely from chat template / training data defaults)
- Sampling parameters used were llama-cpp-python defaults, not the model README's recommended settings (likely affected output quality)

### RWKV-Runner evaluation

| Aspect | Finding |
|--------|---------|
| Architecture | Thin wrapper around llama-cpp-python for GGUF models |
| Web UI (web mode) | Strips Configs, Models, Downloads, Train, About tabs — only Chat + Completion + Settings visible |
| GPU placement | No control — cannot specify which GPU(s) to use or how to split layers |
| Added value | None over using llama.cpp / llama-cpp-python directly |

**Conclusion:** RWKV-Runner dropped from the project. For GGUF inference, use llama.cpp directly.

### What was confirmed (tested)

- **GGUF via llama.cpp works.** Both Q8_0 (single-GPU) and FP16 (dual-GPU) loaded and ran successfully via llama-cpp-python. Quantized GGUF and layer offloading across GPUs both work.
- **Recurrent state is real.** 63 MiB fixed state regardless of context, confirmed via `llama_memory_recurrent` in llama.cpp logs. No KV-cache.
- **RWKV-Runner adds no value.** It's a thin wrapper around llama-cpp-python. For future tests, a direct implementation with llama.cpp using own scripts or a simple wrapper will work better — more control over GPU placement, sampling parameters, and configuration.

### What needs more testing

- **State degradation:** Known architectural property (lossy compression of older context) but not measured — at what point does recall actually break down?
- **Edge / constrained hardware:** RWKV's linear complexity vs Transformer's quadratic attention on weak CPUs with long sequences and tiny models. Potentially interesting niche, untested.
- **Large-batch inference:** No KV-cache means more concurrent requests fit in VRAM. Theoretical advantage for high-concurrency long-context workloads, but counterarguments exist (short RAG contexts make KV-cache small anyway, PagedAttention shrinks the problem, MoE is already efficient).
- **Training and fine-tuning:** Completely untested. RWKV reportedly has real advantages here (linear memory, infctx mode, state tuning). See [inference_guide.md](inference_guide.md) for details.
- **RWKV-8 ROSA:** Experimental architecture that directly targets the lossy state problem. See [inference_guide.md](inference_guide.md).

### Preliminary observations: RWKV vs Transformers

> **These are first impressions from discussion and architectural reasoning, NOT measured benchmarks. None of this has been tested or quantified. RWKV could be an interesting alternative in certain scenarios — these observations need deeper investigation before drawing conclusions.**

| Observation | Counterargument |
|-------------|-----------------|
| Lossy state means RWKV loses detail from older context | For tasks where exact recall doesn't matter (summarization, style continuation, sentiment tagging), lossy compression is fine and the simplicity is an advantage |
| KV-cache savings don't matter on large GPU clusters | On a single low-budget GPU serving many concurrent users, no KV-cache means more requests fit in VRAM. Small difference, but real |
| Tiny Transformers (Phi, Qwen) may beat RWKV on edge hardware | Phi/Qwen still have quadratic attention. On very long sequences at tiny model size on a weak CPU, RWKV's linear complexity is faster. Not tested — worth investigating for edge use cases |
| 256K Transformer context beats RWKV's infinite lossy state for most tasks | On CPU-only or very low RAM devices, a 256K KV-cache needs gigabytes. RWKV uses constant RAM — on truly constrained hardware it may be the only workable option |
| RWKV claimed to excel at large-batch inference (no KV-cache = more concurrent requests in same VRAM) | With short contexts (RAG) KV-cache is small anyway; MoE is already efficient; PagedAttention shrinks the problem; if a larger RWKV model is needed for same quality, VRAM savings are eaten up |

**Bottom line:** RWKV-7 runs and works via llama.cpp with GGUF — quantization and multi-GPU layer offloading are functional. The RWKV-Runner wrapper adds nothing; llama.cpp directly (with own scripts) is the better path forward. On hardware with a modern GPU or decent ARM chip with enough RAM, a small quantized Transformer likely wins on most practical metrics. The potential niche for RWKV is weak CPU + very long sequences + tiny model size — but this is untested. Training/fine-tuning advantages and RWKV-8 ROSA are open areas that could change this picture.
