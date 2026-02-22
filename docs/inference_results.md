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
