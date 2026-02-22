# Lessons Learned

Ongoing log of what worked and what didn't during development. Primarily intended as context for AI assistants to avoid repeating mistakes, but useful for anyone picking up the project.

---

## Don't assume Transformer tooling applies to RNN models

**Lesson:** RWKV-7 is a fundamentally different architecture from Transformer LLMs. The tools and workflows familiar from the Transformer world (GGUF quantization, llama.cpp, KV-cache management, layer splitting across GPUs) may not be needed or appropriate.

**Example:** Initial instinct was to look for GGUF/llama.cpp/ollama support for RWKV. But the whole reason those tools exist is to manage Transformer-specific problems (KV-cache growth, layer distribution). RWKV has constant memory usage regardless of context length — the problem doesn't exist.

**Rule:** Evaluate RWKV on its own terms. Start with native PyTorch inference and the `rwkv` pip package before reaching for Transformer-era tooling. Only adopt external tools if they solve a problem that actually exists for this architecture.

---

## Demo scripts have hardcoded paths and architecture params

**Lesson:** All four RWKV-v7 demo scripts (`rwkv_v7_demo.py`, `rwkv_v7_demo_rnn.py`, `rwkv_v7_demo_fast.py`, `rwkv_v7_numpy.py`) have hardcoded model paths pointing to the original developer's filesystem and hardcoded architecture parameters (`n_layer`, `n_embd`) for a specific model size.

**Details:**
- Model paths must be changed before running any script.
- RNN and fast scripts expect `MODEL_NAME` **without** `.pth` extension (it's appended in code). GPT and numpy scripts expect the full path **with** `.pth`.
- Architecture params per model size: 0.1B = 12 layers / 768 embd, 0.4B = 24/1024, 1.5B = 24/2048, 2.9B = 32/2560, 7.2B = 32/4096, 13.3B = 61/4096.
- `head_size` is always 64 — never change this.
- All scripts must run from `RWKV-LM/RWKV-v7/` as working directory (tokenizer and CUDA kernel source paths are relative).

**Rule:** When switching model sizes, always update both the model path AND the architecture parameters. Check the `.pth` extension convention for the specific script.

---

## Our models are g1d, not g1a — but variant-specific demo scripts exist

**Lesson:** The generic demo scripts (`rwkv_v7_demo.py`, `rwkv_v7_demo_rnn.py`, `rwkv_v7_demo_fast.py`) reference `rwkv7-g1a-*` model names, but our downloaded models are mostly `rwkv7-g1d-*`. However, the repo also contains variant-specific scripts:
- `rwkv_v7a_demo.py` — GPT+RNN demo, references `rwkv7a-g1b-0.1b` weights
- `rwkv_v7b_demo.py` — GPT+RNN demo, references `rwkv7b-g1b-0.1b` weights (matches our `rwkv7b-g1b-0.1b-20250822-ctx4096.pth`)

The LoRA dimension parameters (`D_DECAY_LORA`, `D_AAA_LORA`, `D_MV_LORA`, `D_GATE_LORA`) in `rwkv_v7_demo.py` may need adjustment for the g1d variant.

**Anomaly:** The `rwkv7a-g1d-0.1b` file is 2.0 GB and `rwkv7b-g1b-0.1b` is 3.7 GB, while the standard `rwkv7-g1d-0.1b` is only 382 MB — all labeled as 0.1B. The larger files likely use a different architecture or include extra data. Investigate before using.

**Rule:** If model loading fails or produces garbage output, inspect the weight keys (`torch.load(...).keys()`) and compare against what the script expects. The RNN and fast scripts are more robust here — they read dimensions from the weight tensors directly. Consider using variant-specific scripts (`rwkv_v7a_demo.py`, `rwkv_v7b_demo.py`) when running variant-specific weights.

---

## RNN-mode demo needs no custom CUDA kernel

**Lesson:** Of the three main demo scripts, only `rwkv_v7_demo_rnn.py` does NOT require custom CUDA kernel compilation. It uses `torch.compile` (or `torch.jit.script` as fallback). This makes it the safest first test — if the environment works for this script, the basic PyTorch + CUDA setup is correct. CUDA JIT compilation issues (nvcc, CUDA_HOME, gcc version) only surface with the GPT-mode and hybrid-mode scripts.

**Rule:** Always test RNN-mode first when verifying a new environment.

---

## Use latest stable PyTorch, not conservative old versions

**Lesson:** Initial environment plan proposed PyTorch 2.5.1 with Python 3.11 out of caution. This was unnecessarily conservative — PyTorch 2.10.0 (Jan 2026) supports Python 3.10-3.14 and CUDA 12.6 wheels (cu126). There was no known incompatibility requiring older versions.

**Rule:** Check PyTorch's actual current stable release before pinning versions. Don't downgrade without a concrete, verified reason. As of Feb 2026: PyTorch 2.10.0 stable, Python 3.12 fine, cu126 wheels available.

---

## PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

**Lesson:** From the local-media-gen project running on the same hardware (RTX 4090 + 5070 Ti). Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces VRAM fragmentation during longer sessions, especially when loading/unloading models or running many inference rounds.

**Rule:** Set this environment variable when running PyTorch inference. Low cost, prevents subtle OOM issues that only appear after extended use.

---

## CUDA_VISIBLE_DEVICES for explicit GPU targeting

**Lesson:** With two GPUs in the system (RTX 4090 at index 0, RTX 5070 Ti at index 1), always set `CUDA_VISIBLE_DEVICES=0` when working with the 4090 only. This prevents PyTorch from accidentally touching the 5070 Ti, which may need different CUDA support (Blackwell/sm_120 requires PyTorch nightly cu128 or CUDA 13.0).

**Rule:** Set `CUDA_VISIBLE_DEVICES` explicitly. Don't rely on defaults.

---
