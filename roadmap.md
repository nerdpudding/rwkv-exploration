# Roadmap — rwkv-exploration

## Sprint 1: Get Running & Understand Basics (complete)

- [x] Set up conda environment (`rwkv7`, Python 3.12, PyTorch 2.10.0 cu126)
- [x] Create own inference scripts in `scripts/` (based on demos, with our model paths)
- [x] Run RNN-mode with 0.1B model — simplest, no CUDA kernel needed
- [x] Run GPT-mode with 0.1B model — tests CUDA JIT kernel compilation
- [x] Run hybrid mode — understand the prefill + RNN combo
- [x] Document VRAM usage and token speed (see `docs/inference_results.md`)
- [x] Test the `rwkv` pip package as an alternative to the raw demo scripts
- [x] Scale up to 2.9B model and run a basic chat prompt
- [x] Benchmark all models (0.1B–13.3B) — speed, VRAM, quality comparison (13.3B: OOM)
- [x] Write inference guide explaining modes, Transformer comparison, and benchmarks

## Sprint 2: GGUF via RWKV-Runner Docker (complete)

Tested RWKV-Runner + Docker with GGUF models (13.3B Q8 and FP16) on dual-GPU hardware.

See archived plan: `archive/2026-02-22_PLAN_sprint2_rwkv_runner_docker.md`

- [x] Clone RWKV-Runner repo
- [x] Research: RWKV-Runner backends, API, limitations
- [x] Research: GGUF models available, llama.cpp preserves RNN state
- [x] Research: multi-GPU solutions from llama_cpp and local-media-gen projects
- [x] Build custom Docker image (CUDA 13.0, llama-cpp-python sm_89+sm_120)
- [x] Download 13.3B GGUF models (Q8_0 + FP16) from HuggingFace
- [x] Test single-GPU: 13.3B Q8 GGUF — loaded, 9741 MiB VRAM
- [x] Test dual-GPU: 13.3B FP16 GGUF — 41/20 layer split across RTX 4090 + RTX 5070 Ti
- [x] Test chat experience — works, but no system prompt, sampling params not optimized
- [x] Evaluate RWKV-Runner — thin wrapper, no GPU control, web UI stripped in web mode
- [x] Conclusion: RWKV-Runner dropped, llama.cpp directly is the better path
- [x] Document findings in inference_results.md and inference_guide.md
- [~] Test native path: 7.2B .pth via Runner — not tested via Runner, already done in Sprint 1
- [~] Test tool calling — not tested

**Key outcome:** GGUF via llama.cpp works (quantization + multi-GPU layer offloading confirmed). RWKV-Runner adds no value — removed from project. For future GGUF inference, use llama.cpp directly with own scripts or wrapper.

## Sprint 3: Further investigation (not started)

First impressions from Sprint 1 and 2 are documented but preliminary. These areas need deeper investigation:

- [ ] Measure state degradation: at what context length does recall actually break down?
- [ ] Test edge use case: RWKV vs small Transformer on weak CPU with long sequences
- [ ] Test training/fine-tuning: infctx mode, state tuning, LoRA via RWKV-PEFT
- [ ] Evaluate large-batch inference: does no KV-cache actually help with concurrent requests?
- [ ] Monitor RWKV-8 ROSA development — could solve the lossy state problem
- [ ] Direct llama.cpp integration: own scripts/wrapper for GGUF inference with proper GPU control and sampling

## Future (optional)

- [ ] RWKV-7 fine-tuning via `RWKV-LM/RWKV-v7/train_temp/train.py` (full fine-tuning, not LoRA)
- [ ] Investigate RWKV-PEFT for parameter-efficient fine-tuning
- [ ] Ollama / LM Studio compatibility testing with GGUF models

## Status

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1 | Complete | All modes tested, pip package works, all models benchmarked (0.1B–13.3B) |
| Sprint 2 | Complete | GGUF via llama.cpp works, RWKV-Runner dropped, first impressions documented |
| Sprint 3 | Not started | Deeper investigation needed: state degradation, edge, training, RWKV-8 ROSA |
