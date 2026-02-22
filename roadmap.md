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

## Sprint 2: Practical Usage — RWKV-Runner + GGUF on Docker

Run RWKV as an actual chatbot with an OpenAI-compatible API via [RWKV-Runner](https://github.com/josStorer/RWKV-Runner), containerized in Docker with dual-GPU support. GGUF via llama.cpp is the primary inference path (preserves RNN O(1) state, enables quantization and proven multi-GPU offloading).

See full plan: `claude_plans/PLAN_sprint2_rwkv_runner_docker.md`

- [x] Clone RWKV-Runner repo (gitignored, like RWKV-LM)
- [x] Research: RWKV-Runner backends, API, limitations
- [x] Research: GGUF models available, llama.cpp preserves RNN state
- [x] Research: multi-GPU solutions from llama_cpp and local-media-gen projects
- [ ] Build custom Docker image (CUDA 13.0, PyTorch nightly cu128, llama-cpp-python)
- [ ] Download 13.3B GGUF models (Q8_0 + FP16) from HuggingFace
- [ ] Test single-GPU: 13.3B Q8 GGUF on RTX 4090 via llama.cpp backend
- [ ] Test dual-GPU: 13.3B FP16 GGUF across RTX 4090 + RTX 5070 Ti
- [ ] Test native path: 7.2B .pth on RTX 4090 via rwkv pip backend
- [ ] Test chat experience: multi-turn conversation, context retention, tool calling
- [ ] Document findings in inference_results.md, inference_guide.md, setup_guide.md

## Sprint 3: Conclusions & Wrap-up

- [ ] Write a comparison summary: RNN vs Transformer for different use cases
- [ ] Evaluate if/when RWKV would be the better choice over a Transformer
- [ ] Final documentation pass — ensure all findings are captured

## Future (optional)

- [ ] RWKV-7 fine-tuning via `RWKV-LM/RWKV-v7/train_temp/train.py` (full fine-tuning, not LoRA)
- [ ] Investigate RWKV-PEFT for parameter-efficient fine-tuning
- [ ] Ollama / LM Studio compatibility testing with GGUF models

## Status

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1 | Complete | All modes tested, pip package works, all models benchmarked (0.1B–13.3B) |
| Sprint 2 | In progress | Plan ready, research done. Next: build Docker image |
| Sprint 3 | Not started | |
