# Roadmap — rwkv-exploration

## Sprint 1: Get Running & Understand Basics

- [x] Set up conda environment (`rwkv7`, Python 3.12, PyTorch 2.10.0 cu126)
- [x] Create own inference scripts in `scripts/` (based on demos, with our model paths)
- [x] Run RNN-mode with 0.1B model — simplest, no CUDA kernel needed
- [x] Run GPT-mode with 0.1B model — tests CUDA JIT kernel compilation
- [x] Run hybrid mode — understand the prefill + RNN combo
- [x] Document VRAM usage and token speed (see `docs/inference_results.md`)
- [x] Record findings in lessons_learned.md
- [ ] Test the `rwkv` pip package as an alternative to the raw demo scripts
- [ ] Scale up to 2.9B model and run a basic chat prompt

## Sprint 2: Compare & Benchmark

- [ ] Run a multi-turn chat session with 2.9B — evaluate quality and context retention
- [ ] Measure VRAM usage over long context vs a Transformer model of similar size
- [ ] Test different sampling parameters (temp, top_p, presence/frequency penalties)
- [ ] Try the "think" prompt format and compare output quality
- [ ] Compare chat quality against a ~3B Transformer model (e.g., Phi, Qwen)
- [ ] Try the 7.2B model on RTX 4090 — measure speed and quality improvement

## Sprint 3: Deep Dive & Containerize

- [ ] Explore multi-GPU options (if applicable for RWKV architecture)
- [ ] Test Albatross inference engine for optimized performance
- [ ] Build a Docker container for reproducible inference
- [ ] Write a comparison summary: RNN vs Transformer for different use cases
- [ ] Evaluate if/when RWKV would be the better choice over a Transformer

## Status

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1 | In progress | 0.1B runs in all three modes. Next: rwkv pip package + scale to 2.9B |
| Sprint 2 | Not started | |
| Sprint 3 | Not started | |
