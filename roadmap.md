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

## Sprint 2: Practical Usage with RWKV-Runner

Now that the low-level understanding is solid, shift to practical usage: run RWKV as an actual chatbot with a GUI and OpenAI-compatible API via [RWKV-Runner](https://github.com/josStorer/RWKV-Runner).

- [ ] Clone RWKV-Runner repo (gitignored, like RWKV-LM)
- [ ] Get RWKV-Runner running on Linux (no Windows) — Docker preferred
- [ ] Test the chat GUI with the 7.2B model
- [ ] Test the OpenAI-compatible API endpoint
- [ ] Multi-turn chat session — evaluate quality and context retention in practice
- [ ] Compare the chat experience against a similarly-sized Transformer model
- [ ] Document practical findings: where RWKV shines vs where Transformers are better

## Sprint 3: Conclusions & Wrap-up

- [ ] Write a comparison summary: RNN vs Transformer for different use cases
- [ ] Evaluate if/when RWKV would be the better choice over a Transformer
- [ ] Final documentation pass — ensure all findings are captured

## Status

| Sprint | Status | Notes |
|--------|--------|-------|
| Sprint 1 | Complete | All modes tested, pip package works, all models benchmarked (0.1B–13.3B, 13.3B OOM) |
| Sprint 2 | Up next | RWKV-Runner on Linux/Docker, practical chat + API testing |
| Sprint 3 | Not started | |
