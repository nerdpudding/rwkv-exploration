# Plan: Document Sprint 2 Findings & Remove RWKV-Runner

## Context

Sprint 2 tested RWKV-Runner + Docker with GGUF models (13.3B Q8 and FP16) on dual-GPU hardware. The conclusion: RWKV-Runner adds no value over llama.cpp directly, and the RWKV architecture has no convincing text-LLM use case over Transformers/MoE for our purposes. The user wants findings documented honestly and the Runner + Docker setup fully removed.

Key findings to document:
- GGUF via llama-cpp-python works (both Q8 single-GPU and FP16 dual-GPU loaded successfully)
- Recurrent state confirmed: 63 MiB fixed (no KV cache), `llama_memory_recurrent` in logs
- GPU placement: no control via RWKV-Runner (Q8 split across both GPUs unnecessarily, FP16 auto-split 41/20 layers)
- Chat works but model has identity issues (no system prompt, claims to be ChatGPT/Qwen)
- Web UI in web-mode strips Configs/Models/Downloads/Train/About — only Chat + Completion + Settings
- RWKV-Runner is a thin wrapper around llama-cpp-python, adds nothing over llama.cpp directly
- RWKV state degradation is a known architectural property (lossy compression of older context) but we have NOT measured where it breaks down — no specific token counts tested
- We discussed RWKV's potential niches with counterarguments (see "Recovered discussion context" below) — none of these were tested or quantified
- All conclusions from Sprint 2 are preliminary and need deeper investigation

### Recovered discussion context (from separate session)

The following conclusions and counterarguments were discussed in detail. They represent our thinking, NOT tested/measured results:

1. **Lossy state** — correct, but for tasks where exact token recall doesn't matter (summarization, style continuation, sentiment tagging), lossy compression is often fine and the simplicity is an advantage.

2. **KV-cache savings at scale** — don't matter on large GPU clusters. But on a single low-budget GPU serving multiple concurrent users, no KV-cache means more requests fit in VRAM before hitting the wall. Small difference, but real.

3. **Tiny Transformers vs RWKV on edge** — Phi/Qwen still have quadratic attention complexity. On very long sequences at tiny model size on a weak CPU, RWKV's linear complexity is measurably faster. A 0.4B RWKV vs 0.4B Transformer on a weak CPU with long input: RWKV wins on speed. Quality is another story.

4. **256K Transformer context vs RWKV's infinite lossy state** — 256K wins for most tasks. But on CPU-only or very low RAM devices, a 256K KV-cache needs gigabytes of RAM. RWKV uses constant RAM regardless of sequence length — on truly constrained hardware it's the only workable option.

5. **RWKV-Runner adds nothing over llama.cpp** — fully confirmed, no counterargument. It's a GUI wrapper for people who don't want to touch a terminal.

6. **Large-batch inference** — RWKV is claimed to excel at high-throughput batch inference. The reasoning: with batch=64 at 8K context, a Transformer needs 64 separate KV-caches (can be tens of GB total), while RWKV needs 64 fixed-size states (a few MB each). So you can fit far more concurrent requests in the same VRAM, and throughput scales better because you're not VRAM-limited by KV-caches. **However:** this is the same argument as #2 restated from a throughput angle, and the counterarguments are similar — with short contexts (RAG pipelines, embeddings) the KV-cache is small anyway; MoE models only activate a fraction of parameters per token so they're already efficient; if you need a larger RWKV model for equivalent quality you eat the VRAM savings; and PagedAttention (vLLM) plus other Transformer-side optimizations already shrink the KV-cache problem significantly. Real advantage in theory for long-context high-concurrency workloads, but the conditions are specific and we haven't tested any of this.

**Bottom line from discussion:** Skepticism is justified for any hardware we realistically work with. The genuine remaining niche is weak CPU + very long sequences + tiny model size (microcontroller/very old embedded hardware). On anything with a modern GPU or even a decent ARM chip with enough RAM, a small quantized Transformer wins on every practical metric.

**Important: these are first impressions only.** In some cases RWKV could be an interesting alternative — we haven't ruled it out, we just haven't found the compelling case yet for our use cases.

### Unexplored: training and fine-tuning

We only looked at INFERENCE in Sprint 1 and 2. We have NOT investigated training/fine-tuning at all. Research (not our own testing) indicates RWKV may have real advantages here:

- **Linear memory during training** — Transformers need O(n^2) memory for attention. RWKV is O(n). At 128K tokens, RWKV reportedly achieves 1.37x speedup over Flash-Attention v3, and the gap widens with sequence length.
- **infctx training mode** — unique to RWKV. Splits long sequences into chunks, passes hidden state between them. Fine-tuning on 128K+ tokens with LoRA on a single 24GB GPU is reportedly feasible (~2MB VRAM per 1024-2048 tokens for 7B). Transformers fundamentally cannot do this.
- **State tuning** — unique to RWKV. Fine-tune only the initial state (tiny parameter count: hidden_dim x n_layers). Extremely cheap, reportedly works for alignment. No Transformer equivalent.
- **RWKV-PEFT** exists with LoRA, PiSSA, Bone, State Tuning, INT8/NF4 quantization, DeepSpeed support.

**These are NOT our findings — this is from research/documentation. Needs hands-on testing to verify.**

### Watch: RWKV-8 "Heron" and ROSA

RWKV-8 is experimental (still under development). Its most significant feature is **ROSA** (Rapid Online Suffix Automaton) — designed to directly solve the lossy state problem:

- Introduces discrete bottleneck (VQ-VAE codebooks) to convert hidden vectors to discrete codes
- A suffix automaton (classical CS data structure) indexes these codes on CPU in parallel with GPU computation
- When the model needs distant context, suffix automaton performs exact pattern matching on code history
- This is "neurosymbolic" — bridges neural generalization with symbolic precision

Early results (small scale only):
- 1M param model: 99% accuracy on 40-digit addition/subtraction (requires exact recall)
- 40K param model: 99.8% accuracy on 1-60 digit reversal (requires exact recall)

**If ROSA works at scale, it would eliminate the main argument against RWKV** (lossy state = no real infinite context). But: only tested on tiny models, not independently validated, unclear how the CPU suffix automaton scales. Big "if" for now.

Paper: [arXiv:2602.02499](https://arxiv.org/abs/2602.02499) (ROSA-Tuning, evaluated on Qwen3-Base-1.7B)

## Steps

### 1. Stop and remove Docker container + images

```bash
cd /home/rvanpolen/vibe_claude_kilo_cli_exp/rwkv-exploration
docker compose down
docker rmi rwkv-exploration-rwkv-runner 2>/dev/null  # remove built image
```

### 2. Delete RWKV-Runner directory

Not tracked in git (gitignored), just remove:
```bash
rm -rf RWKV-Runner/
```

### 3. Delete Docker files

These ARE tracked in git:
- `Dockerfile`
- `docker-compose.yml`
- `.dockerignore`

### 4. Update `.gitignore`

Remove the `RWKV-Runner/` line (no longer needed).

### 5. Update `docs/inference_results.md`

Add new section: "2026-02-22 — Sprint 2: GGUF via RWKV-Runner Docker"

Include:
- Setup table (Docker, CUDA 13.0, llama-cpp-python, dual-GPU)
- Q8 loading: 9741 MiB VRAM, loaded on RTX 4090 (but incorrectly split across both GPUs)
- FP16 loading: 41 layers CUDA0 (16,454 MiB) + 20 layers CUDA1 (8,538 MiB) + 512 MiB CPU, 8 graph splits
- Recurrent state: 63 MiB total (42 CUDA0 + 21 CUDA1), constant regardless of context
- Chat test: model works but no system prompt, identity confusion, sampling params different from model README recommendations
- RWKV-Runner evaluation: thin wrapper, web UI filtered in web mode, no GPU placement control
- Preliminary observations section: conclusions + counterarguments from our discussion (clearly marked as untested impressions based on architectural reasoning, not our own benchmarks)

### 6. Update `docs/inference_guide.md`

Add section: "GGUF inference via llama.cpp" explaining:
- llama.cpp implements `llama_memory_recurrent` for RWKV (not KV cache)
- GGUF quantization available (Q4-Q8, FP16)
- Chat template embedded in GGUF metadata
- State degradation is an architectural property (lossy compression) — not yet measured in our tests

Add section: "When to use RWKV — first impressions" with honest, nuanced take:
- Text-LLM chat/code/RAG on modern GPU hardware: no advantage found over Transformers in our testing
- Lossy state is a real limitation for exact recall, but acceptable for summarization, style continuation, sentiment tagging
- Single low-budget GPU with many concurrent users: no KV-cache = more requests fit in VRAM (small but real advantage)
- Weak CPU + long sequences + tiny model: RWKV's linear complexity beats quadratic attention on speed (quality is another story)
- Truly constrained hardware (low RAM, CPU-only): constant memory regardless of sequence length is the only workable option — 256K Transformer KV-cache needs gigabytes
- Bottom line: the genuine niche is weak CPU + very long sequences + tiny model. On modern GPU or decent ARM with enough RAM, quantized Transformers win on every practical metric
- Clearly mark: these are first impressions, RWKV could still be an interesting alternative in some cases
- Clearly mark: these are from discussion, not from our own benchmarks

Add section: "Not yet explored: training and fine-tuning":
- We only tested inference so far — training/fine-tuning is completely untested
- RWKV reportedly has real training advantages: linear memory (vs O(n^2) for Transformers), infctx mode (128K+ fine-tuning on single 24GB GPU), state tuning (unique, extremely cheap)
- RWKV-PEFT project exists with LoRA, PiSSA, State Tuning, quantization support
- Needs hands-on testing — these are not our findings

Add section: "Watch: RWKV-8 and ROSA":
- RWKV-8 "Heron" is experimental, under development
- ROSA (Rapid Online Suffix Automaton) directly targets the lossy state problem
- Neurosymbolic approach: VQ-VAE discretization + suffix automaton for lossless retrieval of distant context
- Early results promising (99%+ accuracy on exact-recall tasks) but only on tiny models
- If it works at scale: eliminates the main argument against RWKV. Big "if" for now

### 7. Update `docs/lessons_learned.md`

Add entries (AI-facing lessons only — NOT conclusions or findings):
- "Always check model README for recommended sampling parameters" — we used default llama-cpp-python sampling instead of the model author's recommended settings, which affected output quality
- "RWKV-Runner web mode strips critical UI tabs" — Configs/Models/Downloads/Train/About are hidden, only Chat + Completion + Settings visible. Discovered after setup, wasted time looking for missing features

### 8. Update `AI_INSTRUCTIONS.md`

- Remove Dockerfile, docker-compose.yml, .dockerignore from hierarchy
- Remove RWKV-Runner from hierarchy and description
- Remove GGUF subfolder from Models (GGUF files stay on disk, just not referenced)
- Update project description to reflect current state (Sprint 2 done, Runner dropped)

### 9. Update `README.md`

- Remove RWKV-Runner from Resources table
- Update status/description
- Keep GGUF models reference (they're still useful via llama.cpp directly)

### 10. Update `roadmap.md`

- Sprint 2: mark completed items, mark Runner-specific items as cancelled with reason
- Sprint 3: replace with "Further investigation" noting preliminary conclusions need deeper research
- Update status table

### 11. Update `todo_2026-02-22.md`

- Mark Sprint 2 execution items appropriately (done/cancelled)
- Archive to `archive/2026-02-22_todo.md`

### 12. Archive Sprint 2 plan

```bash
mv claude_plans/PLAN_sprint2_rwkv_runner_docker.md archive/2026-02-22_PLAN_sprint2_rwkv_runner_docker.md
```

## Files to modify

| File | Action |
|------|--------|
| `Dockerfile` | Delete |
| `docker-compose.yml` | Delete |
| `.dockerignore` | Delete |
| `RWKV-Runner/` | Delete (rm -rf, not in git) |
| `.gitignore` | Remove RWKV-Runner line |
| `docs/inference_results.md` | Add Sprint 2 GGUF/Docker findings |
| `docs/inference_guide.md` | Add GGUF section + "When to use RWKV" |
| `docs/lessons_learned.md` | Add 2 new lessons (AI-facing only) |
| `AI_INSTRUCTIONS.md` | Remove Runner/Docker from hierarchy |
| `README.md` | Remove Runner references |
| `roadmap.md` | Update Sprint 2/3 status |
| `todo_2026-02-22.md` | Update and archive |
| `claude_plans/PLAN_sprint2_rwkv_runner_docker.md` | Move to archive/ |

## Verification

1. `RWKV-Runner/` directory is gone
2. No Docker files in root
3. `docker ps` shows no rwkv-runner container
4. All docs accurately reflect findings without overstatement
5. No broken references in AI_INSTRUCTIONS.md or README.md
6. Conclusions documented as preliminary, not final
