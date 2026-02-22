# Plan: Sprint 2 — RWKV-Runner on Linux/Docker with dual-GPU

## Context

Sprint 1 is complete: we understand RWKV-7 internals, tested all three inference modes, benchmarked all models (0.1B–7.2B, 13.3B OOM'd in fp16). Now we shift to practical usage — running RWKV as a chatbot with an OpenAI-compatible API via RWKV-Runner, containerized in Docker.

## Key research findings

### RWKV-Runner
- The Python backend (`backend-python/`) is a FastAPI server wrapping the same `rwkv` pip package we already use
- OpenAI-compatible API: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, streaming SSE
- Can run standalone without the Golang GUI: `python backend-python/main.py --port 8000`
- State caching built in: reuses RNN state across requests sharing a prefix
- Tool calling via prompt engineering (not native)
- Has a **llama.cpp backend** that auto-detects GGUF files by extension — switches to `llama-cpp-python` for inference
- Only runs RWKV models — not Gemma Recurrent, Llama, or other architectures
- **Existing Dockerfile uses CUDA 11.6 — way too old for our GPUs. Must rebuild.**

### Two inference paths — GGUF is the primary path

| Path | Format | Backend | Quantization | Multi-GPU | RNN state |
|------|--------|---------|-------------|-----------|-----------|
| **GGUF (primary)** | `.gguf` | llama.cpp | Q4–Q8, FP16 | Native GPU offloading (proven) | True O(1) recurrent state — no KV cache |
| Native (learning) | `.pth` | rwkv pip package | fp16/fp32/bf16 only | Strategy split (unproven mixed GPUs) | True O(1) recurrent state |

**Why GGUF is primary:** llama.cpp implements a proper `llama_memory_recurrent` system for RWKV — it does NOT use a KV-cache. The recurrent state is fixed-size, O(1) memory regardless of conversation length. All RNN advantages are preserved. Confirmed by reading the llama.cpp source:
- `llama-arch.cpp`: RWKV7 explicitly listed in `llm_arch_is_recurrent()`
- `llama-model.cpp`: creates `llama_memory_recurrent` instead of `llama_kv_cache` for RWKV
- `models/rwkv7-base.cpp`: state is read from fixed buffer, updated by `ggml_rwkv_wkv7`, written back — never grows
- `llama-memory-recurrent.cpp`: "models like Mamba or RWKV can't have a state partially erased... their state isn't preserved for previous tokens"

GGUF adds quantization, proven multi-GPU, and ecosystem compatibility (Ollama, LM Studio) — with zero RNN disadvantages.

### GGUF models available (13.3B)

Pre-quantized RWKV-7 GGUF: `huggingface.co/shoumenchougou/RWKV7-G1d-13.3B-GGUF`

| Quant | Size | Fits RTX 4090 (24 GB)? | Fits both GPUs (40 GB)? |
|-------|------|------------------------|------------------------|
| Q4_K_M | 8.4 GB | Yes, plenty | — |
| Q5_K_M | 10.0 GB | Yes | — |
| Q6_K | 11.6 GB | Yes | — |
| Q8_0 | 14.7 GB | Yes (similar to 7.2B fp16) | — |
| FP16 | 26.7 GB | No | Yes (24+16=40 GB) |

### Multi-GPU (from our other projects)

- **llama_cpp** (`/vibe_claude_kilo_cli_exp/llama_cpp/`): Docker with `nvidia/cuda:13.0.0-devel`, `-DCMAKE_CUDA_ARCHITECTURES="89;120"`, `count: all`. Already proven with both GPUs (RTX 4090 sm_89 + RTX 5070 Ti sm_120).
- **local-media-gen** (`/vibe_claude_kilo_cli_exp/local-media-gen/`): Docker with `nvidia/cuda:12.8.1-cudnn-devel`, PyTorch nightly cu128, `device_ids: ["0,1"]`
- For GGUF: reuse the llama_cpp approach (CUDA 13.0, dual-arch compilation). This is proven territory.

### Reference files in llama_cpp project (if things go wrong)

These files contain battle-tested solutions for our exact hardware. Consult them when debugging Docker or GPU issues:

| File | What it solves |
|------|----------------|
| `/vibe_claude_kilo_cli_exp/llama_cpp/Dockerfile` | Multi-stage build: CUDA 13.0, sm_89+sm_120, static llama-server binary. Our Dockerfile should mirror the base image and CUDA arch flags. |
| `/vibe_claude_kilo_cli_exp/llama_cpp/docker-compose.yml` | GPU passthrough (`count: all`), volume mounts, env var patterns, healthcheck. Template for our compose file. |
| `/vibe_claude_kilo_cli_exp/llama_cpp/docs/gpu-strategy-guide.md` | Decision tree for GPU placement: single GPU vs tensor split vs layer split. Key insights: fill CUDA0 (4090) first, use `--split-mode layer` for multi-GPU, `MAIN_GPU=0`, graph splits affect performance. |
| `/vibe_claude_kilo_cli_exp/llama_cpp/AI_INSTRUCTIONS.md` | Hardware specs: CUDA0=RTX 4090 24GB (primary), CUDA1=RTX 5070 Ti 16GB (~12.5 usable due to display/OS). |

Key lessons from llama_cpp:
- **CUDA1 (5070 Ti) has ~12.5 GB usable**, not 16 — display/OS takes ~3.5 GB
- **Layer split is sequential**: `total_time = CUDA0_time + transfer_time + CUDA1_time`. More layers on the faster GPU (4090) = faster.
- **`N_GPU_LAYERS=99`** offloads all layers to GPU (llama.cpp distributes automatically)
- **`SPLIT_MODE=layer`** is the default and works for dense models (RWKV is dense, not MoE)
- **`-ub 512`** is the safe default for compute buffer size

## Plan

### Step 1: Build custom Docker image

Create `docker/Dockerfile` and `docker/docker-compose.yml` in our project (not modifying RWKV-Runner/).

The image supports both inference paths but GGUF/llama.cpp is the primary one:

Base: `nvidia/cuda:13.0.0-devel-ubuntu24.04` (same as llama_cpp project — needed for sm_120 compilation)

The image will:
- Install Python 3.12, gcc, ninja
- Install PyTorch nightly cu128 (for native rwkv path + sm_120 support)
- Install `llama-cpp-python` with CUDA, compiled for `CMAKE_CUDA_ARCHITECTURES="89;120"`
- Install RWKV-Runner Python backend dependencies
- Mount model weights and RWKV-Runner as volumes (not copied into image)
- Expose port 8000 for the API

docker-compose.yml:
- Expose both GPUs: `device_ids: ["0,1"]` (same as local-media-gen)
- Mount `Models/` and `RWKV-Runner/` as volumes
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Set `RWKV_CUDA_ON=1` and `RWKV_V7_ON=1`

### Step 2: Download 13.3B GGUF models

Download from `huggingface.co/shoumenchougou/RWKV7-G1d-13.3B-GGUF` into `Models/rwkv7-g1/`:
- `rwkv7-g1d-13.3b-Q8_0.gguf` (14.7 GB) — best quality that fits single RTX 4090
- `rwkv7-g1d-13.3b-20260131-ctx8192-FP16.gguf` (26.7 GB) — for dual-GPU full precision

### Step 3: Test single-GPU GGUF (13.3B Q8 on RTX 4090)

This is the primary test. Load 13.3B Q8 via the llama.cpp backend:
```bash
curl -X POST http://localhost:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/rwkv7-g1d-13.3b-Q8_0.gguf"}'
```

Verify: model loads, API responds, streaming works. Measure speed and compare quality against our 7.2B fp16 benchmark. Q8 should be very close to full precision quality.

### Step 4: Test dual-GPU GGUF (13.3B FP16 across both GPUs)

Load the FP16 GGUF using llama.cpp's native GPU offloading — same mechanism as our llama_cpp project.

If RWKV-Runner's llama.cpp backend doesn't expose multi-GPU layer offload options, we can alternatively run our existing llama-server directly with the GGUF file (already built and working in `/vibe_claude_kilo_cli_exp/llama_cpp/`).

### Step 5: Test native path (7.2B .pth on RTX 4090)

Secondary test — verify the native rwkv pip package path also works through the Docker container:
```bash
curl -X POST http://localhost:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/rwkv7-g1d-7.2b-20260131-ctx8192.pth", "strategy": "cuda:0 fp16", "customCuda": true}'
```

Compare speed against our Sprint 1 benchmark (~51 tok/s).

### Step 6: Test the chat experience

With the best model loaded (likely 13.3B Q8 or FP16):
- Multi-turn conversation via API (test context retention — this is where the RNN state matters)
- Compare response quality: 13.3B Q8 GGUF vs 7.2B native fp16 vs 13.3B FP16 dual-GPU
- Test state caching behavior
- Try the built-in tool calling format

### Step 7: Document findings

- Update `docs/inference_results.md` with Docker/API/GGUF benchmark data
- Update `docs/inference_guide.md` with GGUF section (explain that llama.cpp preserves RNN state) and Docker usage
- Update `docs/setup_guide.md` with Docker setup instructions
- Update roadmap

## Files to create/modify

| File | Action |
|------|--------|
| `docker/Dockerfile` | Create — custom image with llama-cpp-python (dual-arch) + PyTorch nightly |
| `docker/docker-compose.yml` | Create — dual-GPU config, volumes, ports |
| `docs/inference_results.md` | Update — Docker/API/GGUF benchmark results |
| `docs/inference_guide.md` | Update — add GGUF explanation + Docker section |
| `docs/setup_guide.md` | Update — add Docker setup instructions |
| `AI_INSTRUCTIONS.md` | Update — add `docker/` to hierarchy, GGUF models to Models/ |
| `roadmap.md` | Update — check off Sprint 2 items |

## Out of scope (for this sprint)

- **Golang GUI**: We use the Python backend directly.
- **WebGPU/Rust backend**: Not relevant for our Linux CUDA setup.
- **Other model architectures** (Gemma Recurrent, Llama, etc.): RWKV-Runner only runs RWKV models.

## Future: RWKV-7 fine-tuning (optional, separate sprint)

Fine-tuning v7 is possible via `RWKV-LM/RWKV-v7/train_temp/train.py` — already in our cloned repo. Not through RWKV-Runner (which only supports v4/v5/v6 LoRA).

**What it is:**
- Full fine-tuning (all weights), not LoRA/PEFT adapter-based
- Workflow: rename `.pth` checkpoint to `rwkv-init.pth`, run `train.py` with `--train_stage 3 --lr_init 1e-5`
- Works on a single consumer GPU (min 10 GB VRAM for L12-D768/0.1B)

**What it needs:**
- Separate environment: `pytorch-lightning==1.9.5` (old, conflicts with our current env) + DeepSpeed
- Data in binary indexed format (`.bin` + `.idx`), not raw text — conversion step needed
- Training CUDA kernel (`rwkv7_clampw`) is heavier than inference kernel
- WandB for logging

**For actual LoRA (parameter-efficient):**
- Not in the upstream RWKV-LM repo
- Would need a third-party project like RWKV-PEFT on GitHub
- To investigate when/if we get to fine-tuning

## Verification

1. `docker compose up` starts successfully, API is reachable at `localhost:8000`
2. `/switch-model` loads 13.3B Q8 GGUF on single RTX 4090
3. `/v1/chat/completions` returns coherent streaming responses
4. Multi-turn conversation maintains context (RNN state works)
5. 7.2B `.pth` native path also works through the container
6. (Dual-GPU) 13.3B FP16 GGUF loads across both GPUs via llama.cpp offloading
