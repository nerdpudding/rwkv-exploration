# Plan: Sprint 2 — RWKV-Runner on Linux/Docker with optional dual-GPU

## Context

Sprint 1 is complete: we understand RWKV-7 internals, tested all three inference modes, benchmarked all models (0.1B–7.2B, 13.3B OOM'd). Now we shift to practical usage — running RWKV as a chatbot with an OpenAI-compatible API via RWKV-Runner, containerized in Docker.

## Key research findings

### RWKV-Runner
- The Python backend (`backend-python/`) is a FastAPI server wrapping the same `rwkv` pip package we already use
- OpenAI-compatible API: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, streaming SSE
- Can run standalone without the Golang GUI: `python backend-python/main.py --port 8000`
- State caching built in: reuses RNN state across requests sharing a prefix
- Tool calling via prompt engineering (not native)
- Only runs RWKV models — not Gemma Recurrent, Llama, or other architectures
- **Existing Dockerfile uses CUDA 11.6 — way too old for our GPUs. Must rebuild.**

### Multi-GPU (from our other projects)
- **llama_cpp** (`/vibe_claude_kilo_cli_exp/llama_cpp/`): Docker with `nvidia/cuda:13.0.0-devel`, `-DCMAKE_CUDA_ARCHITECTURES="89;120"`, `count: all`
- **local-media-gen** (`/vibe_claude_kilo_cli_exp/local-media-gen/`): Docker with `nvidia/cuda:12.8.1-cudnn-devel`, PyTorch nightly cu128, `device_ids: ["0,1"]`
- Both projects expose both GPUs and work on driver 580.x

### RWKV-7 limitations
- **No int8 quantization**: `RWKV_x070` explicitly rejects `i8` strategies. Only `fp16`, `fp32`, `bf16`
- **No v7 fine-tuning** in RWKV-Runner (only v4/v5/v6 LoRA)
- **Multi-GPU splitting exists in code** via strategy strings: `cuda:0 fp16 *N -> cuda:1 fp16`

### Dual-GPU risk assessment (be honest)
Multi-GPU splitting with our mixed architecture (RTX 4090 sm_89 + RTX 5070 Ti sm_120) is **unproven for RWKV**:
- In llama_cpp it works because we compile explicitly for both architectures
- In local-media-gen it works because different models go to different GPUs (not one model split)
- For RWKV, it would be one model split across two GPUs with **JIT-compiled CUDA kernels on two different architectures** — never tested
- The JIT compiler may need CUDA 13.0 (not just 12.8) for sm_120 kernel compilation
- Even if kernels compile, PCIe transfers between GPUs will cost speed
- It might simply not work — treat this as an experiment, not a guarantee

## Plan

### Step 1: Build custom Docker image

Create `docker/Dockerfile` and `docker/docker-compose.yml` in our project (not modifying RWKV-Runner/).

Base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04` (same approach as local-media-gen)

The image will:
- Install Python 3.12, gcc, ninja (for CUDA kernel JIT)
- Install PyTorch nightly cu128 (for sm_89 + sm_120 support)
- Install RWKV-Runner Python backend dependencies (`backend-python/requirements.txt`)
- Mount model weights and RWKV-Runner as volumes (not copied into image)
- Expose port 8000 for the API

docker-compose.yml:
- Expose both GPUs: `device_ids: ["0,1"]`
- Mount `Models/` and `RWKV-Runner/` as volumes
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- Set `RWKV_CUDA_ON=1` and `RWKV_V7_ON=1`

### Step 2: Test single-GPU first (7.2B on RTX 4090 only)

This is the primary goal. Start the container, load 7.2B with strategy `cuda:0 fp16`:
```bash
curl -X POST http://localhost:8000/switch-model \
  -H "Content-Type: application/json" \
  -d '{"model": "/models/rwkv7-g1d-7.2b-20260131-ctx8192.pth", "strategy": "cuda:0 fp16", "customCuda": true}'
```

Test the OpenAI-compatible API:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is gravity?"}], "stream": true}'
```

Verify: model loads, API responds, streaming works, speed is comparable to our benchmark (~51 tok/s).

### Step 3: Test the chat experience (single GPU)

With 7.2B loaded on RTX 4090:
- Multi-turn conversation via API (test context retention)
- Compare response quality to our earlier benchmark observations
- Test state caching (send same prefix, verify faster response)
- Try the built-in tool calling format

### Step 4: Experiment — dual-GPU with 13.3B (may fail)

**This is an experiment, not a requirement.** If Step 2+3 work, we already have a usable setup.

The 13.3B model has 61 layers. Split strategy attempt:
```
cuda:0 fp16 *37 -> cuda:1 fp16
```
(~37 layers on 4090/24GB, ~24 layers on 5070 Ti/16GB. Adjust based on actual VRAM.)

Possible outcomes:
- **Works**: Great, we can run the biggest model. Document the layer split and speed.
- **Kernel compilation fails on sm_120**: Try CUDA 13.0 base image instead of 12.8. If that also fails, document as unsupported.
- **Crashes or produces garbage**: Document and move on. 7.2B on single GPU is already a good result.

### Step 5: Document findings

- Update `docs/inference_results.md` with Docker/API benchmark data
- Update `docs/inference_guide.md` with Docker usage section
- Update `docs/setup_guide.md` with Docker setup instructions
- Update roadmap

## Files to create/modify

| File | Action |
|------|--------|
| `docker/Dockerfile` | Create — custom image for RWKV-Runner backend |
| `docker/docker-compose.yml` | Create — GPU config, volumes, ports |
| `docs/inference_results.md` | Update — Docker/API benchmark results |
| `docs/inference_guide.md` | Update — add Docker usage section |
| `docs/setup_guide.md` | Update — add Docker setup instructions |
| `AI_INSTRUCTIONS.md` | Update — add `docker/` to hierarchy |
| `roadmap.md` | Update — check off Sprint 2 items |

## Out of scope (for this sprint)

- **GGUF/quantized models**: v7 int8 not supported. GGUF conversion for v7 unconfirmed.
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
2. `/switch-model` loads 7.2B on single GPU, `/v1/chat/completions` returns coherent responses
3. Streaming responses work via SSE
4. Multi-turn conversation maintains context
5. (Experiment) If dual-GPU works: `/switch-model` loads 13.3B across both GPUs
