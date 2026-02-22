# PLAN: Environment Setup & First Inference Test

## Context

Sprint 1 starts now. The project setup is complete (docs, agents, repo cloned, models downloaded). Before we can run anything, we need a working Python environment with PyTorch + CUDA, and then verify it works by running inference with the 0.1B model. This plan covers everything up to and including the first successful inference run.

## System Status (researched by agents)

| Component | Status |
|-----------|--------|
| GPU | RTX 4090 (24 GB), GPU index 0, almost fully free |
| CUDA toolkit | 12.6.3 installed, nvcc available |
| GPU driver | 580.126.16 (supports up to CUDA 13.0) |
| gcc/g++ | 13.3.0 (compatible with CUDA 12.6) |
| ninja | 1.11.1 (speeds up JIT kernel compilation) |
| conda | 24.11.3 installed, 9 existing envs |
| uv | 0.9.22 installed |
| Python (system) | 3.12.3 |

## Step 1: Create conda environment

```bash
conda create -n rwkv7 python=3.12 -y
conda activate rwkv7
```

Why Python 3.12: fully supported by PyTorch 2.10.0, matches system Python (3.12.3), no known incompatibilities with RWKV scripts.

Why conda over uv: already heavily used on this machine, manages CUDA_HOME and library paths cleanly for JIT compilation.

## Step 2: Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy rwkv
```

- PyTorch 2.10.0 (latest stable, released Jan 2026) with CUDA 12.6 wheels — matches system CUDA toolkit exactly.
- `rwkv` pip package (v0.8.32) — official high-level inference API, alternative to raw demo scripts (roadmap Sprint 1 item 5).
- `torchvision`/`torchaudio` not needed for RWKV inference.

## Step 3: Verify the setup

Target RTX 4090 only. Set environment variables explicitly:

```bash
conda activate rwkv7
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected: `2.10.0`, `True`, `NVIDIA GeForce RTX 4090`.

- `CUDA_VISIBLE_DEVICES=0` — explicitly target RTX 4090 only, ignore 5070 Ti for now.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — reduces memory fragmentation during longer sessions (pattern from local-media-gen project on same hardware).

Also verify JIT compilation prerequisites:
```bash
python -c "from torch.utils.cpp_extension import load; print('cpp_extension available')"
```

## Step 4: Create our own inference scripts

The demo scripts in `RWKV-LM/` have hardcoded paths and are part of an external cloned repo. We do NOT edit files in `RWKV-LM/` or `Models/` — those directories are gitignored and managed separately.

**Approach:** Create our own scripts in a `scripts/` directory at the project root. These scripts are based on the demo scripts but adapted for our setup (model paths, architecture params, environment variables). Our scripts live in our git repo; the cloned repo stays untouched as a reference.

**Key challenge:** The demo scripts use relative paths for:
- Tokenizer: `rwkv_vocab_v20230424.txt`
- CUDA kernels: `cuda/wkv7_op.cpp`, `cuda/wkv7.cu`
- LAMBADA data: `misc/lambada_test.jsonl`

Our scripts must use absolute paths to these resources in `RWKV-LM/RWKV-v7/`, or set the working directory appropriately. We also need to add `RWKV-LM/RWKV-v7/` to `sys.path` so that any shared modules resolve.

### Scripts to create in `scripts/`:

**`scripts/run_rnn.py`** (start with RNN-mode — simplest, no CUDA kernel needed):
- Based on `RWKV-LM/RWKV-v7/rwkv_v7_demo_rnn.py`
- Absolute path to model: `Models/rwkv7-g1/rwkv7-g1d-0.1b-20260129-ctx8192` (no .pth)
- Absolute path to tokenizer vocab file
- Architecture params: `n_layer=12`, `n_embd=768` for 0.1B
- Environment setup at top (CUDA_VISIBLE_DEVICES, PYTORCH_CUDA_ALLOC_CONF)

**`scripts/run_gpt.py`** (GPT-mode):
- Based on `RWKV-LM/RWKV-v7/rwkv_v7_demo.py`
- Absolute paths to model (.pth), tokenizer, CUDA kernel sources
- Verify/adjust LoRA dimension params for g1d variant

**`scripts/run_hybrid.py`** (hybrid-mode):
- Based on `RWKV-LM/RWKV-v7/rwkv_v7_demo_fast.py`
- Same absolute path approach

## Step 5: Run first inference — RNN-mode with 0.1B

Start with the RNN-mode script because it's the simplest (no custom CUDA kernel compilation needed):

```bash
conda activate rwkv7
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_rnn.py
```

**What to expect:**
- torch.compile (or torch.jit.script fallback) will compile the time_mixing function
- Token-by-token generation with the 0.1B model
- LAMBADA evaluation runs automatically at the end

## Step 6: Run GPT-mode and hybrid-mode

After RNN-mode works:

```bash
python scripts/run_gpt.py       # GPT-mode — will JIT-compile wkv7 CUDA kernel (30-60s first time)
python scripts/run_hybrid.py    # Hybrid — will JIT-compile wkv7s CUDA kernel
```

The CUDA kernel compilation happens once; subsequent runs use cached binaries.

## Step 7: Record findings

- Note VRAM usage, token speed, any issues
- Update `docs/lessons_learned.md` with setup findings
- Update `roadmap.md` to mark completed items
- Update `todo_2026-02-22.md`

## Potential Issues

1. **CUDA_HOME:** If JIT compilation fails, may need `export CUDA_HOME=/usr/local/cuda-12.6`
2. **g1d vs g1a model variant:** The scripts were written for g1a models. Our models are g1d. The weight format should be the same, but LoRA dimensions might differ. If loading fails, we'll need to inspect the weight keys and adjust params.
3. **LAMBADA eval:** Runs automatically at end of GPT and RNN scripts. Not optional without editing. We'll let it run — it's a useful benchmark.

## Verification

Success criteria for this plan:
- [ ] `conda activate rwkv7` works, torch sees the RTX 4090
- [ ] `rwkv_v7_demo_rnn.py` generates coherent text with the 0.1B model
- [ ] `rwkv_v7_demo.py` compiles CUDA kernel and generates text
- [ ] `rwkv_v7_demo_fast.py` compiles CUDA kernel and generates text
- [ ] Findings documented in lessons_learned.md
