# Setup Guide

Full environment setup for running RWKV-7 inference locally. For a quick start, see the [README](../README.md#quick-start).

## Prerequisites

| Requirement | Minimum | Our setup |
|-------------|---------|-----------|
| NVIDIA GPU | Any with CUDA support | RTX 4090 (24 GB) |
| CUDA toolkit | 12.6+ | 12.6.3 |
| GPU driver | 550+ | 580.126.16 |
| gcc/g++ | 12+ | 13.3.0 |
| ninja | any | 1.11.1 |
| conda | any | 24.11.3 |

gcc and ninja are needed for CUDA kernel JIT compilation (GPT-mode and hybrid-mode scripts).

## External dependencies (not in git)

### RWKV-LM repository

```bash
cd /path/to/rwkv-exploration
git clone https://github.com/BlinkDL/RWKV-LM.git
```

Contains demo scripts, CUDA kernel sources, tokenizer vocabulary, and LAMBADA test data. We reference it but never modify it.

### Model weights

Download from [HuggingFace: BlinkDL/rwkv7-g1](https://huggingface.co/BlinkDL/rwkv7-g1) into `Models/rwkv7-g1/`.

Available models:

| Model | File | Size | Architecture |
|-------|------|------|-------------|
| 0.1B g1d | `rwkv7-g1d-0.1b-20260129-ctx8192.pth` | 382 MB | 12 layers, 768 embd |
| 0.4B g1d | `rwkv7-g1d-0.4b-20260210-ctx8192.pth` | 902 MB | 24 layers, 1024 embd |
| 1.5B g1d | `rwkv7-g1d-1.5b-20260212-ctx8192.pth` | 3.1 GB | 24 layers, 2048 embd |
| 2.9B g1d | `rwkv7-g1d-2.9b-20260131-ctx8192.pth` | 5.9 GB | 32 layers, 2560 embd |
| 7.2B g1d | `rwkv7-g1d-7.2b-20260131-ctx8192.pth` | 14.4 GB | 32 layers, 4096 embd |
| 13.3B g1d | `rwkv7-g1d-13.3b-20260131-ctx8192.pth` | 26.5 GB | 61 layers, 4096 embd |

`head_size` is always 64 for all models.

## Environment setup

### Create conda environment

```bash
conda create -n rwkv7 python=3.12 -y
conda activate rwkv7
```

### Install PyTorch with CUDA 12.6

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy rwkv
```

This installs PyTorch 2.10.0 with CUDA 12.6 wheels. No torchvision/torchaudio needed.

### Verify

```bash
conda activate rwkv7
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected: 2.10.0+cu126, True, NVIDIA GeForce RTX 4090

python -c "from torch.utils.cpp_extension import load; print('cpp_extension available')"
# Expected: cpp_extension available
```

## Environment variables

Set these before running any script:

| Variable | Value | Why |
|----------|-------|-----|
| `CUDA_VISIBLE_DEVICES` | `0` | Target RTX 4090 only. Prevents PyTorch from touching the RTX 5070 Ti (Blackwell/sm_120, needs different CUDA support). |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Reduces VRAM fragmentation during longer sessions. Low cost, prevents subtle OOM after extended use. |

Our scripts in `scripts/` set these automatically via `os.environ` at the top.

## Running inference

### Our scripts (learning — step by step)

```bash
conda activate rwkv7
python scripts/run_rnn.py      # RNN-mode — simplest, no CUDA kernel
python scripts/run_gpt.py      # GPT-mode — compiles wkv7 kernel first time
python scripts/run_hybrid.py   # Hybrid — compiles wkv7s kernel first time
```

Each script loads the 0.1B model by default, generates 500 tokens, then runs LAMBADA evaluation. The LAMBADA eval can be interrupted with Ctrl+C.

To switch model sizes, edit the script and update:
- `MODEL_NAME` or `MODEL_PATH` (the weight file path)
- `args.n_layer` and `args.n_embd` (see model table above)

### rwkv pip package (convenience — cleaner API)

```python
import os
os.environ['RWKV_V7_ON'] = '1'        # must be set BEFORE import
os.environ['RWKV_CUDA_ON'] = '1'       # optional: enables CUDA WKV kernel
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

model = RWKV(model='Models/rwkv7-g1/rwkv7-g1d-0.1b-20260129-ctx8192.pth',
             strategy='cuda fp16')
pipeline = PIPELINE(model, 'RWKV-LM/RWKV-v7/rwkv_vocab_v20230424')

args = PIPELINE_ARGS(temperature=1.0, top_p=0.85,
                     alpha_frequency=0.2, alpha_presence=0.2,
                     token_stop=[0])
pipeline.generate("User: Hello!\n\nAssistant:", token_count=200, args=args,
                   callback=lambda x: print(x, end='', flush=True))
```

The `PIPELINE` adds repetition penalties (alpha_frequency, alpha_presence) and chunked prefill, which our raw scripts don't have.

## Troubleshooting

### CUDA kernel compilation fails

If GPT-mode or hybrid-mode fails during kernel compilation:

1. Check `CUDA_HOME` is set: `echo $CUDA_HOME` — should be `/usr/local/cuda-12.6` or similar
2. Check nvcc is available: `nvcc --version`
3. Check gcc version: `gcc --version` — must be compatible with your CUDA toolkit version

### Model loading fails or produces garbage

If output is nonsensical after loading a model:

1. Check you updated both the model path AND architecture params (n_layer, n_embd)
2. Check the `.pth` extension convention: RNN and hybrid scripts expect path **without** `.pth` (appended in code), GPT script expects path **with** `.pth`
3. Inspect weight keys: `torch.load('model.pth', map_location='cpu').keys()` and compare against what the script expects

### Wrong GPU selected

If PyTorch uses the wrong GPU, verify `CUDA_VISIBLE_DEVICES` is set before any torch import. Our scripts set this via `os.environ` at the very top.
