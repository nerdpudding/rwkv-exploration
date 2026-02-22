########################################################################################################
# RWKV-7 inference via the official rwkv pip package (PIPELINE API)
# Cleaner API than raw demo scripts, with built-in repetition penalties
########################################################################################################

import os, sys, time
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

########################################################################################################
# Paths
########################################################################################################

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "Models", "rwkv7-g1", "rwkv7-g1d-0.1b-20260129-ctx8192")
TOKENIZER_NAME = "rwkv_vocab_v20230424"

########################################################################################################
# Helpers
########################################################################################################

def vram_mb():
    return torch.cuda.memory_allocated() / 1024**2

def vram_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def run_test(name, pipeline, prompt, token_count, args):
    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(f"\nPrompt: {prompt}")
    print(f"Output: {prompt}", end="")

    token_counter = [0]
    def callback(x):
        token_counter[0] += 1
        print(x, end="", flush=True)

    torch.cuda.reset_peak_memory_stats()
    vram_before = vram_mb()
    t0 = time.perf_counter()

    pipeline.generate(prompt, token_count=token_count, args=args, callback=callback)

    torch.cuda.synchronize()
    t1 = time.perf_counter()
    vram_after = vram_mb()
    vram_peak = vram_peak_mb()

    elapsed = t1 - t0
    toks = token_counter[0]
    tok_s = toks / elapsed if elapsed > 0 else 0

    print(f"\n[{toks} tokens, {elapsed:.2f}s, {tok_s:.1f} tok/s, VRAM: {vram_after:.0f} MB (peak {vram_peak:.0f} MB)]")

########################################################################################################
# Load model + pipeline
########################################################################################################

print(f"Loading model: {MODEL_PATH}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM before load: {vram_mb():.0f} MB")

torch.cuda.reset_peak_memory_stats()
model = RWKV(model=MODEL_PATH, strategy="cuda fp16")
pipeline = PIPELINE(model, TOKENIZER_NAME)

print(f"VRAM after load:  {vram_mb():.0f} MB (peak {vram_peak_mb():.0f} MB)")

########################################################################################################
# Sampling configs
########################################################################################################

args_with_penalties = PIPELINE_ARGS(
    temperature=1.0,
    top_p=0.85,
    alpha_frequency=0.2,
    alpha_presence=0.2,
    token_stop=[0],
)

args_no_penalties = PIPELINE_ARGS(
    temperature=1.0,
    top_p=0.85,
    alpha_frequency=0.0,
    alpha_presence=0.0,
    token_stop=[0],
)

########################################################################################################
# Tests
########################################################################################################

run_test(
    "Test 1: Generation WITH repetition penalties",
    pipeline, "The Eiffel tower is in the city of",
    token_count=100, args=args_with_penalties,
)

run_test(
    "Test 2: Generation WITHOUT repetition penalties",
    pipeline, "The Eiffel tower is in the city of",
    token_count=100, args=args_no_penalties,
)

run_test(
    "Test 3: Chat prompt with repetition penalties",
    pipeline, "User: What is 2+2?\n\nAssistant:",
    token_count=200, args=args_with_penalties,
)

run_test(
    "Test 4: Think prompt with repetition penalties",
    pipeline, "User: Explain why the sky is blue.\n\nAssistant: <think>",
    token_count=300, args=args_with_penalties,
)

########################################################################################################

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Model: {os.path.basename(MODEL_PATH)}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM steady state: {vram_mb():.0f} MB")
print(f"VRAM peak (all tests): {vram_peak_mb():.0f} MB")
print(f"Done.")
