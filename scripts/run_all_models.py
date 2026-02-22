########################################################################################################
# RWKV-7 benchmark: test all available models via the rwkv pip package
# Runs each model with a generation test and a chat test, reports speed and VRAM
########################################################################################################

import os, sys, time, gc
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

########################################################################################################
# Config
########################################################################################################

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "Models", "rwkv7-g1")
TOKENIZER_NAME = "rwkv_vocab_v20230424"

# All g1d models, ordered by size
MODELS = [
    {"name": "0.1B",  "file": "rwkv7-g1d-0.1b-20260129-ctx8192"},
    {"name": "0.4B",  "file": "rwkv7-g1d-0.4b-20260210-ctx8192"},
    {"name": "1.5B",  "file": "rwkv7-g1d-1.5b-20260212-ctx8192"},
    {"name": "2.9B",  "file": "rwkv7-g1d-2.9b-20260131-ctx8192"},
    {"name": "7.2B",  "file": "rwkv7-g1d-7.2b-20260131-ctx8192"},
    {"name": "13.3B", "file": "rwkv7-g1d-13.3b-20260131-ctx8192"},
]

ARGS = PIPELINE_ARGS(
    temperature=1.0,
    top_p=0.85,
    alpha_frequency=0.2,
    alpha_presence=0.2,
    token_stop=[0],
)

GPU_NAME = torch.cuda.get_device_name(0)
GPU_TOTAL_MB = torch.cuda.get_device_properties(0).total_memory / 1024**2

########################################################################################################
# Helpers
########################################################################################################

def vram_mb():
    return torch.cuda.memory_allocated() / 1024**2

def vram_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def run_generation(pipeline, prompt, token_count):
    """Run generation, return (output_text, token_count, elapsed, tok_s)"""
    tokens = [0]
    chunks = []

    def callback(x):
        tokens[0] += 1
        chunks.append(x)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    pipeline.generate(prompt, token_count=token_count, args=ARGS, callback=callback)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    tok_s = tokens[0] / elapsed if elapsed > 0 else 0
    text = "".join(chunks)
    return text, tokens[0], elapsed, tok_s

def unload_model():
    """Free GPU memory between models"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

########################################################################################################
# Main
########################################################################################################

print("=" * 70)
print(f"RWKV-7 All Models Benchmark")
print(f"GPU: {GPU_NAME} ({GPU_TOTAL_MB:.0f} MB)")
print(f"PyTorch: {torch.__version__}")
print(f"Strategy: cuda fp16")
print("=" * 70)

results = []

for m in MODELS:
    model_path = os.path.join(MODELS_DIR, m["file"])
    pth_path = model_path + ".pth"
    disk_mb = os.path.getsize(pth_path) / 1024**2 if os.path.exists(pth_path) else 0

    print(f"\n{'─' * 70}")
    print(f"Model: {m['name']} ({m['file']})")
    print(f"Disk size: {disk_mb:.0f} MB")
    print(f"{'─' * 70}")

    if not os.path.exists(pth_path):
        print(f"SKIPPED — file not found: {pth_path}")
        continue

    # Estimate if it fits (rough: disk size ≈ VRAM in fp16)
    if disk_mb > GPU_TOTAL_MB * 0.9:
        print(f"WARNING — disk size ({disk_mb:.0f} MB) close to GPU total ({GPU_TOTAL_MB:.0f} MB), may OOM")

    try:
        unload_model()

        # Load model
        print("Loading...", end=" ", flush=True)
        torch.cuda.reset_peak_memory_stats()
        t_load_start = time.perf_counter()
        model = RWKV(model=model_path, strategy="cuda fp16")
        pipeline = PIPELINE(model, TOKENIZER_NAME)
        t_load_end = time.perf_counter()
        load_time = t_load_end - t_load_start

        vram_loaded = vram_mb()
        vram_load_peak = vram_peak_mb()
        print(f"done ({load_time:.1f}s, VRAM: {vram_loaded:.0f} MB, peak: {vram_load_peak:.0f} MB)")

        # Test 1: Knowledge / factual generation
        print("\n  [Test 1] Factual generation (100 tokens):")
        prompt1 = "The Eiffel tower is in the city of"
        text1, toks1, time1, speed1 = run_generation(pipeline, prompt1, 100)
        print(f"  Prompt: {prompt1}")
        print(f"  Output: {prompt1}{text1[:200]}{'...' if len(text1) > 200 else ''}")
        print(f"  [{toks1} tokens, {time1:.2f}s, {speed1:.1f} tok/s]")

        # Test 2: Chat with think
        print(f"\n  [Test 2] Chat + think (200 tokens):")
        prompt2 = "User: Explain gravity in one paragraph.\n\nAssistant: <think>"
        text2, toks2, time2, speed2 = run_generation(pipeline, prompt2, 200)
        # Show first 300 chars of output
        display = text2[:300].replace('\n', '\n  ')
        print(f"  Prompt: {prompt2}")
        print(f"  Output: {display}{'...' if len(text2) > 300 else ''}")
        print(f"  [{toks2} tokens, {time2:.2f}s, {speed2:.1f} tok/s]")

        # Test 3: Speed test (longer generation, no print overhead)
        print(f"\n  [Test 3] Speed test (500 tokens, silent):")
        prompt3 = "Once upon a time in a land far away,"
        torch.cuda.reset_peak_memory_stats()
        text3, toks3, time3, speed3 = run_generation(pipeline, prompt3, 500)
        vram_gen_peak = vram_peak_mb()
        print(f"  [{toks3} tokens, {time3:.2f}s, {speed3:.1f} tok/s, VRAM peak: {vram_gen_peak:.0f} MB]")

        result = {
            "name": m["name"],
            "disk_mb": disk_mb,
            "vram_loaded_mb": vram_loaded,
            "vram_peak_mb": vram_gen_peak,
            "load_time_s": load_time,
            "tok_s_factual": speed1,
            "tok_s_chat": speed2,
            "tok_s_speed": speed3,
        }
        results.append(result)

        # Clean up before next model
        del pipeline, model

    except torch.cuda.OutOfMemoryError:
        print(f"OUT OF MEMORY — skipping {m['name']}")
        # Make sure we clean up
        if 'model' in dir():
            del model
        if 'pipeline' in dir():
            del pipeline
        unload_model()
        continue
    except Exception as e:
        print(f"ERROR — {e}")
        if 'model' in dir():
            del model
        if 'pipeline' in dir():
            del pipeline
        unload_model()
        continue

########################################################################################################
# Summary table
########################################################################################################

print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"GPU: {GPU_NAME} ({GPU_TOTAL_MB:.0f} MB)")
print()

header = f"{'Model':<8} {'Disk':>8} {'VRAM':>8} {'Peak':>8} {'Load':>7} {'Factual':>9} {'Chat':>9} {'Speed':>9}"
print(header)
print(f"{'':─<8} {'(MB)':─>8} {'(MB)':─>8} {'(MB)':─>8} {'(s)':─>7} {'(tok/s)':─>9} {'(tok/s)':─>9} {'(tok/s)':─>9}")

for r in results:
    print(f"{r['name']:<8} {r['disk_mb']:>8.0f} {r['vram_loaded_mb']:>8.0f} {r['vram_peak_mb']:>8.0f} {r['load_time_s']:>7.1f} {r['tok_s_factual']:>9.1f} {r['tok_s_chat']:>9.1f} {r['tok_s_speed']:>9.1f}")

print("\nDone.")
