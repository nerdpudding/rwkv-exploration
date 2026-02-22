# Project Description: RWKV-7 RNN Exploration

## Origin

This project started from a conversation exploring the differences between RNN and Transformer-based LLMs. The key insight was that while Transformers won the NLP race thanks to parallel self-attention and long context windows, RNNs have a compelling niche: constant memory usage and constant time per token, regardless of context length. Modern RNNs like RWKV-7 bridge the gap by training in parallel (GPT-mode) but running inference as a pure RNN — no KV-cache, no growing memory footprint.

RWKV-7 "Goose" was identified as the current state-of-the-art in this space. It introduces a "meta-in-context learning" mechanism where the model performs a mini gradient-descent step on its own state at each token, effectively adapting to context during inference without additional training. The architecture is claimed to recognize all regular languages and perform state tracking — capabilities that standard Transformers cannot match under equivalent complexity assumptions.

The practical appeal: where a Transformer model's VRAM usage grows with context length (due to KV-cache), RWKV's stays flat. For long conversations or streaming use cases, this could be a significant advantage — especially on consumer hardware.

## Goal

Learn how RNN-based language models differ from Transformer-based LLMs through hands-on experimentation with RWKV-7 "Goose" (G1), the current state-of-the-art pure RNN model.

## Background

The developer has extensive practical experience running Transformer-based LLMs locally (llama.cpp, ollama, vLLM, TensorRT-LLM, SGLang) but zero experience with RNN architectures. The typical workflow involves quantized GGUF models with llama.cpp to split layers across multiple GPUs and manage KV-cache limitations. The key question is: how does RWKV's fundamentally different architecture (constant memory, no KV-cache, linear scaling) change the inference story?

## Hardware

- GPU 1: NVIDIA RTX 4090 (24 GB VRAM, fully available)
- GPU 2: NVIDIA RTX 5070 Ti (16 GB VRAM, ~12 GB available after OS/display)
- CPU: AMD 5800X3D
- RAM: 64 GB
- OS: Ubuntu Desktop (Linux)

## Scope

1. **Understand the architecture**: Learn what makes RWKV-7 fundamentally different from Transformers — constant memory usage, no KV-cache, RNN-mode vs GPT-mode inference, the "in-context learning" mechanism.

2. **Get it running locally**: Set up a working environment (conda or uv first to understand the mechanics, potentially Docker later) and run inference with the available model weights (.pth format, ranging from 0.1B to 13.3B parameters).

3. **Explore inference modes**: Run and compare the different demo scripts (GPT-mode, RNN-mode, fast hybrid mode) to understand the trade-offs between prefill speed and token generation.

4. **Compare with Transformer LLMs**: Evaluate chat quality, speed, and resource usage against familiar Transformer models at comparable parameter counts. Understand where RWKV excels (long context, constant memory, streaming) and where it falls short (reasoning depth at small scales).

5. **Understand the ecosystem**: Learn what tools exist for RWKV inference (the `rwkv` pip package, Albatross engine, the demo scripts) and how they compare — without assuming Transformer-era tooling (GGUF, llama.cpp) is needed or appropriate here.

## What's Already Available

- Cloned repo: `RWKV-LM` (includes RWKV-v7 demo scripts + CUDA kernels)
- Model weights: `Models/rwkv7-g1/` containing G1 models from 0.1B to 13.3B (.pth format)
- Tokenizer vocabulary file included in the repo

## Approach

- Start small (0.1B or 0.4B) to understand the mechanics and verify the setup works
- Scale up to 2.9B or 7.2B for meaningful quality comparisons
- Prefer understanding how things work over using GUI wrappers
- Docker is the eventual preference, but first learn the raw setup to understand what's happening under the hood

## Non-Goals

- Training or fine-tuning (for now)
- Production deployment
- Windows support (Ubuntu only)
- Using GUI wrappers like RWKV-Runner
