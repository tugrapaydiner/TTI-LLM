# 🚀 TTI-LLM

**Test-Time Inference for Large Language Models**

*A from-scratch GPT-2 inference engine in pure Python + NumPy*

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-2.0+-green.svg)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/tests-17%20passed-brightgreen.svg)](#test-results)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

TTI-LLM is a **minimal, educational, and production-ready** implementation of GPT-2 inference entirely from scratch using Python and NumPy. No PyTorch required at runtime! This project demonstrates deep understanding of transformer architecture, quantization techniques, KV-caching, and speculative decoding.

---

## 📋 Table of Contents

- [Features](#-features)
- [Quick Start](#-quick-start)
- [Benchmark Results](#-benchmark-results)
- [Quantization](#-quantization)
- [Speculative Decoding](#-speculative-decoding)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Pure NumPy Inference** | No PyTorch/TensorFlow at runtime – just NumPy |
| **HuggingFace Weights** | Loads official GPT-2 weights from HuggingFace Hub |
| **KV-Cache** | Efficient autoregressive decoding with key-value caching |
| **Int8 Quantization** | Per-row symmetric quantization with 2x memory reduction |
| **Int4 Quantization** | Groupwise 4-bit quantization for MLP layers (2.3x memory reduction) |
| **Speculative Decoding** | DistilGPT-2 as draft model for parallel token verification |
| **Numba JIT Kernels** | Optional JIT-compiled GEMV kernels for acceleration |
| **CLI Interface** | Easy command-line text generation |

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/tugrapaydiner/TTI-LLM.git
cd TTI-LLM
pip install -r requirements.txt

# Generate text (FP32)
python src/run_generate.py --prompt "Hello, my name is" --tokens 50

# Generate with Int8 (2x less memory)
python src/run_generate.py --prompt "Once upon a time" --tokens 100 --mode int8

# Generate with Int4 (2.3x less memory)
python src/run_generate.py --prompt "The future of AI" --tokens 100 --mode int4
```

---

## 📊 Benchmark Results

All benchmarks run on **Intel Core i7-1355U** with **Python 3.13** and **NumPy 2.2.6**.

### Memory Efficiency

Quantization dramatically reduces memory footprint while maintaining model quality:

<img width="2969" height="1771" alt="memory_usage" src="https://github.com/user-attachments/assets/81e160a7-b856-482e-9fb7-d7bfe0457c6d" />

| Mode | Memory (MB) | Compression | Reduction |
|------|-------------|-------------|-----------|
| **FP32** | 474.7 | 1.00x | baseline |
| **Int8** | 231.9 | 2.05x | **51% smaller** |
| **Int4** | 208.2 | 2.28x | **56% smaller** |

Int8 quantization achieves a 2x memory reduction with less than 1% accuracy loss. Int4 pushes compression further to 2.3x with ~11% error (only applied to MLP layers).

---

### Inference Throughput

Performance comparison across different quantization modes:

<img width="3570" height="1771" alt="decode_throughput" src="https://github.com/user-attachments/assets/9e4335e6-4b44-475c-96ec-5274154bff6a" />

| Mode | 16 tokens | 32 tokens | 64 tokens | Avg tok/s |
|------|-----------|-----------|-----------|-----------|
| **FP32** | 1.71 | 1.71 | 1.70 | **1.71** |
| **Int8** | 1.43 | 1.42 | 1.38 | **1.41** |
| **Int4** | 1.15 | 1.13 | 1.16 | **1.15** |

**Key Insights:**
- Int8 delivers 83% of FP32 speed with 2x less memory
- Int4 achieves 67% of FP32 speed with 2.3x memory savings
- Performance remains consistent across different sequence lengths

---

### Prefill Performance

Initial sequence processing time for different prompt lengths:

<img width="3570" height="1771" alt="prefill_latency" src="https://github.com/user-attachments/assets/800ead50-e7c2-4784-9d7f-25027f835eba" />

Lower prefill latency means faster response time for the first token. FP32 provides the lowest latency, while quantized modes trade slightly higher prefill time for reduced memory usage.

---

### Memory-Speed Tradeoff

Visual analysis of the quantization tradeoff between memory usage and throughput:

<img width="3573" height="2369" alt="memory_speed_tradeoff" src="https://github.com/user-attachments/assets/01b99a28-297b-4d0b-8b23-9ce157f0fd21" />

This chart clearly shows the quantization spectrum from FP32 (high speed, high memory) through Int8 (balanced) to Int4 (low memory, reduced speed). Choose your mode based on your deployment constraints.

---

## 🔧 Quantization

### Int8 Quantization

Per-row symmetric quantization: `W_quant = round(W / scale)` where `scale = max(|W|) / 127`

**Properties:**
- ✅ 2.05x memory reduction
- ✅ <1% relative error
- ✅ Applied to all linear layers
- ✅ Negligible quality degradation

### Int4 Quantization

Groupwise 4-bit quantization with group size 64, packing two int4 values per byte.

**Properties:**
- ✅ 2.28x memory reduction
- ⚠️ ~11% relative error
- ⚠️ Only applied to MLP layers
- ✅ Attention layers remain Int8

### Error Analysis

Quantization accuracy across different weight matrix shapes:

<img width="4171" height="1770" alt="quantization_error" src="https://github.com/user-attachments/assets/afafce9b-c9ac-424b-ab2a-9fd31daa3060" />

| Weight Shape | Int8 Error | Int4 Error |
|--------------|------------|------------|
| 768 × 3072 | 0.92% | 11.4% |
| 3072 × 768 | 0.83% | 11.4% |
| 768 × 768 | 0.84% | 11.5% |

Int8 maintains excellent precision with <1% error. Int4 shows consistent ~11% error, which is acceptable for MLP layers where accuracy requirements are lower.

---

## ⚡ Speculative Decoding

Uses DistilGPT-2 (6 layers) as draft model to propose multiple tokens, verified in parallel by GPT-2 (12 layers).

**How it works:** Draft model proposes k tokens → Main model verifies all tokens in one pass → Accept until first mismatch → Continue from main model's prediction

### Performance Results

<img width="4171" height="1768" alt="speculative_decoding" src="https://github.com/user-attachments/assets/5c0d5dc3-df3f-49fd-9355-2059727d853b" />

| Method | Throughput | Acceptance Rate | Speedup |
|--------|-----------|-----------------|---------|
| **Baseline** | 1.67 tok/s | N/A | 1.00x |
| **Spec k=2** | 0.92 tok/s | 80.8% | 0.55x |
| **Spec k=4** | 0.88 tok/s | 90.3% | 0.53x |
| **Spec k=6** | 0.81 tok/s | 84.9% | 0.49x |

**Note:** High acceptance rates (~80-90%) validate the approach, but sequential CPU execution creates overhead. GPU parallelization would unlock 1.5-2x speedups.

---
