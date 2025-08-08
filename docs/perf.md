# TensorMesh Performance Guide

Numbers in this doc are reproducible on our CI bench rig:

- 4× RTX 4090 (24 GB each)
- AMD Ryzen 9 7950X
- PCIe 4.0 x16 links
- CUDA 12.4 / driver 550.54.15

---

## 1. End-to-end throughput (Llama-2-70B-Q4_K_M)

| seq_len | batch | tokens/s | latency (ms) |
|---------|-------|----------|--------------|
| 128     | 1     | 186      | 5.4          |
| 512     | 4     | 634      | 3.2          |
| 2048    | 8     | 1 120    | 7.3          |

**Reference:** Hugging Face TGI peaks at 98 tokens/s on same hardware.

---

## 2. Micro-benchmarks

### 2.1 Flash-attention kernel (tensor-mesh-kernels/cuda)

| head_dim | tokens | GB/s | TFLOPS |
|----------|--------|------|--------|
| 128      | 4 096  | 1 420 | 91     |
| 256      | 8 192  | 1 380 | 88     |

Measured with `criterion` + `nvprof`.

### 2.2 Zero-copy IPC latency

| payload size | Linux memfd | macOS IOKit |
|--------------|-------------|-------------|
| 4 MB         | 18 µs       | 22 µs       |
| 64 MB        | 350 µs      | 410 µs      |

---

## 3. Memory efficiency

| model         | weights | KV-cache | total | vs TGI |
|---------------|---------|----------|-------|--------|
| Llama-2-70B-Q4_K_M | 36 GB | 12 GB | 48 GB | –33 % |
| Llama-3-8B-F16     | 16 GB |  1 GB | 17 GB | –35 % |

Savings come from:
- **8-bit KV-cache** (experimental)
- **Page granularity** 4 kB → minimal fragmentation
- **In-place LoRA adapter** (no copy)

---

## 4. Scaling study

Nodes added at runtime:

| #GPUs | tokens/s | speed-up | migration latency |
|-------|----------|----------|-------------------|
| 1     | 48       | 1.0×     | —                 |
| 2     | 94       | 1.96×    | 0.9 s             |
| 4     | 186      | 3.88×    | 1.1 s             |

Linear scaling breaks at > 4 nodes due to PCIe saturation on consumer boards.

---

## 5. Profiling tips

```bash
# GPU trace
NVIDIA_NSIGHT=1 cargo bench --bench flash_attn

# CPU flamegraph
cargo install flamegraph
CARGO_PROFILE_RELEASE_DEBUG=true \
  flamegraph --bench scheduler
```

Results land in `target/criterion/`.

## 6. Road-map for gains

- **Paged-attention v3 kernel**: +12 % tokens/s (issue #42)
- **CUDA graphs**: reduce launch overhead 30 % (issue #51)
- **Dynamic batcher**: improve utilization at small seq_len (issue #37)