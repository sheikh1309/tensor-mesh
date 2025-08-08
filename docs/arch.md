# TensorMesh Architecture Deep-Dive

This document explains the internal design of TensorMesh.  
It is targeted at kernel authors, scheduler contributors, and anyone who wants to reason about correctness or performance.

---

## 1. High-level flow

1. **Model ingestion**  
   `tensor-mesh-model` parses safetensors / GGUF → produces weights + tokenizer.

2. **Graph lowering**  
   Weights + config → `tensor-mesh-core::Graph` (device-agnostic IR).

3. **Scheduling**  
   `tensor-mesh-scheduler` maps subgraphs to `DeviceHandle`s while minimizing PCIe/NVLink traffic.

4. **Kernel dispatch**  
   Each `Node` is executed by the matching backend in `tensor-mesh-kernels`.

5. **Zero-copy IPC**  
   Output tensors are exposed as `tensor-mesh-ipc::SharedTensor`; peer devices can map them read-only without copies.

---

## 2. Core IR (`tensor-mesh-core`)

### 2.1 Tensor
- Row-major, dense, **immutable** once created.
- `Arc<[u8]>` payload enables shared-memory hand-off.
- Alignment guarantees (256 B) for SIMD & CUDA coalescing.

### 2.2 Graph
- Static DAG, SSA style.
- Nodes: `MatMul`, `RmsNorm`, `RoPE`, `Softmax`, `Add`, `Mul`, etc.
- Metadata: `requires_grad=false`, `quant_scheme`, `cache_hint`.

---

## 3. Device abstraction

```rust
pub trait Device: Send + Sync {
    fn id(&self) -> DeviceId;
    fn memory(&self) -> MemoryStats;
    fn compile(&self, node: &Node) -> Result<Executable>;
}
```

Implementations live in `tensor-mesh-kernels`:

• CudaDevice
• MetalDevice  
• CpuDevice
• WebGpuDevice (future)

## 4. Scheduler internals

### 4.1 Cost model

• PCIe bandwidth table
• Kernel launch latency
• Memory pressure heuristics

### 4.2 Algorithm

• Greedy list scheduler with back-tracking (≤ 100 ms per layer).
• Falls back to ILP solved with `good_lp` crate for > 8 devices.

### 4.3 Migration protocol

• Uses `tensor-mesh-ipc::LeaseToken` (RAII)
• Lease expires after 5 s idle → tensor unmapped → remote can reclaim VRAM.

## 5. Zero-copy IPC (`tensor-mesh-ipc`)

| OS | Mechanism |
|----|-----------|
| Linux | `memfd_create` + `cudaIpcOpenMemHandle` |
| macOS | `mach_make_memory_entry_64` + `IOGPU` |
| Windows | `CreateFileMapping` + `cudaIpcOpenMemHandle` |

All paths expose `SharedTensor { fd: RawFd, offset: u64, len: u64 }` over gRPC.

## 6. LoRA hot-swap

Weights are **reference-counted blobs**.  
LoRA adapter = list of `ΔW = A·B` low-rank matrices.  
Scheduler swaps adapter nodes **in-place** without rebuilding the full graph.

## 7. KV-cache sharing

• Each attention head is **shared round-robin** across devices.
• Cache pages stored as `tensor-mesh-ipc::PagedTensor` (4 kB pages).
• Page-table replicated on every node to avoid centralized metadata.

## 8. Security & safety

• No `unsafe` in core crates; kernels wrap vendor APIs with `unsafe` minimized to FFI boundary.
• **Memory mapping** uses sealed memfd on Linux → prevents host ptr over-read.
• **Sandbox**: each backend runs in its own process; scheduler talks over Unix domain sockets.

## 9. Extending TensorMesh

- Add new ops: implement `Device::compile` and register in `tensor-mesh-kernels`.
- Add new devices: impl `Device` trait + expose in `tensor-mesh-scheduler/DeviceHandle::discover`.