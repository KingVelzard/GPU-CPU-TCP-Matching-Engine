# Heterogeneous CPU-GPU Limit Order Book Engine

A high-performance limit order book engine that processes **2M+ TCP orders/sec** by splitting work between the CPU and GPU based on workload shape — sequential matching on CPU, parallel book maintenance on GPU.

Built in C++ and CUDA on Linux.

---

## What It Does

A limit order book is the core data structure of any financial exchange — it tracks all resting buy and sell orders at every price level and matches incoming orders against them in real time. At scale, this requires careful systems engineering: low-latency networking, efficient data structures, and parallel computation across many instruments simultaneously.

This engine handles the full pipeline:

- Receives orders over TCP from multiple clients simultaneously
- Classifies each order as aggressive (immediately executable) or passive (rests in the book)
- Routes aggressive orders to the CPU matcher for immediate execution
- Batches passive orders and sends them to the GPU for parallel book updates
- Publishes market data snapshots back to clients after each update cycle

---

## Architecture

The key design decision is the CPU/GPU split:

**CPU owns matching.** Order matching is sequential by nature — each fill depends on the previous one. The CPU handles this with a flat price-level array and a doubly-linked list for fast level traversal, executing fills in microseconds.

**GPU owns the book surface.** Adding and cancelling passive orders across multiple instruments is embarrassingly parallel. The GPU runs one thread block per instrument simultaneously, applying updates via atomic operations and maintaining the top price levels in shared memory.

The two sides communicate through pinned (page-locked) memory buffers that both processors can access directly — no explicit data transfers required.

```
TCP Clients
    │
    ▼
Reactor Threads (×8, core-pinned)
    │
    ├──── Aggressive Orders ──── CPU Matcher ──── Fill Confirmations ──── TCP Clients
    │                                │
    │                         Match Commands
    │                                │
    └──── Passive Orders ─────────── ▼
                              GPU Dispatch Thread
                                     │
                                     ▼
                              GPU Kernel (one block per instrument)
                                     │
                              Book Deltas + Market Data Snapshots
                                     │
                                     ▼
                              CPU applies deltas, publishes snapshots
```

---

## Key Technical Work

**Lock-free networking.** 8 epoll reactors each pinned to a dedicated CPU core, using edge-triggered events and non-blocking I/O. Orders arrive over TCP and land directly in GPU-accessible pinned memory — no intermediate copies.

**Lock-free MPSC queue written from scratch.** Replaced a mutex-based queue with a sequence-number-based MPSC ring buffer. Producers (reactor threads) claim slots via CAS; the consumer (GPU dispatch thread) reads without any locking. Careful use of acquire/release memory ordering throughout.

**Seqlock-protected market data.** The GPU writes spread snapshots to pinned memory using a seqlock — a lightweight synchronization primitive that lets the CPU classifier read consistent bid/ask pairs at full speed without ever blocking the GPU.

**Two-stream GPU pipeline.** Near-spread orders fire on a high-priority CUDA stream every 50µs. Far-from-spread orders batch on a low-priority stream every 500µs. A CUDA event guard prevents the two streams from accessing the book simultaneously.

**CPU-GPU state synchronization.** The CPU and GPU each maintain their own copy of the book, kept in sync through two command logs — match commands (CPU→GPU) and book deltas (GPU→CPU). Applied asynchronously via stream callbacks, with idempotent delta writes to handle any retry scenarios.

---

## Performance

| Metric | Value |
|---|---|
| TCP throughput | 2M+ orders/sec |
| Reactors | 8 (one per core) |
| Aggressive order latency | Low microseconds (CPU path only) |
| Passive order batch interval | 50µs (fast) / 500µs (slow) |
| Instruments supported | 8 simultaneous |
| Top price levels cached | 50 per side per instrument |

---

## Stack

| Layer | Technology |
|---|---|
| Language | C++20, CUDA |
| Networking | Linux epoll, POSIX sockets |
| GPU | CUDA streams, atomics, shared memory, pinned memory |
| Profiling | Nsight Systems, Nsight Compute |
| Build | CMake |
| Platform | Linux x86-64 |

---

## Project Structure

```
include/          Core data types (OrderMessage, Feed, MatchCommand, BookDelta, ...)
src/              CPU-side implementation
  ├── cpu/        Classifier and Matcher
  ├── gpu/        GPU kernel, dispatch thread, book layout
  ├── Reactor     Networking layer
  └── Server      Top-level wiring
tests/            Correctness tests (seqlock, MPSC queue, convergence, GPU book diff)
```

---

## Tests

- **MPSC queue** — 8 producers × 100K items, zero loss verified
- **Seqlock** — writer at 20K/sec, reader in tight loop, zero torn reads
- **Convergence** — 200K operations through the full system vs. CPU reference, zero book divergence
- **GPU book diff** — element-by-element comparison of GPU and CPU book state after 10K operations
