# RFC-0001: Distributed Backend

| Field | Value |
|---|---|
| RFC | 0001 |
| Title | Distributed Backend |
| Author(s) | Codex |
| Status | Implemented |
| Created | 2026-08-02 |
| Updated | 2026-08-02 |

## Summary

`orca-distributed` provides the current TCP-based distributed communication layer for Orca. It keeps network handling outside the tensor core, moves tensor payloads through explicit framing, and exposes a minimal `all_reduce` primitive for data-parallel training.

## Motivation

Orca needs a distributed boundary that can synchronize tensors across multiple worker processes without pushing network concerns into `orca-core` or `orca-tensor`. The implementation also needs to fail cleanly, avoid unsafe byte reinterpretation, and preserve backend independence when tensors move in and out of the distributed layer.

## Detailed Design

The crate centers on `DistributedCommunicator`:

- `rank` identifies the local worker.
- `world_size` defines the total process group size.
- `master_addr` provides the rendezvous endpoint.
- `streams` stores connected TCP streams behind a mutex, initialized lazily.

Connection setup uses a simple master/worker topology:

- Rank 0 binds the rendezvous address and accepts `world_size - 1` workers.
- Non-zero ranks retry connection attempts with a short backoff.
- Streams are configured with `TCP_NODELAY` to reduce latency for small control messages.

Tensor exchange uses explicit framing and float conversion:

- Inputs are normalized to `f32` before transport.
- The sender writes the element count as a big-endian `u64`.
- Payload data is serialized as big-endian `f32` bytes.
- The receiver validates the expected element count before reading the payload.
- Results are reconstructed back onto the original backend and dtype.

Error handling stays in `Result` space:

- Mutex poisoning becomes `OrcaError::InternalError`.
- Socket bind, accept, connect, read, and write failures all map to `OrcaError`.
- Shape/count mismatches return `OrcaError::ShapeMismatch`.

The current public primitive is:

- `all_reduce<B: Backend>(&self, tensor: &Tensor<B>) -> Result<Tensor<B>>`

Its behavior is:

1. Validate `rank` and `world_size`.
2. Initialize connections on first use.
3. Convert the tensor to `f32` if needed.
4. Rank 0 gathers values, sums elementwise, and broadcasts the result.
5. Non-zero ranks send their payload and then read back the reduced tensor.

## Alternatives Considered

### Unsafe raw-byte transport

Rejected because it couples the wire format to in-memory layout, makes endianness implicit, and invites UB or silent corruption.

### NCCL-only integration

Rejected for the current baseline because it hard-codes a GPU-centric dependency chain and excludes the CPU-only distributed case this crate currently supports.

## Drawbacks

- Only `f32` is transported on the wire today.
- The implementation is blocking and uses a master/worker rendezvous model.
- The crate currently exposes only `all_reduce`.
- Non-f32 tensors incur conversion overhead before and after transport.

## Prior Art

- PyTorch `torch.distributed`: broader API surface, multiple collective types, and backend-specific transports.
- NCCL: high-performance GPU collectives, but narrower in scope than the current crate.
- Horovod: higher-level distributed training orchestration with a heavier runtime footprint.

## Unresolved Questions

- Should the next collective be `broadcast`, `all_gather`, or `reduce_scatter`?
- Should the wire format remain TCP-first or add a GPU-direct transport path?
- How should process-group rendezvous be externalized for multi-node deployments?
- Should non-f32 payloads be serialized natively instead of round-tripping through `f32`?

## Future Possibilities

- Additional collectives such as `broadcast`, `all_gather`, and `reduce_scatter`.
- Fault-tolerant rendezvous and restart semantics.
- Backend-specific transport accelerators.
- Higher-level distributed training APIs in `orca-nn`.

## Implementation Plan

The baseline implementation already exists in `orca-distributed/src/lib.rs`.

1. Keep the current TCP `all_reduce` path as the supported baseline.
2. Add coverage for additional dtypes and edge cases around rendezvous and transport errors.
3. Add new collectives only through follow-up RFCs so the API surface stays deliberate.
