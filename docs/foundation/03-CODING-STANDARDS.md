# 03 — Coding Standards

> **Status**: Active  
> **Maintainer**: Core Team  
> **Last updated**: 2026-08-02
> **Applies to**: All code in the `orca-runtime` workspace and `orca` Python package

---

## Purpose

This document defines the authoritative coding standards for the Orca deep learning
framework. Every contributor — core maintainer, external contributor, or bot — must
follow these standards. Code that deviates from this document will be rejected during
review. Automated tooling enforces the mechanical rules; human reviewers enforce the
spirit.

The goals are:

1. **Consistency** — any file should look like it was written by the same person.
2. **Safety** — deep learning frameworks run in production; bugs are expensive.
3. **Performance** — zero-cost abstractions first, measured optimizations second.
4. **Readability** — code is read 10× more than it is written.

---

## 1  Rust Coding Standards

All Rust code lives under the `orca-runtime` Cargo workspace. Each sub-crate follows
the naming convention `orca-{name}` (e.g. `orca-tensor`, `orca-autograd`,
`orca-backend-gpu`).

### 1.1  Style and Formatting

We use `rustfmt` (nightly channel) with a project-level configuration. The canonical
config lives at the workspace root:

#### `rustfmt.toml`

```toml
# Orca — rustfmt configuration
# Requires: rustup run nightly cargo fmt

edition                       = "2024"
max_width                     = 100
hard_tabs                     = false
tab_spaces                    = 4
newline_style                 = "Unix"
use_small_heuristics          = "Default"

# Imports
imports_granularity           = "Crate"
group_imports                 = "StdExternalCrate"
reorder_imports               = true
reorder_modules               = true

# Items
fn_params_layout              = "Tall"
struct_lit_single_line         = true
enum_discrim_align_threshold  = 20
where_single_line             = false

# Comments & strings
wrap_comments                 = true
comment_width                 = 100
format_strings                = false
normalize_comments            = true
normalize_doc_attributes      = true

# Control flow
match_arm_blocks              = true
force_multiline_blocks        = false

# Misc
use_field_init_shorthand      = true
use_try_shorthand             = true
force_explicit_abi            = true
overflow_delimited_expr       = true
```

**Import ordering** is enforced by `group_imports = "StdExternalCrate"`, which
produces three blocks separated by blank lines:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use orca_tensor::Shape;
use crate::graph::Node;
```

The order is always: **std → external crates → workspace crates → `crate`/`self`/`super`**.

#### Clippy

We run `clippy` at the **deny** level in CI. The lint configuration lives in the
workspace `Cargo.toml`:

```toml
# Cargo.toml (workspace root)

[workspace.lints.clippy]
# ── Correctness ──────────────────────────────────────────
correctness       = { level = "deny" }
suspicious        = { level = "deny" }
perf              = { level = "deny" }

# ── Pedantic (warn in dev, deny in CI) ───────────────────
pedantic          = { level = "warn" }
cast_possible_truncation   = { level = "allow" }  # too noisy for numeric code
cast_sign_loss             = { level = "allow" }

# ── Nursery (selective) ─────────────────────────────────
missing_const_for_fn       = { level = "warn" }
cognitive_complexity        = { level = "warn" }
or_fun_call                = { level = "warn" }
redundant_pub_crate        = { level = "warn" }

# ── Restriction (selective) ─────────────────────────────
dbg_macro                  = { level = "deny" }
print_stdout               = { level = "deny" }
print_stderr               = { level = "deny" }
todo                       = { level = "warn" }
unimplemented              = { level = "deny" }
unwrap_used                = { level = "deny" }
expect_used                = { level = "warn" }
indexing_slicing            = { level = "warn" }
panic                      = { level = "warn" }

# ── Style ────────────────────────────────────────────────
module_name_repetitions    = { level = "allow" }
must_use_candidate         = { level = "allow" }
return_self_not_must_use   = { level = "allow" }

[workspace.lints.rust]
unsafe_op_in_unsafe_fn     = "deny"
missing_docs               = "warn"
```

Each sub-crate inherits these lints:

```toml
# orca-tensor/Cargo.toml
[lints]
workspace = true
```

> **Rule**: `cargo clippy --workspace --all-targets -- -D warnings` must pass with
> zero diagnostics before any PR is merged.

#### Module Organization

Each crate follows a predictable layout:

```
orca-tensor/
├── Cargo.toml
├── src/
│   ├── lib.rs          # public API re-exports, #![doc], crate-level config
│   ├── shape.rs        # one concept per file
│   ├── dtype.rs
│   ├── tensor/
│   │   ├── mod.rs      # sub-module roll-up
│   │   ├── create.rs   # creation ops (zeros, ones, rand)
│   │   ├── ops.rs      # element-wise operations
│   │   └── view.rs     # slicing, reshaping
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── cpu.rs
│   │   └── cuda.rs
│   └── error.rs        # OrcaError definition for this crate
├── tests/
│   ├── shape_tests.rs
│   └── tensor_ops.rs
└── benches/
    └── matmul.rs
```

Rules:

- One concept per file. If a file exceeds ~500 lines, split it.
- `mod.rs` exists only to re-export child modules. No logic in `mod.rs`.
- `lib.rs` is the public surface: `pub use` statements and `#![doc = ...]`.
- Integration tests go in `tests/`. Benchmarks go in `benches/`.

---

### 1.2  Naming Conventions

| Element              | Convention            | Example                                  |
|----------------------|-----------------------|------------------------------------------|
| Types (struct/enum)  | `PascalCase`          | `Tensor`, `ShapeMismatch`, `CpuBackend`  |
| Traits               | `PascalCase` (adjective/noun) | `Backend`, `Differentiable`, `Serializable` |
| Functions / methods  | `snake_case`          | `matmul`, `reshape`, `to_device`         |
| Constants            | `SCREAMING_SNAKE_CASE`| `MAX_TENSOR_RANK`, `DEFAULT_DTYPE`       |
| Crate names          | `orca-{name}` (kebab) | `orca-tensor`, `orca-autograd`           |
| Module names         | `snake_case`          | `tensor_ops`, `cpu_backend`              |
| Feature flags        | `kebab-case`          | `cuda-support`, `blas-accel`             |
| Type parameters      | Single uppercase      | `T` (type), `B` (backend), `D` (device), `S` (storage) |
| Lifetime parameters  | Short lowercase       | `'a`, `'ctx`, `'graph`                   |
| Enum variants        | `PascalCase`          | `Float32`, `CudaDevice`, `ShapeMismatch` |
| Builder methods      | `with_{field}`        | `with_dtype`, `with_device`              |
| Conversion traits    | `as_`/`to_`/`into_`   | `as_slice`, `to_vec`, `into_inner`       |
| Fallible conversions | `try_{verb}`          | `try_reshape`, `try_into`                |

**Generic parameter conventions for Orca**:

```rust
// T  — element type (f32, f64, bf16, etc.)
// B  — backend (CpuBackend, CudaBackend)
// D  — device specifier
// S  — storage implementation
// G  — gradient type
// N  — const-generic dimension count
fn matmul<T, B>(lhs: &Tensor<T, B>, rhs: &Tensor<T, B>) -> Result<Tensor<T, B>>
where
    T: Float,
    B: Backend,
{ ... }
```

---

### 1.3  Error Handling

Orca uses a unified error type rooted in `orca-core`:

```rust
use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum OrcaError {
    #[error("shape mismatch in `{operation}`: expected {expected}, got {got}")]
    ShapeMismatch {
        expected: Shape,
        got: Shape,
        operation: &'static str,
    },

    #[error("unsupported dtype `{dtype}` for operation `{operation}`")]
    UnsupportedDtype {
        dtype: DType,
        operation: &'static str,
    },

    #[error("device error on {device}: {message}")]
    DeviceError {
        device: DeviceId,
        message: String,
    },

    #[error("index {index} out of bounds for axis {axis} with size {size}")]
    IndexOutOfBounds {
        index: isize,
        axis: usize,
        size: usize,
    },

    #[error("autograd error: {message}")]
    AutogradError { message: String },

    #[error("serialization error: {0}")]
    SerializationError(String),

    #[error("{context}: {source}")]
    Internal {
        context: String,
        #[source]
        source: Box<OrcaError>,
    },
}

/// Convenience alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, OrcaError>;
```

**Rules**:

| # | Rule |
|---|------|
| 1 | All public functions return `Result<T, OrcaError>` (or the alias `Result<T>`). |
| 2 | **Never** use `.unwrap()` or `.expect()` in library code. Tests are exempt. |
| 3 | Error messages must be **user-friendly and actionable**. Include the operation name, shapes, dtypes, and any other context needed to diagnose the issue without reading source code. |
| 4 | Use the `Internal` variant to **chain context** onto lower-level errors. |
| 5 | Backend-specific crates may define their own error types, but they must convert into `OrcaError` via `From<BackendError> for OrcaError`. |
| 6 | Use `thiserror` for all error definitions. Do not implement `Display` or `Error` manually. |
| 7 | For operations with preconditions, prefer **returning `Err`** over panicking. Reserve panics for logic bugs (violated invariants). |

**Context pattern** (preferred):

```rust
pub fn reshape(&self, new_shape: &Shape) -> Result<Tensor<T, B>> {
    if self.shape().num_elements() != new_shape.num_elements() {
        return Err(OrcaError::ShapeMismatch {
            expected: new_shape.clone(),
            got: self.shape().clone(),
            operation: "reshape",
        });
    }
    // ...
}
```

---

### 1.4  Documentation

#### Requirements

| Scope           | Requirement |
|-----------------|-------------|
| Every `pub` item | Must have a `///` doc comment. |
| Every module     | Must have a `//!` header explaining purpose and giving a short example. |
| Complex algorithms | Must link to a paper or reference in the doc comment. |
| `pub(crate)` helpers | Optional but encouraged. |
| Internal-but-public items | Use `#[doc(hidden)]`. |

#### Doc Comment Structure

```rust
/// Computes the matrix product of two tensors.
///
/// Performs a batched matrix multiplication when inputs have more than 2
/// dimensions, broadcasting along batch dimensions following NumPy rules.
/// Delegates to BLAS where available, falling back to a tiled algorithm.
///
/// # Arguments
///
/// * `lhs` — Left-hand tensor of shape `[..., M, K]`
/// * `rhs` — Right-hand tensor of shape `[..., K, N]`
///
/// # Returns
///
/// A tensor of shape `[..., M, N]`.
///
/// # Errors
///
/// Returns [`OrcaError::ShapeMismatch`] if the inner dimensions are
/// incompatible (i.e. `lhs.shape[-1] != rhs.shape[-2]`).
///
/// # Examples
///
/// ```rust
/// use orca_tensor::Tensor;
///
/// let a = Tensor::<f32>::randn(&[2, 3]);
/// let b = Tensor::<f32>::randn(&[3, 4]);
/// let c = a.matmul(&b)?;
/// assert_eq!(c.shape(), &[2, 4]);
/// # Ok::<(), orca_core::OrcaError>(())
/// ```
///
/// # Performance
///
/// Time complexity: O(M·N·K) per batch element. Uses BLAS `sgemm`/`dgemm`
/// when the `blas-accel` feature is enabled.
pub fn matmul(&self, rhs: &Tensor<T, B>) -> Result<Tensor<T, B>> { ... }
```

**Module-level docs**:

```rust
//! # Tensor Creation Operations
//!
//! This module provides factory functions for creating tensors with
//! specific initialization patterns: zeros, ones, random normal,
//! random uniform, arange, linspace, and eye.
//!
//! All creation functions accept a [`Shape`] (or something convertible
//! to one) and return a [`Tensor`] on the default device.
//!
//! ```rust
//! use orca_tensor::Tensor;
//!
//! let zeros = Tensor::<f32>::zeros(&[3, 3]);
//! let eye   = Tensor::<f32>::eye(4);
//! ```
```

---

### 1.5  Testing

#### Test Organization

| Kind             | Location                   | Framework   |
|------------------|----------------------------|-------------|
| Unit tests       | `#[cfg(test)]` inline module | `#[test]` + standard assertions |
| Integration tests| `tests/` directory          | `#[test]`   |
| Property tests   | Inline or `tests/`         | `proptest`  |
| Benchmarks       | `benches/`                 | `criterion` |
| Fuzz tests       | `fuzz/`                    | `cargo-fuzz` (libFuzzer) |

#### Naming Convention

```
test_{function}_{scenario}_{expected_result}
```

Examples:

```rust
#[test]
fn test_reshape_compatible_shapes_returns_reshaped() { ... }

#[test]
fn test_reshape_incompatible_elements_returns_shape_mismatch() { ... }

#[test]
fn test_matmul_batched_broadcasts_correctly() { ... }
```

#### Property-Based Testing

All numerical code must have property tests validating algebraic identities:

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_transpose_involution(rows in 1usize..64, cols in 1usize..64) {
        let t = Tensor::<f32>::randn(&[rows, cols]);
        let tt = t.transpose()?.transpose()?;
        prop_assert!(t.allclose(&tt, 1e-6));
    }

    #[test]
    fn test_add_commutative(
        rows in 1usize..32,
        cols in 1usize..32,
    ) {
        let a = Tensor::<f32>::randn(&[rows, cols]);
        let b = Tensor::<f32>::randn(&[rows, cols]);
        let ab = (&a + &b)?;
        let ba = (&b + &a)?;
        prop_assert!(ab.allclose(&ba, 1e-6));
    }
}
```

#### Criterion Benchmarks

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use orca_tensor::Tensor;

fn bench_matmul_square(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul/square");
    for &n in &[64, 128, 256, 512, 1024] {
        group.bench_with_input(
            criterion::BenchmarkId::new("f32", n),
            &n,
            |b, &n| {
                let a = Tensor::<f32>::randn(&[n, n]);
                let rhs = Tensor::<f32>::randn(&[n, n]);
                b.iter(|| black_box(a.matmul(&rhs).unwrap()));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_matmul_square);
criterion_main!(benches);
```

#### Coverage

- **Target**: ≥ 80% line coverage for `orca-core`, `orca-tensor`, and `orca-autograd`.
- **Measurement**: `cargo llvm-cov --workspace --lcov --output-path lcov.info`.
- **Enforcement**: Coverage checks run in CI and block merges if the target is not met
  for core crates.
- **Exemptions**: Backend FFI glue and generated code may be excluded via
  `#[cfg(not(tarpaulin_include))]`.

#### Regression Tests

Every bug fix PR **must** include a regression test. The test must:

1. Reproduce the bug (fail without the fix).
2. Be named `test_regression_{issue_number}_{brief_description}`.
3. Reference the issue in a comment.

```rust
#[test]
fn test_regression_142_broadcast_scalar_rank0() {
    // Regression: https://github.com/orca-ml/orca/issues/142
    // Broadcasting a rank-0 tensor against a rank-2 tensor panicked.
    let scalar = Tensor::<f32>::from_scalar(2.0);
    let matrix = Tensor::<f32>::ones(&[3, 3]);
    let result = (&scalar + &matrix).expect("should not panic");
    assert_eq!(result.shape(), &[3, 3]);
}
```

---

### 1.6  Unsafe Code

Orca is a safe-by-default project. Unsafe code is permitted only when
strictly necessary for performance or FFI.

**Rules**:

1. Every `unsafe` block **must** have a `// SAFETY:` comment immediately above or
   inside it, explaining *why* the invariant holds.
2. Unsafe code must be encapsulated behind a safe public API.
3. New `unsafe` blocks require approval from **at least two** maintainers.
4. All `unsafe` usage is tracked in `docs/UNSAFE-AUDIT.md`.
5. `#![deny(unsafe_op_in_unsafe_fn)]` is enabled workspace-wide (see lint config).
6. Prefer safe alternatives: `bytemuck` for transmutes, `ndarray` for raw pointer
   arithmetic, safe wrappers from `orca-ffi`.

**Acceptable uses**:

| Use case                      | Example                               |
|-------------------------------|---------------------------------------|
| CUDA/cuDNN FFI calls          | `cuda_runtime_sys::cudaMemcpy(...)`   |
| BLAS FFI calls                | `cblas_sgemm(...)`                    |
| Performance-critical hot path | `get_unchecked` inside a bounds-checked loop |
| `Send`/`Sync` impls for FFI  | GPU stream handles                    |

**Example**:

```rust
/// Returns the element at `index` without bounds checking.
///
/// # Safety
///
/// The caller must ensure that `index < self.len()`.
pub unsafe fn get_unchecked(&self, index: usize) -> &T {
    // SAFETY: The caller has guaranteed `index < self.len()`.
    // The underlying buffer was allocated with `self.len()` elements
    // in `Storage::alloc`, so this pointer offset is in-bounds.
    unsafe { &*self.ptr.add(index) }
}
```

---

### 1.7  Performance

| # | Guideline |
|---|-----------|
| 1 | **Benchmark before optimizing.** No optimization without a `criterion` benchmark proving the improvement. Include before/after numbers in the PR description. |
| 2 | Use `#[inline]` only on small functions (≤ 5 lines) on hot paths. Never `#[inline(always)]` without benchmarks. |
| 3 | Prefer stack allocation for small, fixed-size data. Use `SmallVec<[T; 4]>` or `tinyvec` for dimension lists (most tensors are ≤ 4-D). |
| 4 | Document time complexity in doc comments for all public algorithms. |
| 5 | Avoid allocations in tight loops. Pre-allocate and reuse buffers. |
| 6 | Use `Iterator` chains over index-based loops — they optimize better. |
| 7 | Gate expensive runtime checks behind `debug_assertions` where safe. |
| 8 | For SIMD, prefer `std::simd` (portable) or `pulp` over hand-written intrinsics. |

**SmallVec example for shapes**:

```rust
use smallvec::SmallVec;

/// A tensor shape. Stores up to 4 dimensions inline (no heap allocation).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: SmallVec<[usize; 4]>,
}
```

---

### 1.8  Concurrency

Deep learning workloads are inherently parallel. Orca must be safe to use
from multiple threads.

| # | Rule |
|---|------|
| 1 | All core types (`Tensor`, `Graph`, `Module`) must be `Send + Sync`. |
| 2 | Use `Arc<T>` for shared ownership across threads. Never use `Rc<T>` in library code. |
| 3 | Prefer `RwLock` over `Mutex` when reads dominate (e.g. parameter storage). |
| 4 | **Never hold a lock across an `.await` point.** Use `tokio::sync::Mutex` if async locking is required. |
| 5 | Use `crossbeam` scoped threads or `rayon` for data parallelism. |
| 6 | Document thread-safety guarantees on every type that holds interior mutability. |
| 7 | Avoid `static mut`. Use `OnceLock` or `LazyLock` for global initialization. |

**Example — thread-safe parameter store**:

```rust
use std::sync::{Arc, RwLock};

/// A thread-safe container for model parameters.
///
/// # Thread Safety
///
/// `ParameterStore` is `Send + Sync`. Multiple threads may read
/// parameters concurrently via [`read`]. Exclusive write access is
/// acquired via [`write`] and blocks readers.
pub struct ParameterStore {
    inner: Arc<RwLock<HashMap<String, Tensor<f32>>>>,
}
```

---

### 1.9  Dependencies

| # | Rule |
|---|------|
| 1 | Every new dependency requires a justification comment in the PR. |
| 2 | Prefer well-maintained crates with ≥ 1 000 downloads/day and a recent release (< 6 months). |
| 3 | Pin **major** versions: `serde = "1"`, not `serde = "1.0.197"` and never `serde = "*"`. |
| 4 | No `*` version specifications — ever. |
| 5 | Run `cargo audit` weekly in CI. Any `RUSTSEC` advisory on a direct dependency blocks the release. |
| 6 | Run `cargo deny check` to enforce license compatibility (allow-list: MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib). |
| 7 | Vendored C/C++ code (e.g. oneDNN) lives under `vendor/` with its own LICENSE file and build script. |
| 8 | Optional heavy dependencies (CUDA, ROCm) must be behind feature flags. |

**`deny.toml`** (workspace root):

```toml
[advisories]
vulnerability = "deny"
unmaintained  = "warn"
yanked        = "deny"

[licenses]
unlicensed    = "deny"
allow         = [
    "MIT",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Zlib",
    "Unicode-3.0",
]
copyleft      = "deny"

[bans]
multiple-versions = "warn"
wildcards         = "deny"
```

---

## 2  Python Coding Standards

The `orca` Python package provides the user-facing API. It is built with
**PyO3** and distributed as a single wheel.

### 2.1  Style and Formatting

- **PEP 8** compliance is mandatory.
- **Formatter / linter**: `ruff` (replaces `black`, `isort`, `flake8`).
- **Maximum line length**: 100 characters (matching Rust).
- **Quotes**: double quotes (`"`).

#### `pyproject.toml` (ruff section)

```toml
[tool.ruff]
target-version = "py310"
line-length    = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "RUF",  # ruff-specific rules
    "D",    # pydocstyle
    "ANN",  # flake8-annotations
    "S",    # flake8-bandit (security)
    "PT",   # flake8-pytest-style
    "TCH",  # flake8-type-checking
]
ignore = [
    "D100",   # missing docstring in public module (handled by module header)
    "ANN101", # missing type annotation for `self`
    "ANN102", # missing type annotation for `cls`
    "D203",   # conflicts with D211
    "D213",   # conflicts with D212
]

[tool.ruff.lint.pydocstyle]
convention = "google"

[tool.ruff.lint.isort]
known-first-party = ["orca"]
force-sort-within-sections = true
```

### 2.2  API Design

The Python API should feel native to Python users, following the conventions
of **NumPy** and **PyTorch** where appropriate.

| Principle | Convention |
|-----------|-----------|
| Functions / methods | `snake_case` — `orca.zeros()`, `tensor.reshape()` |
| Classes | `PascalCase` — `orca.Tensor`, `orca.Module` |
| Constants | `UPPER_CASE` — `orca.float32`, `orca.cuda` (lowercase for dtype/device singletons is acceptable, matching PyTorch) |
| Private helpers | Leading underscore — `_validate_shape()` |
| "Dunder" methods | Implement `__repr__`, `__str__`, `__len__`, `__getitem__`, `__eq__`, arithmetic operators |
| Context managers | `orca.no_grad()`, `orca.device(...)` |
| Properties | Use `@property` — never `get_shape()` / `set_shape()` |

**Example**:

```python
import orca

t = orca.randn(3, 4, dtype=orca.float32, device="cpu")
print(t.shape)       # property, not method
print(t)             # __str__: human-readable summary
print(repr(t))       # __repr__: unambiguous, copy-pastable

with orca.no_grad():
    y = t @ t.T      # __matmul__, __getattr__
```

### 2.3  Type Annotations

| # | Rule |
|---|------|
| 1 | All public functions and methods must have **complete** type annotations. |
| 2 | Use `from __future__ import annotations` at the top of every file for PEP 604 union syntax (`X \| None`). |
| 3 | Use `typing` / `collections.abc` for generic types: `Sequence`, `Mapping`, `Iterator`. |
| 4 | Ship `.pyi` stub files for all PyO3-generated modules (`orca/_orca.pyi`). |
| 5 | CI runs `mypy --strict` on the entire `orca` package. |
| 6 | Use `@overload` to express multiple valid call signatures. |

**Stub file example** (`orca/_orca.pyi`):

```python
from __future__ import annotations
from typing import Sequence, overload

class Tensor:
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def device(self) -> str: ...

    @overload
    def reshape(self, shape: tuple[int, ...]) -> Tensor: ...
    @overload
    def reshape(self, *shape: int) -> Tensor: ...
    def reshape(self, *args: int | tuple[int, ...]) -> Tensor: ...

    def matmul(self, other: Tensor) -> Tensor: ...
    def __add__(self, other: Tensor | float) -> Tensor: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
```

#### Docstrings (Google style)

```python
def reshape(self, *shape: int) -> Tensor:
    """Reshape the tensor to the given shape.

    The total number of elements must remain unchanged. One dimension
    may be specified as ``-1``, in which case it is inferred.

    Args:
        *shape: The desired shape as positional integers, or a single
            tuple of integers.

    Returns:
        A new ``Tensor`` with the specified shape sharing the same
        underlying storage.

    Raises:
        OrcaError: If the new shape is incompatible with the current
            number of elements.

    Examples:
        >>> t = orca.randn(2, 3, 4)
        >>> t.reshape(6, 4).shape
        (6, 4)
        >>> t.reshape(-1).shape
        (24,)
    """
```

### 2.4  Testing

| Tool       | Purpose                              |
|------------|--------------------------------------|
| `pytest`   | Test runner                          |
| `pytest-xdist` | Parallel test execution         |
| `hypothesis` | Property-based testing (Python)   |
| `memray`   | Memory leak detection                |

#### `pyproject.toml` (pytest section)

```toml
[tool.pytest.ini_options]
testpaths   = ["tests"]
addopts     = "-ra -q --strict-markers --tb=short"
markers     = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "gpu: requires a GPU device",
    "parametrize_dtype: parametrize over dtypes",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning:pkg_resources",
]
```

#### Parametrized dtype × device tests

```python
import pytest
import orca

DTYPES  = [orca.float32, orca.float64]
DEVICES = ["cpu"]
if orca.cuda.is_available():
    DEVICES.append("cuda")

@pytest.fixture(params=DTYPES, ids=lambda d: d.name)
def dtype(request):
    return request.param

@pytest.fixture(params=DEVICES, ids=str)
def device(request):
    return request.param

def test_zeros_shape(dtype, device):
    t = orca.zeros(3, 4, dtype=dtype, device=device)
    assert t.shape == (3, 4)
    assert t.dtype == dtype
    assert t.device == device
```

#### Numerical Accuracy

```python
import orca
import numpy as np

def test_matmul_numerical_accuracy():
    a_np = np.random.randn(16, 32).astype(np.float32)
    b_np = np.random.randn(32, 8).astype(np.float32)
    expected = a_np @ b_np

    a = orca.from_numpy(a_np)
    b = orca.from_numpy(b_np)
    result = (a @ b).numpy()

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
```

#### Memory Leak Tests

```python
import pytest

@pytest.mark.slow
def test_tensor_no_memory_leak():
    """Ensure tensors are freed when they go out of scope."""
    import gc
    import tracemalloc

    tracemalloc.start()

    for _ in range(1_000):
        t = orca.randn(256, 256)
        del t

    gc.collect()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Peak should not exceed the size of a single tensor + overhead.
    max_expected = 256 * 256 * 4 * 2  # ~512 KB
    assert peak < max_expected, f"Suspected memory leak: peak = {peak} bytes"
```

---

## 3  Git Standards

### 3.1  Commit Messages

Follow [Conventional Commits v1.0](https://www.conventionalcommits.org/en/v1.0.0/):

```
type(scope): short imperative description

Optional body explaining WHY this change is made, not WHAT.
Wrap at 72 characters.

Refs: #123
BREAKING CHANGE: description of breaking change (if any)
```

**Types** (exhaustive list):

| Type       | When to use                                   |
|------------|-----------------------------------------------|
| `feat`     | New feature or capability                     |
| `fix`      | Bug fix                                       |
| `refactor` | Code restructuring with no behavior change    |
| `docs`     | Documentation only                            |
| `test`     | Adding or correcting tests                    |
| `bench`    | Benchmark additions or changes                |
| `ci`       | CI/CD pipeline changes                        |
| `chore`    | Dependency updates, tooling, housekeeping     |
| `perf`     | Performance improvement (backed by benchmark) |
| `style`    | Formatting, whitespace (no logic change)      |
| `revert`   | Revert a previous commit                      |

**Scope** (use the crate or component name):

```
feat(tensor): add einsum operation
fix(autograd): correct gradient accumulation for in-place ops
docs(python): add migration guide from PyTorch
ci(release): add aarch64-linux wheel build
test(orca-backend-gpu): add multi-GPU broadcast tests
```

### 3.2  Branch Strategy

```
main        ←── always releasable, protected
  │
  ├── dev   ←── integration branch, CI-gated merge to main
  │    │
  │    ├── feat/einsum-op
  │    ├── feat/mixed-precision
  │    ├── fix/grad-accumulation
  │    └── rfc/0007-custom-backend-api
  │
  └── release/v0.3.x  ←── maintenance branch for patch releases
```

| Branch pattern       | Purpose                            | Merges into |
|----------------------|------------------------------------|-------------|
| `main`               | Always releasable                  | —           |
| `dev`                | Integration, nightly builds        | `main`      |
| `feat/{name}`        | Feature development                | `dev`       |
| `fix/{name}`         | Bug fixes                          | `dev`       |
| `rfc/{number}`       | RFC implementation                 | `dev`       |
| `release/v{X.Y}.x`  | Maintenance branch (patch only)    | `main`      |
| `hotfix/{name}`      | Critical fix for `main`            | `main` + `dev` |

**Rules**:

- Feature branches must be rebased on `dev` before merging (linear history).
- `dev → main` merges happen via a release PR and require passing the full CI matrix.
- `release/v*` branches accept only cherry-picked fixes.

### 3.3  Pull Request Process

1. **Open a PR** against `dev` (or `main` for hotfixes).
2. **Fill out the PR template** (description, motivation, testing, breaking changes).
3. **CI must pass**: all checks green before review is requested.
4. **At least 1 approval** required (2 for `unsafe` code or public API changes).
5. **Tests are mandatory**: every PR must include or update tests.
6. **Documentation must be updated** if the PR changes any public API.
7. **Squash merge** for feature branches; merge commit for release PRs.
8. **Delete the branch** after merge.

**PR title** follows Conventional Commits format:

```
feat(tensor): add einsum operation (#234)
```

---

## 4  CI/CD Standards

### 4.1  CI Pipeline

Every push and PR triggers the following matrix:

```yaml
# Simplified representation of CI stages

stages:
  - lint
  - test
  - bench
  - package

lint:
  rust:
    - cargo fmt --all -- --check
    - cargo clippy --workspace --all-targets -- -D warnings
    - cargo deny check
    - cargo audit
  python:
    - ruff check python/
    - ruff format --check python/
    - mypy --strict python/orca/

test:
  rust:
    - cargo test --workspace
    - cargo test --workspace --features cuda-support   # GPU runner
  python:
    - pytest tests/ -x --tb=short
    - pytest tests/ -x --tb=short -m gpu               # GPU runner
  platforms:
    - ubuntu-latest
    - macos-latest       # ARM64
    - windows-latest

bench:
  # Runs on `dev` and `main` only, not on every PR
  - cargo bench --workspace -- --output-format bencher
  # Results are uploaded to a tracking dashboard

package:
  # Runs on tagged releases only
  - maturin build --release
  - twine upload dist/*
```

### 4.2  Quality Gates

A PR **cannot** be merged unless all of the following pass:

| Gate                    | Tool                     | Threshold           |
|-------------------------|--------------------------|---------------------|
| Rust formatting         | `rustfmt`                | Zero diff           |
| Rust lints              | `clippy`                 | Zero warnings       |
| Rust tests              | `cargo test`             | All pass            |
| Rust coverage           | `cargo llvm-cov`         | ≥ 80% (core crates) |
| Rust advisories         | `cargo audit`            | No vulnerabilities  |
| License compliance      | `cargo deny`             | All allowed         |
| Python formatting       | `ruff format`            | Zero diff           |
| Python lints            | `ruff check`             | Zero warnings       |
| Python type checking    | `mypy --strict`          | Zero errors         |
| Python tests            | `pytest`                 | All pass            |
| Commit message format   | `commitlint` or custom   | Conventional Commits |
| PR title format         | GitHub Action check      | Conventional Commits |

### 4.3  Release Process

1. A release PR merges `dev` into `main`.
2. CI runs the full matrix including GPU tests and benchmarks.
3. A maintainer tags the commit: `git tag -s v0.3.0`.
4. The tag triggers the release pipeline:
   - Build wheels for Linux (x86_64, aarch64), macOS (arm64), Windows (x86_64).
   - Publish to PyPI via `maturin publish`.
   - Publish Rust crates to crates.io via `cargo publish` (topological order).
   - Generate release notes from Conventional Commits.
   - Create a GitHub Release with changelogs and attached artifacts.
5. Post-release: bump version in `dev`, open the next milestone.

### 4.4  Dependency Auditing

| Cadence   | Action                                          |
|-----------|-------------------------------------------------|
| Every PR  | `cargo deny check` (licenses + bans)            |
| Weekly    | `cargo audit` (scheduled CI job)                 |
| Monthly   | Manual review of `Cargo.lock` diff               |
| Quarterly | Full dependency tree audit with `cargo vet`      |

---

## 5  Code Review Checklist

Every reviewer should evaluate PRs against this checklist. Copy it into
your review comment and check off each item.

### Correctness

- [ ] The code does what the PR description says it does.
- [ ] Edge cases are handled (empty tensors, scalar inputs, zero-sized dimensions).
- [ ] Error paths return meaningful `OrcaError` variants, not panics.
- [ ] No `.unwrap()` or `.expect()` in library code.
- [ ] Numerical precision is considered (f32 accumulation, catastrophic cancellation).

### Safety

- [ ] No new `unsafe` without a `// SAFETY:` comment and two approvals.
- [ ] No data races or aliasing violations.
- [ ] FFI calls validate inputs and handle null/error returns.
- [ ] No unbounded memory allocation based on user input.

### API Design

- [ ] Public API follows naming conventions (§1.2, §2.2).
- [ ] New public types/functions have doc comments with examples.
- [ ] Breaking changes are documented and justified.
- [ ] Python bindings expose a Pythonic API (properties, dunder methods).
- [ ] Type signatures use the correct generic bounds.

### Testing

- [ ] New code has unit tests covering the happy path.
- [ ] Error cases are tested.
- [ ] Bug fixes include a regression test referencing the issue.
- [ ] Property-based tests for numerical / algebraic properties.
- [ ] Tests are deterministic (seeded RNG, no timing dependencies).

### Performance

- [ ] No unnecessary allocations in hot paths.
- [ ] Time complexity is documented for new algorithms.
- [ ] Performance claims are backed by benchmarks in the PR.
- [ ] No `clone()` where a borrow suffices.
- [ ] Large tensors are not copied when a view would work.

### Style

- [ ] `cargo fmt` and `ruff format` produce no diff.
- [ ] `cargo clippy` and `ruff check` produce no warnings.
- [ ] Import ordering follows the convention (std → external → internal).
- [ ] No commented-out code.
- [ ] No TODO/FIXME without an associated issue number.

### Documentation

- [ ] All new public items have `///` doc comments.
- [ ] Module-level `//!` docs are updated if the module's scope changed.
- [ ] CHANGELOG is updated (or will be auto-generated).
- [ ] Migration notes added if the change is breaking.

### Security

- [ ] No hardcoded secrets or credentials.
- [ ] User-supplied paths are sanitized.
- [ ] Deserialization of model files validates magic bytes and version.
- [ ] Dependencies have no known vulnerabilities (`cargo audit`).

### Concurrency

- [ ] New types that should be `Send + Sync` are verified to be.
- [ ] Locks are not held across await points.
- [ ] Shared state uses appropriate synchronization primitives.
- [ ] No `Rc<T>` in library code (use `Arc<T>`).

---

## Appendix A — Quick Reference Commands

```bash
# ── Rust ─────────────────────────────────────────────
cargo fmt --all                           # Format
cargo fmt --all -- --check                # Check formatting
cargo clippy --workspace --all-targets -- -D warnings   # Lint
cargo test --workspace                    # Test
cargo test --workspace -- --ignored       # Run ignored/slow tests
cargo llvm-cov --workspace --html        # Coverage report
cargo bench --workspace                   # Benchmarks
cargo audit                               # Security audit
cargo deny check                          # License + ban check

# ── Python ───────────────────────────────────────────
ruff check python/                        # Lint
ruff format python/                       # Format
ruff format --check python/               # Check formatting
mypy --strict python/orca/                # Type check
pytest tests/ -x                          # Test
pytest tests/ -x -m "not slow"            # Skip slow tests
pytest tests/ -x -m gpu                   # GPU tests only

# ── Git ──────────────────────────────────────────────
git log --oneline -20                     # Recent history
git rebase -i dev                         # Rebase feature branch
```

---

## Appendix B — Toolchain Versions

The following table lists the minimum required toolchain versions. These are
enforced in CI via `rust-toolchain.toml` and `pyproject.toml`.

| Tool        | Minimum version | Pinned in                 |
|-------------|-----------------|---------------------------|
| Rust        | 1.85.0          | `rust-toolchain.toml`     |
| Python      | 3.10            | `pyproject.toml`          |
| `rustfmt`   | nightly         | `rust-toolchain.toml`     |
| `clippy`    | stable          | (bundled with rustc)      |
| `ruff`      | 0.8.0           | `pyproject.toml`          |
| `mypy`      | 1.13            | `pyproject.toml`          |
| `pytest`    | 8.0             | `pyproject.toml`          |
| `maturin`   | 1.8             | `pyproject.toml`          |
| `criterion` | 0.5             | `Cargo.toml`              |
| `proptest`  | 1.5             | `Cargo.toml`              |

---

*This document is versioned alongside the codebase. Propose changes via a
PR with the `docs` scope: `docs(standards): update X`.*
