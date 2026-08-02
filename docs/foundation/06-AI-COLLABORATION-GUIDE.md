# AI Collaboration Guide

> **How AI assistants should contribute to Orca.**

This document defines the rules, conventions, and protocols that any AI assistant
must follow when writing code, reviewing code, or making architectural decisions
for the Orca deep learning framework. Adherence is not optional — violations will
be caught in CI or code review.

| Field | Value |
|---|---|
| **Status** | Living document |
| **Audience** | AI assistants, developers using AI tools |
| **Created** | 2026-07-03 |
| **Last updated** | 2026-08-02 |

---

## 1. Purpose

AI-assisted development is a force multiplier — but only when the AI understands
the project's architecture, conventions, and quality bar. Without guardrails, AI
contributions introduce inconsistency, architectural drift, and subtle bugs.

This guide exists to:

1. **Ensure consistency.** Every AI-generated contribution follows the same patterns,
   naming conventions, error handling strategy, and file organization as human-written
   code.

2. **Prevent architectural violations.** The crate dependency graph, trait system, and
   memory model are load-bearing design decisions. Violating them creates technical
   debt that compounds across the entire project.

3. **Raise the quality floor.** AI must produce code that is production-grade on the
   first pass — not "good enough to iterate on." Tests, documentation, and error
   handling are non-negotiable.

4. **Provide a checklist.** When in doubt, the AI should consult this document. If
   this document doesn't answer the question, the AI should ask a human rather than
   guess.

---

## 2. Before Writing Any Code

Every AI session that involves code changes must begin with a preparation step.
Do not generate code until you have completed this checklist.

### Required Reading (in order)

| Priority | Document | Why |
|---|---|---|
| 1 | `02-ARCHITECTURE.md` | Understand the crate graph, trait system, and data flow |
| 2 | `03-CODING-STANDARDS.md` | Understand formatting, naming, error handling, and testing requirements |
| 3 | `05-ROADMAP.md` | Understand what phase the project is in and what is in scope |
| 4 | Relevant RFC (if any) | Understand the design decisions for the feature being implemented |
| 5 | Existing source code in the target crate | Understand established patterns and conventions |

### Pre-Flight Checks

Before writing any code, verify:

- [ ] **Which crate does this change belong to?** Never put code in the wrong crate.
      If unsure, check the Architecture Bible's crate responsibility matrix.

- [ ] **Does this change require a new public type or trait?** If yes, write the type
      signature and doc comment *first*, get approval, then implement.

- [ ] **Does this change touch the crate dependency graph?** If yes, verify the new
      dependency is allowed. The Architecture Bible defines the legal dependency edges.

- [ ] **Is there an existing pattern for this kind of change?** Search the codebase
      for similar implementations. Follow the existing pattern exactly unless there is
      a documented reason to deviate.

- [ ] **What phase is the project in?** Do not implement features from future phases.
      If the change requires a primitive that doesn't exist yet, stop and flag it.

---

## 3. Architecture Compliance

These rules are **hard constraints**. No exceptions without an approved RFC.

### Crate Dependency Graph

The current Rust dependency layers are:

- `orca-core` has no workspace dependencies.
- `orca-tensor` depends on `orca-core`.
- `orca-autograd` depends on `orca-core` and `orca-tensor`.
- `orca-backend-cpu` depends on `orca-core` and `orca-tensor`.
- `orca-backend-gpu` depends on `orca-core`, `orca-tensor`, and `orca-backend-cpu`.
- `orca-distributed` depends on `orca-core` and `orca-tensor`.
- `orca-serialize` depends on `orca-core` and `orca-tensor`.
- `orca-python` depends on `orca-core`, `orca-tensor`, `orca-autograd`, `orca-backend-cpu`, `orca-backend-gpu`, `orca-distributed`, and `orca-serialize`.

The Python package also contains higher-level modules under `python/orca/`
(`nn`, `optim`, and `data`) that sit on top of the Rust bindings.

**Rules:**

1. **Never add an edge that creates a cycle.** If A depends on B, B must not depend
   on A, directly or transitively.

2. **Never add an upward dependency.** Lower crates (`orca-core`, `orca-tensor`) must
   never depend on higher layers (`orca-autograd`, `orca-backend-cpu`,
   `orca-backend-gpu`, `orca-distributed`, `orca-serialize`, or `orca-python`).
   Information flows down via traits, not up via concrete types.

3. **Never make `orca-core` depend on anything outside `std`.** It is the foundation.

4. **New crates require an RFC.** Do not create a new crate without a design document
   that specifies its position in the dependency graph.

### Trait System

- Use traits for **abstraction boundaries** between crates.
- Implement traits on **concrete types** in the crate that owns the concrete type.
- Never use `dyn Trait` when generics will do. Prefer static dispatch.
- Never add a method to an existing trait without checking all implementors.

### Error Handling

- All fallible operations return `Result<T, OrcaError>`.
- **Never use `unwrap()` in library code.** Use `expect()` only in tests.
- **Never use `panic!()` in library code.** Convert all panics to `Result::Err`.
- Propagate errors with `?`. Add context with `.map_err()` or `.context()`.
- Define error variants in `orca-core`. Do not create ad-hoc error types in other crates.

### Memory Model

- Tensors own their data by default.
- Views borrow from the owning tensor and carry a lifetime.
- GPU memory is managed by the caching allocator. Never call `cudaMalloc` directly.
- Never use global mutable state. No `lazy_static!` with `Mutex` for shared state.
- Never use `Rc<RefCell<T>>`. Use Rust's ownership model properly.

---

## 4. Code Quality Rules

All code generated by AI must meet these standards. CI enforces most of them
automatically. If CI fails, the contribution is rejected.

### Compilation

```bash
# Must pass with zero warnings
cargo build --workspace 2>&1 | grep -c "warning" # must be 0

# Must pass clippy with warnings as errors
cargo clippy --workspace -- -D warnings

# Must be formatted
cargo fmt --workspace --check
```

### Documentation

- **Every public item** (`pub fn`, `pub struct`, `pub enum`, `pub trait`, `pub type`)
  must have a rustdoc comment.
- Doc comments must include:
  - A one-line summary (imperative mood: "Creates a tensor", not "This creates").
  - A `# Examples` section for non-trivial items.
  - A `# Errors` section for fallible functions, listing each error variant.
  - A `# Panics` section if the function can panic (should be rare in library code).
- Python-facing classes and methods must have Python docstrings via `#[pyo3(text_signature)]`.

### Testing

| Requirement | Details |
|---|---|
| Happy path test | Every public function has at least one test that exercises the normal case |
| Error case test | Every fallible function has tests that trigger each error variant |
| Edge case test | Tests for empty tensors, zero-sized dimensions, single-element tensors |
| Gradient check | All differentiable ops have numerical gradient verification (finite differences) |
| Determinism test | Operations must produce identical results given the same input and seed |

```rust
// Example: test naming convention
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_broadcasts_scalar_to_vector() { /* ... */ }

    #[test]
    fn add_returns_shape_mismatch_for_incompatible_shapes() { /* ... */ }

    #[test]
    fn add_handles_empty_tensor() { /* ... */ }
}
```

### Test Naming Convention

Tests follow the pattern: `{function_name}_{scenario}_{expected_behavior}`.

- `matmul_2d_by_2d_returns_correct_shape`
- `reshape_to_incompatible_size_returns_error`
- `backward_through_relu_produces_correct_gradient`

---

## 5. Design Review Protocol

Not all changes are equal. The review protocol scales with the impact of the change.

### Tier 1: Bug fixes, documentation, test additions

- **Process:** Direct PR. No RFC needed.
- **AI responsibility:** Include the root cause analysis in the PR description.

### Tier 2: New functions, new methods on existing types

- **Process:** PR with design justification in the description.
- **AI responsibility:** Explain why this API shape was chosen over alternatives.
  Show the public API surface and usage examples.

### Tier 3: New types, new traits, new modules

- **Process:** Write a design brief (mini-RFC) in the PR description.
- **AI responsibility:** Document:
  - The problem this type/trait solves.
  - Why existing types/traits are insufficient.
  - The public API (all public methods with signatures).
  - How this interacts with the crate dependency graph.
  - At least two usage examples.

### Tier 4: New crates, architectural changes

- **Process:** Full RFC required. Do not write code until the RFC is approved.
- **AI responsibility:** Draft the RFC. Include:
  - Motivation and problem statement.
  - Detailed design with type signatures.
  - Position in the crate dependency graph.
  - Impact on existing code.
  - Alternatives considered and reasons for rejection.
  - Rollout plan.

### Performance-Sensitive Code

Any code in the hot path (tensor ops, autograd, CUDA kernels) must include:

- **Benchmarks:** Use `criterion` for Rust benchmarks. Compare against a baseline.
- **Flamegraph:** For non-obvious performance characteristics, include profiling data.
- **Complexity analysis:** State the time and space complexity in the doc comment.

### Unsafe Code

Any `unsafe` block must include a `// SAFETY:` comment that proves:

1. All invariants required by the unsafe API are upheld.
2. No undefined behavior is possible.
3. The unsafe block is as small as possible.

```rust
// SAFETY: `ptr` is valid because it was obtained from `Vec::as_mut_ptr()` on a
// Vec that is alive for the duration of this scope. The index `i` is bounds-
// checked on the line above. No other references to this element exist.
unsafe { *ptr.add(i) = value; }
```

---

## 6. Naming Consistency

Use these established patterns. Do not invent new conventions.

### Types

| Pattern | Example | Used For |
|---|---|---|
| `*Builder` | `TensorBuilder` | Builder pattern types that construct a complex object step-by-step |
| `*Error` | `ShapeMismatchError` | Error variants within `OrcaError` |
| `*Config` | `TrainerConfig` | Configuration structs passed to constructors or training loops |
| `*Options` | `ConvOptions` | Optional parameters for operations (distinct from config) |
| `*Backend` | `CudaBackend` | Backend implementations for device abstraction |
| `*Guard` | `AutocastGuard` | RAII guards that restore state on drop |
| `*Context` | `BackwardContext` | State passed through the autograd tape |

### Methods

| Pattern | Example | Used For |
|---|---|---|
| `with_*` | `with_dtype(DType::F32)` | Builder methods that set a field and return `Self` |
| `into_*` | `into_tensor()` | Ownership-transferring conversions (consumes `self`) |
| `as_*` | `as_slice()` | Borrowing conversions (borrows `&self`) |
| `try_*` | `try_reshape()` | Fallible operations that return `Result` |
| `from_*` | `from_numpy()` | Constructor that creates the type from another representation |
| `to_*` | `to_device()` | Conversion that may allocate (creates a new value, may consume or borrow) |
| `is_*` | `is_contiguous()` | Boolean predicates |
| `set_*` | `set_requires_grad()` | Mutable setters |
| `*_mut` | `data_mut()` | Methods returning `&mut T` |

### Crate Naming

- All Rust crates are prefixed with `orca-`: `orca-core`, `orca-tensor`,
  `orca-autograd`, `orca-backend-cpu`, `orca-backend-gpu`, `orca-distributed`,
  `orca-serialize`, and `orca-python`.
- Crate names use kebab-case: `orca-backend-gpu`, not `orca_backend_gpu`.
- Module names within crates use snake_case.

### Constant and Static Naming

- Constants: `SCREAMING_SNAKE_CASE` — `const DEFAULT_LEARNING_RATE: f64 = 0.001;`
- Type parameters: single uppercase letter or descriptive — `T`, `D: Device`, `B: Backend`.
- Lifetime parameters: descriptive when possible — `'tensor`, `'ctx`.

---

## 7. File Organization

### Where to Put Things

```
orca-{crate}/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Public API re-exports, crate-level docs
│   ├── {module}.rs          # Module implementation (for small modules)
│   └── {module}/            # Module directory (for large modules)
│       ├── mod.rs           # Public API for this module
│       ├── {submodule}.rs   # Implementation details
│       └── tests.rs         # Unit tests (if not using inline #[cfg(test)])
├── tests/                   # Integration tests
│   └── test_{feature}.rs
└── benches/                 # Benchmarks
    └── bench_{feature}.rs
```

### Rules

| Item | Location | Rationale |
|---|---|---|
| New public type | `src/{module}.rs` or `src/{module}/mod.rs` | One type per file when the type is large; co-locate small related types |
| New public trait | `src/traits.rs` or `src/{module}/traits.rs` | Traits are discoverability-critical; keep them in predictable locations |
| New error variant | `orca-core/src/error.rs` | All errors live in one place |
| Unit tests | Inline `#[cfg(test)] mod tests` at bottom of the file | Tests live next to the code they test |
| Integration tests | `tests/test_{feature}.rs` | Tests that cross module boundaries |
| Benchmarks | `benches/bench_{feature}.rs` | Use `criterion` crate |
| Examples | `examples/{example_name}.rs` | Runnable with `cargo run --example` |
| Python bindings | `orca-python/src/{module}.rs` | Mirror the Rust module structure |

### Re-exports

- `lib.rs` should re-export the public API using `pub use`.
- Users should never need to reach into submodules: `use orca_tensor::Tensor`, not
  `use orca_tensor::storage::dense::DenseTensor`.

---

## 8. PR Description Template

Every AI-proposed PR must include these sections:

1. **Summary** — One paragraph: what and why.
2. **Motivation** — Link to issue, RFC, or roadmap phase.
3. **Design Decisions** — Why this approach; what alternatives were rejected.
4. **Changes** — New/modified public API (with signatures and diffs), internal changes.
5. **Testing checklist** — Happy path, error cases, edge cases, gradient checks, benchmarks.
6. **Documentation checklist** — Rustdoc, Python docstrings, examples, Architecture Bible.
7. **Build checklist** — `cargo build` (0 warnings), `cargo clippy -- -D warnings`, `cargo fmt --check`, `cargo test`, no `unwrap()`/`panic!()` in lib code.

---

## 9. Common Mistakes to Avoid

Each of these is a **hard reject** in code review.

| # | Anti-Pattern | Fix |
|---|---|---|
| 1 | `unwrap()` in library code | Use `?` with `.map_err()` to convert to `OrcaError` |
| 2 | Inventing custom error types | Use `OrcaError` variants defined in `orca-core` |
| 3 | Adding upward/circular deps | Never make lower crates depend on higher crates |
| 4 | Global mutable state (`lazy_static! + Mutex`) | Pass state explicitly via function arguments |
| 5 | Missing rustdoc on `pub` items | Every `pub` item needs summary, `# Errors`, `# Examples` |
| 6 | Unconstrained generics (`<T>`) | Always add meaningful trait bounds (`T: TensorLike + Send + Sync`) |
| 7 | `String` where an enum belongs | Use typed enums (`Device`, `DType`) not stringly-typed APIs |
| 8 | Exposing `&Vec<T>` | Return `&[T]` — never leak internal collection types |
| 9 | Functions > 50 lines | Decompose into named, tested helper functions |
| 10 | Missing `Send + Sync` bounds | All tensor and module types must be thread-safe |

### Example: Correct Error Handling

```rust
// ❌ let shape = tensor.shape().unwrap();
// ✅
let shape = tensor.shape().map_err(|e| OrcaError::InvalidArgument {
    message: format!("failed to get shape: {e}"),
})?;
```

### Example: Correct Documentation

```rust
/// Computes the matrix product of two 2-D tensors.
///
/// # Errors
///
/// Returns [`OrcaError::ShapeMismatch`] if inner dimensions do not match.
///
/// # Examples
///
/// ```
/// let c = matmul(&a, &b)?;
/// assert_eq!(c.shape(), &[2, 2]);
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> { /* ... */ }
```

---

## 10. Context Window Management

AI assistants have limited context windows. Use these strategies to work effectively
within those constraints.

### What to Read First

When working on a change, read files in this priority order:

1. **This guide** (`06-AI-COLLABORATION-GUIDE.md`) — you are here.
2. **The Architecture Bible** (`02-ARCHITECTURE.md`) — understand the structure.
3. **The target file** — understand what you're modifying.
4. **Tests for the target file** — understand expected behavior.
5. **Adjacent files in the same module** — understand local conventions.
6. **The relevant RFC** (if any) — understand design decisions.

### When Context Gets Tight

If your context window is filling up:

1. **Summarize what you've read** before proceeding. Write a brief summary of the
   key constraints, then you can "forget" the raw source.

2. **Work on one function at a time.** Don't try to implement an entire module in
   one pass. Implement, test, then move to the next function.

3. **Use the public API as a contract.** If you know the function signature and the
   doc comment, you can implement without re-reading the caller's code.

4. **Ask for specific files** rather than reading entire directories.

### Large File Strategy

For files exceeding 300 lines:

1. Read the top-level structure first (imports, type definitions, trait implementations).
2. Read the specific function or block you need to modify.
3. Read the tests for that function.
4. Make your change.
5. Verify your change against the function signature and doc comment.

### Multi-File Changes

When a change spans multiple files:

1. **Plan first.** List all files that need to change and what changes each needs.
2. **Start from the bottom of the dependency graph.** Change `orca-core` before
   `orca-tensor`, `orca-tensor` before `orca-autograd`, and the Rust bindings before
   the Python layers they expose.
3. **Compile after each file.** Don't accumulate changes across files without
   verifying compilation.
4. **Write tests last.** Ensure the implementation compiles before writing tests.

### Maintaining Consistency Across Sessions

AI context does not persist between sessions. To maintain consistency:

1. **Always re-read this guide** at the start of a new session.
2. **Check recent git history** (`git log --oneline -20`) to understand recent changes.
3. **Run the test suite** before starting work to verify the codebase is green.
4. **Read the CHANGELOG** to understand what has been added recently.
5. **Never assume** — always verify by reading the actual source code.

---

## Appendix: Quick Reference Card

```
ORCA AI — SESSION START CHECKLIST
=================================
[ ] Read 02-ARCHITECTURE.md + 03-CODING-STANDARDS.md
[ ] Identify target crate, check dependency graph
[ ] Search for existing patterns, verify project phase

HARD RULES: No unwrap/panic in lib | No global state | No circular deps
           All pub items documented | All new code tested | Result<T, OrcaError>

NAMING: *Builder/with_*() | try_*() | as_*()/into_*() | *Config | *Error

SUBMIT: build(0 warn) | clippy -D warnings | fmt --check | test | rustdoc
```

---

> [!NOTE]
> This guide evolves with the project. If you encounter a situation not covered
> here, flag it to a human maintainer. Do not guess — ask.
