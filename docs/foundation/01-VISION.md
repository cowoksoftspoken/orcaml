# Orca — Vision & Philosophy

> **"Simple by default. Powerful when needed."**

---

## 1. Why Orca Exists

The deep learning ecosystem has a problem that nobody talks about honestly:
**every existing framework forces developers into a false choice.**

On one side, there are frameworks designed for beginners — easy to start,
impossible to extend. They hide complexity behind magic, and when that magic
breaks (and it always breaks), users hit a wall with no way through.

On the other side, there are frameworks built for researchers and production
engineers — immensely powerful, but with learning curves measured in months,
APIs that require tribal knowledge, and abstractions that leak at every seam.

**PyTorch** chose flexibility. It won the research world but left production
engineers building custom infrastructure around it. Its Python-all-the-way
approach makes deployment painful. Its eager execution model makes optimization
hard. Its C++ core (ATen/c10) is a maze that few contributors fully understand.

**TensorFlow** chose production. It won the deployment world but alienated
researchers with its graph-first API (v1) and then spent years trying to
recover with eager mode (v2), creating a fragmented ecosystem.

**JAX** chose mathematical elegance. It's beautiful for researchers who think
in pure functions, but its functional paradigm and XLA dependency create a
steep learning curve and limit ecosystem growth.

None of them solved the fundamental problem: **progressive complexity.**

Orca exists to prove that this trade-off is false.

A framework can be both:
- **Instantly productive** for a student training their first neural network
- **Infinitely extensible** for a researcher designing a novel architecture
- **Production-ready** for an engineer deploying to millions of users

The key insight is **progressive abstraction** — a layered architecture where
each layer is a clean, stable abstraction over the one below it. Users never
hit a wall. They simply descend one level deeper when they need more control.

---

## 2. Core Beliefs

### 2.1 Progressive Abstraction Over Fixed Abstraction

Most frameworks pick one abstraction level and force everyone to use it.

Orca provides **four levels of abstraction**, each built on the one below:

```
┌─────────────────────────────────────────────────┐
│  Level 1: One-Line API                          │
│  model.fit(data)                                │
│  For: Beginners, quick experiments, prototyping  │
├─────────────────────────────────────────────────┤
│  Level 2: Configurable Training                 │
│  Trainer(model, optimizer, loss).fit(loader)     │
│  For: Standard workflows, hyperparameter tuning  │
├─────────────────────────────────────────────────┤
│  Level 3: Custom Training Loops                 │
│  for batch in loader:                           │
│      loss = model(batch)                        │
│      loss.backward()                            │
│      optimizer.step()                           │
│  For: Research, novel training procedures        │
├─────────────────────────────────────────────────┤
│  Level 4: Direct Tensor Operations              │
│  t = orca.tensor([1, 2, 3])                     │
│  t = orca.ops.custom_kernel(t)                  │
│  For: Framework developers, custom ops, backends │
└─────────────────────────────────────────────────┘
```

**The critical property**: moving from Level 1 to Level 4 requires
*adding code*, never *rewriting code*. A user's Level 1 project can
evolve into a Level 4 project without starting over.

### 2.2 Explicit Over Implicit

Debugging implicit behavior is the single largest time sink in deep learning
development. Hidden dtype conversions, silent broadcasting, automatic device
placement, gradient detachment — these "conveniences" become nightmares.

Orca prefers explicit behavior:
- **No silent dtype conversion.** If you multiply a float32 tensor by a
  float16 tensor, you get a clear error, not a silent upcast.
- **No hidden device transfers.** Moving a tensor to GPU requires an
  explicit `.to(device)` call, never a surprise copy.
- **No magic globals.** There's no hidden global state controlling behavior.
  Everything flows through explicit parameters.

This doesn't mean verbose. It means **predictable**.

### 2.3 Errors Are a Feature

A framework's error messages are as important as its API design. Most
frameworks treat errors as an afterthought — cryptic stack traces, unhelpful
messages, debug information scattered across warnings and logs.

Orca treats error messages as a first-class feature:

```
OrcaError::ShapeMismatch {
    operation: "matmul",
    left: Shape([32, 128]),
    right: Shape([64, 256]),
    hint: "For matmul(A, B), A's last dimension (128) must equal
           B's second-to-last dimension (64). Did you forget to
           transpose B? Try: matmul(a, b.T)"
}
```

Every error should:
1. Tell you **what** went wrong
2. Tell you **where** it went wrong
3. Tell you **why** it went wrong
4. Suggest **how** to fix it

### 2.4 Performance Is Non-Negotiable, But Readability Comes First

Orca is implemented in Rust for a reason. The runtime must be fast —
competitive with C++ frameworks — without sacrificing code clarity.

This means:
- **Zero-cost abstractions** wherever possible
- **No unnecessary allocations** on hot paths
- **Cache-friendly data layouts** for tensor operations
- **SIMD utilization** for CPU kernels
- **Lock-free data structures** where contention matters

But performance is never an excuse for unreadable code. A 5% speedup
that makes the code unmaintainable is a bad trade. A 50% speedup that
makes the code unmaintainable requires an RFC and extensive justification.

### 2.5 Modularity Is Not Optional

Every component of Orca is a standalone module with clear boundaries.
There is no monolithic core. There is no god object.

You can use:
- `orca-tensor` without `orca-autograd` (for numerical computing)
- `orca-autograd` without `orca-python` (for custom gradient computations in Rust)
- `python/orca/nn` without `python/orca/data` (if you have your own data pipeline)
- `orca-serialize` without `python/orca/nn` (for custom serialization needs)

This modularity is enforced at the crate level in Rust. Circular
dependencies between crates are a hard compilation error, not a style
suggestion.

### 2.6 The Python Layer Is Thin and Honest

Python is the API surface. Rust is the engine.

The Python layer should:
- **Mirror** the Rust API faithfully (no Python-only features that bypass Rust)
- **Add** Pythonic convenience (operator overloading, context managers, etc.)
- **Never** contain performance-critical logic
- **Always** delegate computation to Rust

This means Python developers get the ergonomics they expect, while
Rust developers can use the same core engine from Rust directly.

### 2.7 Composition Over Inheritance

Neural network modules are composed, not inherited. There is no deep
class hierarchy. A `Transformer` is a composition of `MultiHeadAttention`,
`LayerNorm`, and `Linear` — not a subclass of `BaseTransformer` which
subclasses `BaseModule` which subclasses `BaseComponent`.

```python
# Yes — composition
class Transformer(orca.nn.Module):
    def __init__(self, d_model, n_heads, n_layers):
        self.layers = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.norm = orca.nn.LayerNorm(d_model)

# No — inheritance hierarchy
class Transformer(BaseTransformer):
    ...
```

---

## 3. Who Is Orca For?

### 3.1 Primary Audiences

**Students and Learners**
- First framework experience
- Need immediate results with minimal boilerplate
- Must not be overwhelmed by options
- Use Level 1 and Level 2 APIs

**Researchers**
- Need full control over training procedures
- Implement novel architectures and loss functions
- Require efficient autograd with custom operations
- Use Level 3 and Level 4 APIs

**Production Engineers**
- Need reliable, deterministic inference
- Require model serialization and deployment
- Need performance profiling and optimization
- Use Level 2 for training, production runtime for serving

**Framework Contributors**
- Need clean, well-documented internals
- Must understand ownership and memory model
- Extend with custom backends and operations
- Use Level 4 APIs and Rust crate APIs directly

### 3.2 Non-Goals

Orca is **not**:

- **A PyTorch clone.** We learn from PyTorch's successes and failures,
  but we don't aim for API compatibility.

- **A wrapper.** We don't delegate to ONNX Runtime, LibTorch, or any
  other existing runtime. Our runtime is our own.

- **A research-only tool.** We design for production from day one.

- **A deployment-only tool.** We support the full lifecycle from
  experimentation to production.

- **Feature-complete on day one.** We prefer a small, excellent core
  over a large, mediocre surface area. Features are added through the
  RFC process and implemented when they're ready.

---

## 4. Design Philosophy in Practice

### 4.1 The "MNIST Test"

Every API decision must pass the MNIST test:
*Can a student train MNIST in under 10 lines of Python?*

```python
import orca

model = orca.nn.Sequential(
    orca.nn.Linear(784, 128),
    orca.nn.ReLU(),
    orca.nn.Linear(128, 10),
)

model.fit(orca.datasets.MNIST(), epochs=5)
```

If our API makes this harder, something is wrong.

### 4.2 The "GPT Test"

Every architecture decision must pass the GPT test:
*Can a researcher implement a GPT variant without fighting the framework?*

This means:
- Custom attention mechanisms must be easy to implement
- Custom loss functions must integrate seamlessly
- Gradient manipulation (accumulation, clipping, scaling) must be explicit
- Mixed precision must be a configuration, not a rewrite

### 4.3 The "Production Test"

Every runtime decision must pass the production test:
*Can an engineer deploy a trained model with predictable latency and memory?*

This means:
- Inference mode with zero autograd overhead
- Deterministic memory allocation (no GC surprises)
- Serialization that's forward-compatible
- Profiling hooks for latency and memory analysis

### 4.4 The "Debugging Test"

Every error and failure must pass the debugging test:
*Can a developer identify and fix the problem within 5 minutes?*

This means:
- Error messages include all relevant context
- Stack traces point to user code, not framework internals
- Shape mismatches are caught early, not during backward pass
- Type errors are caught at tensor creation, not at operation time

---

## 5. Naming Philosophy

The name **Orca** was chosen deliberately:

- **Power with intelligence.** Orcas are apex predators — not through
  brute force, but through sophisticated cooperation and strategy.
  Our framework should be powerful through smart design, not through
  raw feature count.

- **Social and cooperative.** Orcas hunt in pods. Our framework is
  designed for a community — contributors, plugin authors, and users
  working together through a well-governed open-source process.

- **Adaptable.** Orcas thrive in every ocean, from Arctic to tropical.
  Our framework should work everywhere — from a student's laptop to
  a production GPU cluster.

- **Memorable.** In a world of frameworks named after abstract
  mathematical concepts, a concrete, vivid name stands out.

### Naming Conventions

All components follow aquatic/marine metaphors where appropriate:

| Component | Metaphor |
|-----------|----------|
| Framework | Orca |
| Version Codenames | Ocean-themed (see Roadmap) |
| Internal Subsystems | Named functionally (tensor, autograd, nn) |

We do NOT force the metaphor. Internal module names are clear and
descriptive (`orca-tensor`, not `orca-fin`). The metaphor is for branding
and milestone names only.

---

## 6. Relationship to Existing Frameworks

### What We Learn From PyTorch
- ✅ Eager execution is the right default
- ✅ Pythonic API matters enormously
- ✅ Research flexibility drives adoption
- ❌ Python runtime creates performance ceiling
- ❌ C++/Python boundary is painful to maintain
- ❌ No progressive abstraction

### What We Learn From JAX
- ✅ Functional transformations are elegant
- ✅ Composable transforms (vmap, jit, grad) are powerful
- ✅ XLA compilation enables optimization
- ❌ Purely functional paradigm alienates most developers
- ❌ Pytree system is confusing for beginners
- ❌ Error messages are notoriously unhelpful

### What We Learn From TensorFlow
- ✅ Production deployment story matters
- ✅ Serving infrastructure is critical
- ✅ Ecosystem breadth drives enterprise adoption
- ❌ Graph-first was the wrong default
- ❌ API instability (v1 → v2) destroyed trust
- ❌ Complexity grew out of control

### What We Learn From Burn/Candle (Rust)
- ✅ Rust is viable for ML runtimes
- ✅ Type safety catches bugs early
- ✅ Performance is competitive
- ❌ Python interop needs more attention
- ❌ Ecosystem is still immature
- ❌ API ergonomics lag behind PyTorch

### What We Learn From tinygrad
- ✅ Simplicity is powerful
- ✅ Small codebases are maintainable
- ✅ Novel IR designs can be effective
- ❌ Too minimal for production use
- ❌ Single-maintainer risk
- ❌ Documentation is sparse

---

## 7. Success Criteria

Orca will be considered successful when:

### Short-Term (v0.1 - v0.3)
- [ ] A student can train MNIST in under 10 lines
- [ ] A researcher can implement a custom model without framework hacks
- [ ] The codebase has >80% test coverage
- [ ] Documentation covers every public API
- [ ] At least 5 external contributors have submitted merged PRs

### Medium-Term (v0.4 - v1.0)
- [ ] GPU training performance within 2x of PyTorch for standard models
- [ ] ONNX interoperability enables model exchange
- [ ] At least 3 research papers cite Orca
- [ ] Production deployment is documented and tested
- [ ] Plugin ecosystem has at least 10 community plugins

### Long-Term (v1.0 - v2.0)
- [ ] Performance competitive with PyTorch on standard benchmarks
- [ ] Used in production by at least 5 organizations
- [ ] Compiler provides measurable speedups over eager mode
- [ ] Distributed training scales to 8+ GPUs efficiently
- [ ] Active community with >100 contributors
- [ ] Framework is self-sustaining (not dependent on any single person)

---

## 8. Governance Preview

Orca is an open-source project governed by:

- **Benevolent Dictator For Life (BDFL)** model initially, transitioning
  to a **Core Team** model as the community grows.
- **RFC process** for all significant design decisions (see RFC Process doc).
- **Code of Conduct** based on the Contributor Covenant.
- **Semantic versioning** with clear deprecation policies.
- **Regular release cadence** (monthly pre-1.0, quarterly post-1.0).

---

## 9. Closing Statement

Orca is not about building another framework.

It's about building the framework we wish existed — one where simplicity
and power are not enemies, where beginners and experts share the same
tool, where the gap between research and production is measured in
configuration changes rather than rewrites.

We build Orca because we believe the deep learning ecosystem deserves
better developer experience without sacrificing any capability.

**Simple by default. Powerful when needed.**

---

*This document is a living artifact. It will evolve through the RFC process
as the project grows. Every major change to this document requires
community review.*

*Last updated: 2025-07-03*
*Status: DRAFT*
*Authors: Core Team*
