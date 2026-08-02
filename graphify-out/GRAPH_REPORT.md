# Graph Report - .  (2026-08-02)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 1366 nodes · 3048 edges · 72 communities (64 shown, 8 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 68 edges (avg confidence: 0.53)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `4a2937d9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55
- Community 56
- Community 57
- Community 58
- Community 59
- Community 60
- Community 61
- Community 62
- Community 63
- Community 64
- Community 65
- Community 66
- Community 67
- Community 68
- Community 69
- Community 71

## God Nodes (most connected - your core abstractions)
1. `Shape` - 160 edges
2. `Storage` - 156 edges
3. `DType` - 153 edges
4. `Module` - 61 edges
5. `GpuBackend` - 60 edges
6. `CpuBackend` - 55 edges
7. `PyTensor` - 53 edges
8. `Tensor<B>` - 47 edges
9. `Autodiff<B>` - 46 edges
10. `Parameter` - 38 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `DataLoader`  [INFERRED]
  examples/cv/train_cnn.py → python/orca/data/dataloader.py
- `main()` --calls--> `Adam`  [INFERRED]
  examples/cv/train_cnn.py → python/orca/optim/adam.py
- `main()` --calls--> `DataLoader`  [INFERRED]
  examples/nlp/train_transformer.py → python/orca/data/dataloader.py
- `main()` --calls--> `Adam`  [INFERRED]
  examples/nlp/train_transformer.py → python/orca/optim/adam.py
- `DigitsDataset` --inherits--> `Dataset`  [EXTRACTED]
  examples/cv/train_cnn.py → python/orca/data/dataloader.py

## Import Cycles
- None detected.

## Communities (72 total, 8 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (51): Box, AddBackward, Autodiff, AutodiffStorage, CastBackward, Conv2dBackward, Conv2dBackward<B>, DivBackward (+43 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (22): Bound, CompareOp, global_backend_cpu(), global_backend_gpu(), load_tensors(), Operand, orca_python(), PyDevice (+14 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (16): From, Autodiff<B>, ExpBackwardOp, GatherBackwardOp, LogBackwardOp, MaxToShapeBackward, ReluBackward, Result (+8 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (14): ComputePipeline, GpuBackend, Pipelines, Arc, Buffer, BufferAddress, BufferUsages, Default (+6 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (11): Div, Into, Mul, B, Option, Result, Self, Vec (+3 more)

### Community 5 - "Community 5"
Cohesion: 0.12
Nodes (7): AutodiffStorage<S>, CpuBackend, Option, Result, T, Vec, Storage

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (27): Embedding, A simple lookup table that stores embeddings of a fixed dictionary and size., Parameter, Args: tensor (Tensor): The tensor to be wrapped as a parameter., In-place update of the parameter's tensor data. Typically used by Optimizers…, Moves the parameter to a specified device (e.g., 'cpu' or 'gpu'). Args: device…, A kind of Tensor that is to be considered a module parameter. Parameters are…, Adam (+19 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (45): 1. Introduction, 2. When to Write an RFC, 3. RFC Lifecycle, 4. RFC Template, 5. Review Process, 6. RFC Numbering and Storage, 7. Example RFC, Alternative A: Treat bf16 as an External Plugin (+37 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (42): 10. Context Window Management, 1. Purpose, 2. Before Writing Any Code, 3. Architecture Compliance, 4. Code Quality Rules, 5. Design Review Protocol, 6. Naming Consistency, 7. File Organization (+34 more)

### Community 9 - "Community 9"
Cohesion: 0.08
Nodes (21): download_file(), load(), Downloads/loads configuration and pretrained weights from Hugging Face Hub,…, BERT, BERTBlock, BERT model architecture blueprint., GPT2, GPT2Block (+13 more)

### Community 10 - "Community 10"
Cohesion: 0.08
Nodes (25): Add, AutogradTensorExt, B, Option, Result, Self, Tensor<Autodiff<B>>, DistributedCommunicator (+17 more)

### Community 11 - "Community 11"
Cohesion: 0.06
Nodes (38): fixture, parametrize, cpu_device(), gpu_device(), Check analytical autograd correctness using numerical finite differences., Fixture to provide GPU device string., Verify learning rate schedulers update optimizer learning rates correctly., Verify Linear layer gradients using numerical finite differences. (+30 more)

### Community 12 - "Community 12"
Cohesion: 0.07
Nodes (14): autocast, GradScaler, Gradient scaling for mixed precision training., DistributedDataParallel, Synchronizes and averages gradients across all workers using TCP sockets., DistributedDataParallel wrapper for Orca modules to enable data-parallel…, no_grad, Context manager and decorator to temporarily disable gradient computation. (+6 more)

### Community 13 - "Community 13"
Cohesion: 0.15
Nodes (7): chunk(), _device_to_str(), einsum(), _format_tensor_data(), Splits the tensor into a specific number of chunks along a given dimension.…, Evaluates the Einstein summation convention on the operands. Args: equation…, _tensor_str()

### Community 14 - "Community 14"
Cohesion: 0.06
Nodes (32): 1. Why Orca Exists, 2.1 Progressive Abstraction Over Fixed Abstraction, 2.2 Explicit Over Implicit, 2.3 Errors Are a Feature, 2.4 Performance Is Non-Negotiable, But Readability Comes First, 2.5 Modularity Is Not Optional, 2.6 The Python Layer Is Thin and Honest, 2.7 Composition Over Inheritance (+24 more)

### Community 15 - "Community 15"
Cohesion: 0.13
Nodes (20): GELU, Applies the Gaussian Error Linear Unit (GELU) function. This uses the…, MultiHeadAttention, Allows the model to jointly attend to information from different representation…, Forward pass for MultiHeadAttention. Args: query (Tensor): Query tensor of…, Dropout, During training, randomly zeroes some of the elements of the input tensor with…, Linear (+12 more)

### Community 16 - "Community 16"
Cohesion: 0.12
Nodes (18): MemoryPool, Buffer, BufferAddress, BufferUsages, Default, HashMap, Self, Vec (+10 more)

### Community 17 - "Community 17"
Cohesion: 0.09
Nodes (15): Flatten, Flattens all dimensions except the first (batch) dimension., Module, Any, Adds a buffer to the module. This is typically used to register a state that…, Returns a dictionary containing a whole state of the module. Args: prefix…, Copies parameters and buffers from `state` into this module and its…, Saves the module parameters to a file in Safetensors format. Args: filepath… (+7 more)

### Community 18 - "Community 18"
Cohesion: 0.09
Nodes (13): FmtResult, Layout, AlignedBuffer, CpuByteStorage, Arc, Clone, Debug, Drop (+5 more)

### Community 19 - "Community 19"
Cohesion: 0.08
Nodes (25): 1. System Overview, 2. Crate Architecture, 3.1 Tensor and Memory Layout, 3.2 Backend Trait, 3.3 Autograd Engine (Tape-Based), 3.4 Module System, 3. Core Abstractions, 4. Python-Rust Bridge (PyO3) (+17 more)

### Community 20 - "Community 20"
Cohesion: 0.09
Nodes (22): 1. Hardware Scheduling and Shader Launch Overhead, 1. Robust Error Handling, 2. Strict Crate Hierarchy, 2. Workload Scaling and Parallelization Gains, 3. Autograd Tape Integrity, 4. GPU Shader Design, 5. Code Quality and Testing, Advanced Orchestration API (+14 more)

### Community 21 - "Community 21"
Cohesion: 0.10
Nodes (9): Deref, Device, Display, Formatter, Result, bf16_conversion_rounds_and_preserves_nan(), f16_conversion_roundtrips_common_values(), OrcaError (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.13
Nodes (12): export_onnx(), get_parameter_names(), ModelTracer, patch_tensor_methods(), Any, Exports an Orca model to ONNX format (opset 17+) using Tape Tracing., restore_tensor_methods(), import_onnx() (+4 more)

### Community 23 - "Community 23"
Cohesion: 0.14
Nodes (9): Copy, bool, CpuFloat, CpuNumeric, Debug, Default, Self, Send (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.12
Nodes (10): Compose, Normalize, RandomCrop, RandomFlip, Normalize a tensor image with mean and standard deviation., Convert a list or numpy array to orca.Tensor., Composes several transforms together., Crop the given image at a random location. (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.17
Nodes (20): _calculate_correct_fan(), _calculate_fan_in_and_fan_out(), _calculate_gain(), _init_tensor(), kaiming_normal_(), kaiming_uniform_(), normal_(), ones_() (+12 more)

### Community 26 - "Community 26"
Cohesion: 0.11
Nodes (11): Applies the Softmax function to an n-dimensional input Tensor. Rescales the…, Forward pass for Softmax. Args: x (Tensor): Input tensor. Returns: Tensor:…, Helper function to create a scalar tensor., Applies the Hyperbolic Tangent (Tanh) function element-wise. Formula: `Tanh(x)…, Forward pass for Tanh. Args: x (Tensor): Input tensor. Returns: Tensor: Output…, Forward pass for GELU. Args: x (Tensor): Input tensor. Returns: Tensor: Output…, _scalar(), Softmax (+3 more)

### Community 27 - "Community 27"
Cohesion: 0.19
Nodes (5): CosineAnnealingLR, LinearWarmup, LRScheduler, Base class for learning rate schedulers., StepLR

### Community 28 - "Community 28"
Cohesion: 0.23
Nodes (7): Dataset, random_split(), Dataset wrapping a subset of another dataset., Randomly split a dataset into non-overlapping new datasets of given lengths., An abstract class representing a Dataset. All datasets that represent a map…, SubsetDataset, T_co

### Community 29 - "Community 29"
Cohesion: 0.17
Nodes (7): ArrayDataset, CSVDataset, Any, High-level Dataset wrapper for loading tabular data directly from a CSV file., High-level Dataset wrapper for in-memory arrays (lists, numpy arrays, or…, Verify ArrayDataset, CSVDataset, and random_split load and split data correctly., test_high_level_dataset_wrappers()

### Community 30 - "Community 30"
Cohesion: 0.64
Nodes (8): orca-autograd, orca-backend-cpu, orca-backend-gpu, orca-core, orca-distributed, orca-python, orca-serialize, orca-tensor

### Community 31 - "Community 31"
Cohesion: 0.18
Nodes (11): 2.1  Style and Formatting, 2.2  API Design, 2.3  Type Annotations, 2.4  Testing, 2  Python Coding Standards, Docstrings (Google style), Memory Leak Tests, Numerical Accuracy (+3 more)

### Community 32 - "Community 32"
Cohesion: 0.20
Nodes (5): DigitsDataset, main(), DataLoader, Data loader. Combines a dataset and a sampler, and provides an iterable over…, Returns an iterator over the dataset batches. Yields: Tuple[Tensor, Tensor]: A…

### Community 33 - "Community 33"
Cohesion: 0.22
Nodes (5): main(), Transformer sequence classification example. Task: classify whether a…, Synthetic dataset where the label depends on the first token., SequenceDataset, TransformerClassifier

### Community 34 - "Community 34"
Cohesion: 0.20
Nodes (10): 1.2  Naming Conventions, 1.3  Error Handling, 1.4  Documentation, 1.6  Unsafe Code, 1.7  Performance, 1.8  Concurrency, 1.9  Dependencies, 1  Rust Coding Standards (+2 more)

### Community 35 - "Community 35"
Cohesion: 0.20
Nodes (10): 5  Code Review Checklist, API Design, Concurrency, Correctness, Documentation, Performance, Safety, Security (+2 more)

### Community 36 - "Community 36"
Cohesion: 0.20
Nodes (6): CrossEntropyLoss, MSELoss, Forward pass for MSELoss. Args: pred (Tensor): Predictions tensor. target…, Computes the cross entropy loss between input logits and target. Target is…, Creates a criterion that measures the mean squared error (squared L2 norm)…, Forward pass for CrossEntropyLoss. Args: pred (Tensor): Logits (unnormalized…

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (8): 03 — Coding Standards, 3.1  Commit Messages, 3.2  Branch Strategy, 3.3  Pull Request Process, 3  Git Standards, Appendix A — Quick Reference Commands, Appendix B — Toolchain Versions, Purpose

### Community 38 - "Community 38"
Cohesion: 0.28
Nodes (3): DigitsDataset, main(), SimpleCNN

### Community 39 - "Community 39"
Cohesion: 0.28
Nodes (4): A sequential container. Modules will be added to it in the order they are…, Sequential, Model, Base class for all neural network models in Orca. Subclassing ``nn.Model``…

### Community 40 - "Community 40"
Cohesion: 0.43
Nodes (7): compute_numerical_gradient(), Computes numerical gradients for a model's parameters using finite differences., test_batchnorm_grad(), test_conv2d_grad(), test_cross_entropy_grad(), test_layernorm_grad(), test_linear_relu_grad()

### Community 41 - "Community 41"
Cohesion: 0.25
Nodes (4): AdaptiveAvgPool2d, MaxPool2d, Applies a 2D adaptive average pooling over an input signal composed of several…, Applies a 2D max pooling over an input signal composed of several input planes.

### Community 42 - "Community 42"
Cohesion: 0.29
Nodes (4): QuantizedLinear, Creates a QuantizedLinear layer from a trained float Linear layer., Calibrates and quantizes the weight tensor to INT8 symmetrically., Quantized version of the Linear layer using 8-bit symmetric quantization.

### Community 43 - "Community 43"
Cohesion: 0.29
Nodes (6): 🚀 Next Objective, Orca Framework - Agent Instructions, ⚠️ Rules & Coding Standards for Agents, 🚦 Status Proyek (Completed Phases), 🏗️ Struktur Repositori, 🎯 Visi Proyek

### Community 44 - "Community 44"
Cohesion: 0.29
Nodes (7): 1.5  Testing, Coverage, Criterion Benchmarks, Naming Convention, Property-Based Testing, Regression Tests, Test Organization

### Community 45 - "Community 45"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 0: Foundation (v0.1.0) — "First Breath", Risks, Success Metrics

### Community 46 - "Community 46"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 1: Neural Networks (v0.2.0) — "First Hunt", Risks, Success Metrics

### Community 47 - "Community 47"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 2: Research Ready (v0.3.0) — "Deep Dive", Risks, Success Metrics

### Community 48 - "Community 48"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 3: GPU Acceleration (v0.4.0) — "Breaking Surface", Risks, Success Metrics

### Community 49 - "Community 49"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 4: Ecosystem (v0.5.0) — "Pod Formation", Risks, Success Metrics

### Community 50 - "Community 50"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 5: Scale (v1.0.0) — "Open Ocean", Risks, Success Metrics

### Community 51 - "Community 51"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 6: Compiler (v1.5.0) — "Echolocation", Risks, Success Metrics

### Community 52 - "Community 52"
Cohesion: 0.29
Nodes (7): Dependencies, Exit Criteria, Goals, Key Deliverables, Phase 7: World-Class (v2.0.0) — "Apex Predator", Risks, Success Metrics

### Community 53 - "Community 53"
Cohesion: 0.33
Nodes (3): Option, Self, Vec

### Community 54 - "Community 54"
Cohesion: 0.33
Nodes (5): Cross-Phase Concerns, Decision Log, Orca Roadmap, Timeline Overview, Versioning Contract

### Community 55 - "Community 55"
Cohesion: 0.40
Nodes (5): Backend, Clone, Debug, Send, Sync

### Community 56 - "Community 56"
Cohesion: 0.40
Nodes (3): Verify that exporting to ONNX and importing back reproduces correct values and…, SimpleMLP, test_onnx_roundtrip()

### Community 57 - "Community 57"
Cohesion: 0.40
Nodes (5): 4.1  CI Pipeline, 4.2  Quality Gates, 4.3  Release Process, 4.4  Dependency Auditing, 4  CI/CD Standards

### Community 58 - "Community 58"
Cohesion: 0.40
Nodes (3): Conv2d, Forward pass of the Conv2d layer. Args: x (Tensor): Input tensor of shape…, Applies a 2D convolution over an input signal composed of several input planes.…

### Community 60 - "Community 60"
Cohesion: 0.50
Nodes (4): 1.1  Style and Formatting, Clippy, Module Organization, `rustfmt.toml`

### Community 62 - "Community 62"
Cohesion: 0.67
Nodes (3): Result, test_autograd_activation(), test_basic_autograd()

### Community 63 - "Community 63"
Cohesion: 0.50
Nodes (3): Forward pass for ReLU. Args: x (Tensor): Input tensor. Returns: Tensor: Output…, Applies the rectified linear unit function element-wise. Formula: `ReLU(x) =…, ReLU

### Community 64 - "Community 64"
Cohesion: 0.50
Nodes (3): Applies the sigmoid function element-wise. Formula: `Sigmoid(x) = 1 / (1 +…, Forward pass for Sigmoid. Args: x (Tensor): Input tensor. Returns: Tensor:…, Sigmoid

## Knowledge Gaps
- **231 isolated node(s):** `orca`, `🎯 Visi Proyek`, `🏗️ Struktur Repositori`, `🚦 Status Proyek (Completed Phases)`, `🚀 Next Objective` (+226 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **8 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Tensor` connect `Community 10` to `Community 1`, `Community 2`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **Why does `Storage` connect `Community 5` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 10`, `Community 16`, `Community 18`, `Community 55`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `Shape` connect `Community 2` to `Community 0`, `Community 66`, `Community 3`, `Community 4`, `Community 5`, `Community 69`, `Community 10`, `Community 53`, `Community 21`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Are the 23 inferred relationships involving `Module` (e.g. with `GELU` and `ReLU`) actually correct?**
  _`Module` has 23 INFERRED edges - model-reasoned connections that need verification._
- **What connects `orca`, `🎯 Visi Proyek`, `🏗️ Struktur Repositori` to the rest of the system?**
  _231 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.051685393258426963 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.08757908757908758 - nodes in this community are weakly interconnected._