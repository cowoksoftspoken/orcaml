use crate::backend::Backend;
use orca_core::{DType, Device, OrcaError, Result, Shape};
use std::ops::Add;

/// The core multidimensional array representation in Orca.
///
/// A Tensor is parameterized over a generic `Backend` trait, enabling
/// zero-cost dispatch to different execution environments (CPU, CUDA, etc.)
#[derive(Debug, Clone)]
pub struct Tensor<B: Backend> {
    /// The underlying raw data storage.
    storage: B::Storage,
    /// The dimensions of the tensor.
    shape: Shape,
    /// Memory strides (used for view semantics).
    strides: Vec<usize>,
    /// The data type of the tensor elements.
    dtype: DType,
    /// The backend instance associated with this tensor.
    backend: B,
}

impl<B: Backend> Tensor<B> {
    pub fn zeros(backend: B, shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let strides = Self::compute_contiguous_strides(&shape);
        let storage = backend.zeros(&shape, dtype)?;

        Ok(Self {
            storage,
            shape,
            strides,
            dtype,
            backend,
        })
    }

    /// Creates a new tensor filled with ones.
    pub fn ones(backend: B, shape: impl Into<Shape>, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let data = vec![1.0; shape.num_elements()];
        Self::from_f32_slice(backend, &data, shape)?.to_dtype(dtype)
    }

    /// Creates a 0-dimensional (scalar) tensor with a specific value.
    pub fn scalar(backend: B, value: f32, dtype: DType) -> Result<Self> {
        let shape = Shape::new(vec![1]);
        Self::from_f32_slice(backend, &[value], shape)?.to_dtype(dtype)
    }

    /// Creates a new tensor filled with random uniform values between low and high.
    pub fn rand_uniform(
        backend: B,
        shape: impl Into<Shape>,
        low: f32,
        high: f32,
        dtype: DType,
    ) -> Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let num_elements = shape.num_elements();

        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(rng.gen_range(low..high));
        }

        Self::from_f32_slice(backend, &data, shape)?.to_dtype(dtype)
    }

    /// Creates a new tensor filled with random normal values.
    pub fn randn(
        backend: B,
        shape: impl Into<Shape>,
        mean: f32,
        std: f32,
        dtype: DType,
    ) -> Result<Self> {
        use rand_distr::{Distribution, Normal};
        let shape = shape.into();
        let num_elements = shape.num_elements();

        let normal = Normal::new(mean, std).map_err(|_| OrcaError::ShapeMismatch {
            op: "randn",
            expected: "valid std dev".to_string(),
            got: "invalid std dev".to_string(),
        })?;

        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(num_elements);
        for _ in 0..num_elements {
            data.push(normal.sample(&mut rng));
        }

        Self::from_f32_slice(backend, &data, shape)?.to_dtype(dtype)
    }

    /// Creates a dropout mask tensor scaled by 1/(1-p).
    pub fn rand_dropout_mask(
        backend: B,
        shape: impl Into<Shape>,
        p: f32,
        dtype: DType,
    ) -> Result<Self> {
        use rand::Rng;
        let shape = shape.into();
        let num_elements = shape.num_elements();

        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(num_elements);
        let scale = 1.0 / (1.0 - p);

        for _ in 0..num_elements {
            let val = if rng.gen::<f32>() > p { scale } else { 0.0 };
            data.push(val);
        }

        Self::from_f32_slice(backend, &data, shape)?.to_dtype(dtype)
    }

    /// Returns the shape of this tensor.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the strides of this tensor.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the data type of the tensor.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Casts the tensor to a target data type.
    pub fn to_dtype(&self, target_dtype: DType) -> Result<Self> {
        if self.dtype == target_dtype {
            return Ok(self.clone());
        }
        let storage = self
            .backend
            .cast(&self.storage, &self.shape, self.dtype, target_dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: target_dtype,
            backend: self.backend.clone(),
        })
    }

    /// Returns the device where this tensor is stored.
    pub fn device(&self) -> Device {
        self.backend.device()
    }

    /// Gets a reference to the backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Gets a reference to the underlying storage.
    pub fn storage(&self) -> &B::Storage {
        &self.storage
    }

    /// Gets a mutable reference to the underlying storage.
    pub fn storage_mut(&mut self) -> &mut B::Storage {
        &mut self.storage
    }

    /// Returns the number of dimensions.
    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Creates a tensor from raw components.
    pub fn from_raw_parts(
        backend: B,
        storage: B::Storage,
        shape: Shape,
        strides: Vec<usize>,
        dtype: DType,
    ) -> Self {
        Self {
            backend,
            storage,
            shape,
            strides,
            dtype,
        }
    }

    /// Creates a tensor from an f32 slice.
    pub fn from_f32_slice(backend: B, data: &[f32], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let strides = Self::compute_contiguous_strides(&shape);
        if data.len() != shape.num_elements() {
            return Err(OrcaError::ShapeMismatch {
                op: "from_f32_slice",
                expected: format!("{} elements", shape.num_elements()),
                got: format!("{} elements", data.len()),
            });
        }
        let storage = backend.from_f32_slice(&shape, data)?;

        Ok(Self {
            storage,
            shape,
            strides,
            dtype: DType::F32,
            backend,
        })
    }

    /// Extracts the tensor data as an f32 vec.
    pub fn has_nan_or_inf(&self) -> Result<bool> {
        self.backend.has_nan_or_inf(&self.storage, self.dtype)
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.backend.to_bytes(&self.storage)
    }

    pub fn from_bytes(backend: B, bytes: &[u8], shape: Shape, dtype: DType) -> Result<Self> {
        let storage = backend.from_bytes(&shape, bytes, dtype)?;
        let mut strides = vec![0; shape.rank()];
        let mut current = 1;
        for i in (0..shape.rank()).rev() {
            strides[i] = current;
            current *= shape.0[i];
        }
        Ok(Self {
            storage,
            shape,
            strides,
            dtype,
            backend,
        })
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.dtype != DType::F32 {
            return Err(OrcaError::UnsupportedDType {
                op: "to_f32_vec",
                dtype: self.dtype,
            });
        }
        self.backend.to_f32_vec(&self.storage)
    }

    /// Helper to compute contiguous strides for a given shape.
    fn compute_contiguous_strides(shape: &Shape) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![1; shape.rank()];
        for i in (0..shape.rank() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Matrix multiplication.
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        let rank1 = self.shape.rank();
        let rank2 = rhs.shape.rank();
        if rank1 < 2 || rank2 < 2 {
            return Err(OrcaError::ShapeMismatch {
                op: "matmul",
                expected: ">= 2D tensors".into(),
                got: format!("{}D and {}D", rank1, rank2),
            });
        }
        if self.shape[rank1 - 1] != rhs.shape[rank2 - 2] {
            return Err(OrcaError::ShapeMismatch {
                op: "matmul",
                expected: format!(
                    "Inner dimensions must match: {} == {}",
                    self.shape[rank1 - 1],
                    rhs.shape[rank2 - 2]
                ),
                got: format!("{} != {}", self.shape[rank1 - 1], rhs.shape[rank2 - 2]),
            });
        }

        // Ensure batch dimensions match
        if rank1 != rank2 {
            // For now, require same rank
            return Err(OrcaError::ShapeMismatch {
                op: "matmul",
                expected: "Same rank for batched matmul".into(),
                got: format!("{}D and {}D", rank1, rank2),
            });
        }
        for i in 0..rank1 - 2 {
            if self.shape[i] != rhs.shape[i] {
                return Err(OrcaError::ShapeMismatch {
                    op: "matmul",
                    expected: "Matching batch dimensions".into(),
                    got: format!("{} != {}", self.shape[i], rhs.shape[i]),
                });
            }
        }

        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(
                self.backend.device(),
                rhs.backend.device(),
            ));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::DTypeMismatch {
                op: "matmul",
                expected: self.dtype,
                got: rhs.dtype,
            });
        }
        let lhs_storage = self.storage.clone();
        let rhs_storage = rhs.storage.clone();
        let target_dtype = self.dtype;

        let mut out_shape_vec = self.shape.to_vec();
        out_shape_vec[rank1 - 2] = self.shape[rank1 - 2];
        out_shape_vec[rank1 - 1] = rhs.shape[rank2 - 1];
        let out_shape = Shape::new(out_shape_vec);
        let storage = self.backend.matmul(
            &lhs_storage,
            &rhs_storage,
            &self.shape,
            &rhs.shape,
            target_dtype,
        )?;
        let strides = Self::compute_contiguous_strides(&out_shape);

        Ok(Tensor {
            storage,
            shape: out_shape,
            strides,
            dtype: target_dtype,
            backend: self.backend.clone(),
        })
    }

    /// Multiply by a scalar.
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self> {
        let storage = self
            .backend
            .mul_scalar(&self.storage, scalar, &self.shape, self.dtype)?;

        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    /// Applies ReLU activation element-wise.
    pub fn relu(&self) -> Result<Self> {
        let storage = self.backend.relu(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sigmoid(&self) -> Result<Self> {
        let storage = self
            .backend
            .sigmoid(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn expand(&self, out_shape: &Shape) -> Result<Self> {
        let storage = self
            .backend
            .expand(&self.storage, &self.shape, out_shape, self.dtype)?;
        // Strides for the expanded tensor should be contiguous for now since we physically copied the data
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sum_to_shape(&self, out_shape: &Shape) -> Result<Self> {
        let storage =
            self.backend
                .sum_to_shape(&self.storage, &self.shape, out_shape, self.dtype)?;
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn max_to_shape(&self, out_shape: &Shape) -> Result<Self> {
        let storage =
            self.backend
                .max_to_shape(&self.storage, &self.shape, out_shape, self.dtype)?;
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sum(&self) -> Result<Self> {
        let out_shape = Shape::new(vec![1]);
        self.sum_to_shape(&out_shape)
    }

    pub fn reshape(&self, out_shape: &Shape) -> Result<Self> {
        let storage = self
            .backend
            .reshape(&self.storage, &self.shape, out_shape, self.dtype)?;
        let mut strides = vec![0; out_shape.rank()];
        let mut current = 1;
        for i in (0..out_shape.rank()).rev() {
            strides[i] = current;
            current *= out_shape.0[i];
        }
        Ok(Self {
            storage,
            shape: out_shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn exp(&self) -> Result<Self> {
        let storage = self.backend.exp(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn log(&self) -> Result<Self> {
        let storage = self.backend.log(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Self> {
        let storage = self.backend.conv2d(
            &self.storage,
            &weight.storage,
            bias.map(|b| &b.storage),
            &self.shape,
            &weight.shape,
            padding,
            stride,
            dilation,
            groups,
            self.dtype,
        )?;

        let in_h = self.shape[2];
        let in_w = self.shape[3];
        let k_h = weight.shape[2];
        let k_w = weight.shape[3];
        let out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;
        let out_shape = Shape::new(vec![self.shape[0], weight.shape[0], out_h, out_w]);
        let strides = Self::compute_contiguous_strides(&out_shape);

        Ok(Self {
            storage,
            shape: out_shape,
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn sqrt(&self) -> Result<Self> {
        let storage = self.backend.sqrt(&self.storage, &self.shape, self.dtype)?;
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn gather(&self, dim: usize, index: &Self) -> Result<Self> {
        let storage = self.backend.gather(
            &self.storage,
            dim,
            &index.storage,
            &self.shape,
            &index.shape,
            self.dtype,
        )?;
        let strides = Self::compute_contiguous_strides(&index.shape);
        Ok(Self {
            storage,
            shape: index.shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn scatter(&self, dim: usize, index: &Self, src: &Self) -> Result<Self> {
        let storage = self.backend.scatter(
            &self.storage,
            dim,
            &index.storage,
            &src.storage,
            &self.shape,
            &index.shape,
            self.dtype,
        )?;
        let strides = Self::compute_contiguous_strides(&self.shape);
        Ok(Self {
            storage,
            shape: self.shape.clone(),
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let storage = self
            .backend
            .transpose(&self.storage, &self.shape, dim0, dim1, self.dtype)?;
        let mut new_shape_vec = self.shape.to_vec();
        new_shape_vec.swap(dim0, dim1);
        let shape = Shape::new(new_shape_vec);
        let strides = Self::compute_contiguous_strides(&shape);
        Ok(Self {
            storage,
            shape,
            strides,
            dtype: self.dtype,
            backend: self.backend.clone(),
        })
    }

    pub fn einsum(equation: &str, inputs: &[&Self]) -> Result<Self> {
        let parts: Vec<&str> = equation.split("->").collect();
        if parts.len() != 2 {
            return Err(OrcaError::InternalError("Invalid einsum equation".into()));
        }

        let lhs_rhs: Vec<&str> = parts[0].split(',').collect();
        if lhs_rhs.len() != 2 || inputs.len() != 2 {
            return Err(OrcaError::InternalError(
                "Currently only 2-operand einsum is supported".into(),
            ));
        }

        let a_str = lhs_rhs[0].trim();
        let b_str = lhs_rhs[1].trim();
        let out_str = parts[1].trim();

        let a = inputs[0];
        let b = inputs[1];

        // Find matching characters between a_str and b_str that are NOT in out_str. These are the contraction axes.
        let mut contract_a = Vec::new();
        let mut contract_b = Vec::new();
        let mut batch_a = Vec::new();
        let mut batch_b = Vec::new();

        for (i, c) in a_str.chars().enumerate() {
            if !out_str.contains(c) {
                contract_a.push(i);
            } else if b_str.contains(c) {
                batch_a.push(i);
            }
        }

        for (i, c) in b_str.chars().enumerate() {
            if !out_str.contains(c) {
                contract_b.push(i);
            } else if a_str.contains(c) {
                batch_b.push(i);
            }
        }

        // Very basic matcher for batched matmul where last 2 dims are matrix dims
        // This is still limited but dynamically detects transpositions instead of hardcoding equations

        // If B's contract dimension is its last dimension, we need to transpose it so it's the second-to-last
        // Actually, standard matmul is (..., M, K) @ (..., K, N)
        // If a_str = "bhqd", b_str = "bhkd" (K=d) -> Q @ K.T -> b needs transpose of last two dims
        let b_contract = contract_b.last().copied().unwrap_or(b_str.len() - 2);

        if b_contract == b_str.len() - 1 {
            // b needs transposition for matmul
            let b_t = b.transpose(b_str.len() - 2, b_str.len() - 1)?;
            a.matmul(&b_t)
        } else {
            a.matmul(b)
        }
    }
}

use std::ops::{Div, Mul, Sub};

impl<B: Backend> Div for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn div(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "div",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(
                self.backend.device(),
                rhs.backend.device(),
            ));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::DTypeMismatch {
                op: "div",
                expected: self.dtype,
                got: rhs.dtype,
            });
        }
        let lhs_storage = self.storage.clone();
        let rhs_storage = rhs.storage.clone();
        let target_dtype = self.dtype;

        let storage = self
            .backend
            .div(&lhs_storage, &rhs_storage, &self.shape, target_dtype)?;

        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: target_dtype,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> Add for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "add",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(
                self.backend.device(),
                rhs.backend.device(),
            ));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::DTypeMismatch {
                op: "add",
                expected: self.dtype,
                got: rhs.dtype,
            });
        }
        let lhs_storage = self.storage.clone();
        let rhs_storage = rhs.storage.clone();
        let target_dtype = self.dtype;

        let storage = self
            .backend
            .add(&lhs_storage, &rhs_storage, &self.shape, target_dtype)?;

        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: target_dtype,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> Sub for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "sub",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(
                self.backend.device(),
                rhs.backend.device(),
            ));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::DTypeMismatch {
                op: "sub",
                expected: self.dtype,
                got: rhs.dtype,
            });
        }
        let lhs_storage = self.storage.clone();
        let rhs_storage = rhs.storage.clone();
        let target_dtype = self.dtype;

        let storage = self
            .backend
            .sub(&lhs_storage, &rhs_storage, &self.shape, target_dtype)?;

        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: target_dtype,
            backend: self.backend.clone(),
        })
    }
}

impl<B: Backend> Mul for &Tensor<B> {
    type Output = Result<Tensor<B>>;

    fn mul(self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            return Err(OrcaError::ShapeMismatch {
                op: "mul",
                expected: self.shape.to_string(),
                got: rhs.shape.to_string(),
            });
        }
        if self.backend.device() != rhs.backend.device() {
            return Err(OrcaError::DeviceMismatch(
                self.backend.device(),
                rhs.backend.device(),
            ));
        }
        if self.dtype != rhs.dtype {
            return Err(OrcaError::DTypeMismatch {
                op: "mul",
                expected: self.dtype,
                got: rhs.dtype,
            });
        }
        let lhs_storage = self.storage.clone();
        let rhs_storage = rhs.storage.clone();
        let target_dtype = self.dtype;

        let storage = self
            .backend
            .mul(&lhs_storage, &rhs_storage, &self.shape, target_dtype)?;

        Ok(Tensor {
            storage,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            dtype: target_dtype,
            backend: self.backend.clone(),
        })
    }
}
