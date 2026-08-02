#![allow(clippy::too_many_arguments, clippy::wrong_self_convention)]
use orca_core::{DType, Device, Result, Shape};
use std::fmt::Debug;

/// Trait defining the requirements for tensor storage.
/// Storage encapsulates the raw memory management.
pub trait Storage: Clone + Send + Sync + Debug + 'static {
    /// The size of the storage in elements.
    fn len(&self) -> usize;

    /// Whether the storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait defining a computational backend.
/// A backend is responsible for allocating storage and executing operations.
pub trait Backend: Clone + Send + Sync + Debug + 'static {
    /// The specific storage type used by this backend.
    type Storage: Storage;

    /// The physical device this backend targets.
    fn device(&self) -> Device;

    /// Create a new storage block with the given shape and data type.
    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage>;

    /// Create a storage from an f32 slice.
    #[allow(clippy::wrong_self_convention)]
    fn from_f32_slice(&self, shape: &Shape, data: &[f32]) -> Result<Self::Storage>;

    /// Extract data into an f32 vec.
    fn to_f32_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>>;

    /// Element-wise addition.
    /// Matrix multiplication of two 2D tensors.
    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    /// Transpose dimensions of an N-dimensional tensor.
    fn transpose(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dim0: usize,
        dim1: usize,
        dtype: DType,
    ) -> Result<Self::Storage>;

    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn mul_scalar(
        &self,
        storage: &Self::Storage,
        scalar: f32,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 2.0 Math Operations
    fn mul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn relu(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    fn sigmoid(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 2.0 Backward Math Operations
    fn relu_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn sigmoid_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 3.0 Broadcasting and Reductions
    fn expand(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn sum_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 4.0 Reshape and Transcendental
    fn reshape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn exp(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    fn log(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    fn exp_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn log_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 6.0 Math Operations
    fn div(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn sqrt(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage>;
    fn div_backward_lhs(
        &self,
        grad_out: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn div_backward_rhs(
        &self,
        grad_out: &Self::Storage,
        lhs_primal: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn sqrt_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    /// Accumulates two gradients of the same shape and dtype.
    fn accumulate_grad(&self, lhs: &Self::Storage, rhs: &Self::Storage) -> Result<Self::Storage>;

    // Phase 1.5 Convolution
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        input: &Self::Storage,
        weight: &Self::Storage,
        bias: Option<&Self::Storage>,
        in_shape: &Shape,
        weight_shape: &Shape,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        dtype: DType,
    ) -> Result<Self::Storage>;

    #[allow(clippy::too_many_arguments)]
    fn conv2d_backward_input(
        &self,
        grad_out: &Self::Storage,
        weight: &Self::Storage,
        in_shape: &Shape,
        weight_shape: &Shape,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        dtype: DType,
    ) -> Result<Self::Storage>;

    #[allow(clippy::too_many_arguments)]
    fn conv2d_backward_weight(
        &self,
        grad_out: &Self::Storage,
        input: &Self::Storage,
        in_shape: &Shape,
        weight_shape: &Shape,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        dtype: DType,
    ) -> Result<Self::Storage>;

    fn conv2d_backward_bias(
        &self,
        grad_out: &Self::Storage,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 6.1 Casting
    fn cast(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        current_dtype: DType,
        target_dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 2.1 Indexing
    fn scatter(
        &self,
        storage: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        src: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn gather(
        &self,
        storage: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn scatter_backward_src(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn scatter_backward_base(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn gather_backward(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;

    // Phase 1-2 Production Hardening
    fn from_bytes(&self, shape: &Shape, bytes: &[u8], dtype: DType) -> Result<Self::Storage>;
    fn to_bytes(&self, storage: &Self::Storage) -> Result<Vec<u8>>;
    fn has_nan_or_inf(&self, storage: &Self::Storage, dtype: DType) -> Result<bool>;
    fn max_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
    fn max_to_shape_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        out_primal: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage>;
}
