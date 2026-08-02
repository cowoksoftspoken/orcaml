#![allow(clippy::needless_range_loop)]
use matrixmultiply::sgemm;
use rayon::prelude::*;
use std::fmt::Debug;

use crate::math::CpuNumeric;
use crate::storage::CpuByteStorage;
use crate::{dispatch_dtype, dispatch_float};
use orca_core::{DType, Device, OrcaError, Result, Shape};
use orca_tensor::{Backend, Storage};

/// The CPU execution backend.
#[derive(Debug, Clone, Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn pad_shape_left(shape: &[usize], target_rank: usize) -> Vec<usize> {
        let mut padded = vec![1; target_rank];
        if shape.len() <= target_rank {
            let offset = target_rank - shape.len();
            for (i, &dim) in shape.iter().enumerate() {
                padded[offset + i] = dim;
            }
        }
        padded
    }

    pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![0; shape.len()];
        let mut current = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = current;
            current *= shape[i];
        }
        strides
    }

    fn linear_to_multi(mut index: usize, shape: &[usize]) -> Vec<usize> {
        let mut multi = vec![0; shape.len()];
        for i in (0..shape.len()).rev() {
            multi[i] = index % shape[i];
            index /= shape[i];
        }
        multi
    }

    fn multi_to_linear(multi: &[usize], strides: &[usize]) -> usize {
        multi.iter().zip(strides.iter()).map(|(m, s)| m * s).sum()
    }

    pub fn from_slice<T: Copy>(
        &self,
        shape: &Shape,
        data: &[T],
        _dtype: DType,
    ) -> Result<CpuByteStorage> {
        let element_size = std::mem::size_of::<T>();
        let num_elements = shape.num_elements();
        let total_bytes = num_elements * element_size;
        let mut storage = CpuByteStorage::new(total_bytes, num_elements, element_size)?;
        storage.as_mut_slice::<T>().copy_from_slice(data);
        Ok(storage)
    }
}

impl Backend for CpuBackend {
    type Storage = CpuByteStorage;

    fn device(&self) -> Device {
        Device::Cpu
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let element_size = match dtype {
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::F16 | DType::BF16 => 2,
            DType::U8 | DType::Bool => 1,
        };

        let num_elements = shape.num_elements();
        let total_bytes = num_elements * element_size;

        CpuByteStorage::new(total_bytes, num_elements, element_size)
    }

    fn from_f32_slice(&self, shape: &Shape, data: &[f32]) -> Result<Self::Storage> {
        self.from_slice::<f32>(shape, data, DType::F32)
    }

    fn to_f32_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>> {
        // Technically this should be aware of dtype and convert!
        // But since to_f32_vec doesn't take dtype, we assume the storage contains f32.
        let bytes = storage.as_bytes();
        let mut vec = vec![0.0f32; storage.len()];

        let dest_bytes: &mut [u8] =
            unsafe { std::slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, bytes.len()) };
        dest_bytes.copy_from_slice(bytes);

        Ok(vec)
    }

    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let lhs_f32 = self.cast(lhs, shape, dtype, DType::F32)?;
            let rhs_f32 = self.cast(rhs, shape, dtype, DType::F32)?;
            let res_f32 = self.add(&lhs_f32, &rhs_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_dtype!(dtype, T, {
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            let result: Vec<T> = lhs_slice
                .par_iter()
                .zip(rhs_slice.par_iter())
                .map(|(a, b)| a.add(*b))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let lhs_f32 = self.cast(lhs, lhs_shape, dtype, DType::F32)?;
            let rhs_f32 = self.cast(rhs, rhs_shape, dtype, DType::F32)?;
            let res_f32 = self.matmul(&lhs_f32, &rhs_f32, lhs_shape, rhs_shape, DType::F32)?;

            let rank = lhs_shape.rank();
            let m = lhs_shape[rank - 2];
            let n = rhs_shape[rank - 1];
            let mut out_shape_dims = lhs_shape.to_vec();
            out_shape_dims[rank - 2] = m;
            out_shape_dims[rank - 1] = n;
            let out_shape = Shape::new(out_shape_dims);

            return self.cast(&res_f32, &out_shape, DType::F32, dtype);
        }

        if dtype != DType::F32 {
            return Err(OrcaError::UnsupportedDType {
                op: "matmul (cpu)",
                dtype,
            });
        }

        // Ensure shapes are compatible for batched matmul
        let rank = lhs_shape.rank();
        if rank < 2 || rhs_shape.rank() < 2 {
            return Err(OrcaError::InternalError(
                "Matmul requires at least 2D shapes".into(),
            ));
        }

        // Calculate batch size
        let mut batch_size = 1;
        for i in 0..rank - 2 {
            if lhs_shape[i] != rhs_shape[i] {
                return Err(OrcaError::InternalError(
                    "Matmul batch dimensions must match".into(),
                ));
            }
            batch_size *= lhs_shape[i];
        }

        let m = lhs_shape[rank - 2];
        let k = lhs_shape[rank - 1];
        let n = rhs_shape[rank - 1]; // assuming rhs is also (..., K, N)

        let lhs_f32 =
            unsafe { std::slice::from_raw_parts(lhs.as_bytes().as_ptr() as *const f32, lhs.len()) };
        let rhs_f32 =
            unsafe { std::slice::from_raw_parts(rhs.as_bytes().as_ptr() as *const f32, rhs.len()) };

        let mut result = vec![0.0f32; batch_size * m * n];

        for b in 0..batch_size {
            let lhs_offset = b * m * k;
            let rhs_offset = b * k * n;
            let res_offset = b * m * n;

            unsafe {
                sgemm(
                    m,
                    k,
                    n,
                    1.0,
                    lhs_f32.as_ptr().add(lhs_offset),
                    k as isize,
                    1,
                    rhs_f32.as_ptr().add(rhs_offset),
                    n as isize,
                    1,
                    0.0,
                    result.as_mut_ptr().add(res_offset),
                    n as isize,
                    1,
                );
            }
        }

        let mut out_shape_vec = lhs_shape.to_vec();
        out_shape_vec[rank - 2] = m;
        out_shape_vec[rank - 1] = n;
        let out_shape = Shape::new(out_shape_vec);
        self.from_f32_slice(&out_shape, &result)
    }

    fn transpose(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dim0: usize,
        dim1: usize,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.transpose(&s_f32, shape, dim0, dim1, DType::F32)?;
            let mut out_shape = shape.to_vec();
            out_shape.swap(dim0, dim1);
            return self.cast(&res_f32, &Shape::new(out_shape), DType::F32, dtype);
        }

        if dim0 >= shape.rank() || dim1 >= shape.rank() {
            return Err(OrcaError::InternalError(
                "Invalid transpose dimensions".into(),
            ));
        }

        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut out_slice = vec![T::zero(); storage.len()];

            let mut out_shape_vec = shape.to_vec();
            out_shape_vec.swap(dim0, dim1);

            // Compute strides for out mapping to in
            let mut in_strides = vec![1; shape.rank()];
            for i in (0..shape.rank() - 1).rev() {
                in_strides[i] = in_strides[i + 1] * shape[i + 1];
            }

            let mut out_strides = vec![1; shape.rank()];
            for i in (0..shape.rank() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape_vec[i + 1];
            }

            // Loop over all elements (can be optimized but fine for research readiness)
            for out_idx in 0..storage.len() {
                let mut current_idx = out_idx;
                let mut in_idx = 0;

                for i in 0..shape.rank() {
                    let coord = current_idx / out_strides[i];
                    current_idx %= out_strides[i];

                    // Map back to input coordinates
                    let original_dim = if i == dim0 {
                        dim1
                    } else if i == dim1 {
                        dim0
                    } else {
                        i
                    };
                    in_idx += coord * in_strides[original_dim];
                }

                out_slice[out_idx] = in_slice[in_idx];
            }

            let out_shape = Shape::new(out_shape_vec);
            self.from_slice::<T>(&out_shape, &out_slice, dtype)
        })
    }

    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let lhs_f32 = self.cast(lhs, shape, dtype, DType::F32)?;
            let rhs_f32 = self.cast(rhs, shape, dtype, DType::F32)?;
            let res_f32 = self.sub(&lhs_f32, &rhs_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_dtype!(dtype, T, {
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            let result: Vec<T> = lhs_slice
                .par_iter()
                .zip(rhs_slice.par_iter())
                .map(|(a, b)| a.sub(*b))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn mul_scalar(
        &self,
        storage: &Self::Storage,
        scalar: f32,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.mul_scalar(&s_f32, scalar, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let scalar_t = T::from_f32(scalar);

            let result: Vec<T> = in_slice.par_iter().map(|v| v.mul(scalar_t)).collect();

            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn mul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let lhs_f32 = self.cast(lhs, shape, dtype, DType::F32)?;
            let rhs_f32 = self.cast(rhs, shape, dtype, DType::F32)?;
            let res_f32 = self.mul(&lhs_f32, &rhs_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        if dtype != DType::F32 {
            return Err(OrcaError::UnsupportedDType {
                op: "mul (cpu)",
                dtype,
            });
        }

        let lhs_f32 =
            unsafe { std::slice::from_raw_parts(lhs.as_bytes().as_ptr() as *const f32, lhs.len()) };
        let rhs_f32 =
            unsafe { std::slice::from_raw_parts(rhs.as_bytes().as_ptr() as *const f32, rhs.len()) };

        let result: Vec<f32> = lhs_f32
            .par_iter()
            .zip(rhs_f32.par_iter())
            .map(|(a, b)| a * b)
            .collect();

        self.from_f32_slice(shape, &result)
    }

    fn relu(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.relu(&s_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let result: Vec<T> = in_slice
                .par_iter()
                .map(|val| if val.to_f32() > 0.0 { *val } else { T::zero() })
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sigmoid(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.sigmoid(&s_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let result: Vec<T> = in_slice
                .par_iter()
                .map(|val| T::from_f32(1.0 / (1.0 + (-val.to_f32()).exp())))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn relu_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let i_f32 = self.cast(in_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.relu_backward(&g_f32, &i_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let in_slice = in_primal.as_slice::<T>();
            let result: Vec<T> = grad_slice
                .par_iter()
                .zip(in_slice.par_iter())
                .map(|(g, i)| if i.to_f32() > 0.0 { *g } else { T::zero() })
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sigmoid_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let o_f32 = self.cast(out_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.sigmoid_backward(&g_f32, &o_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let out_slice = out_primal.as_slice::<T>();
            let result: Vec<T> = grad_slice
                .par_iter()
                .zip(out_slice.par_iter())
                .map(|(g, y)| g.mul(*y).mul(T::one().sub(*y)))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn expand(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, in_shape, dtype, DType::F32)?;
            let res_f32 = self.expand(&s_f32, in_shape, out_shape, DType::F32)?;
            return self.cast(&res_f32, out_shape, DType::F32, dtype);
        }

        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = vec![T::zero(); out_shape.num_elements()];

            let padded_in = Self::pad_shape_left(&in_shape.0, out_shape.rank());
            let in_strides = Self::compute_strides(&padded_in);

            for i in 0..result.len() {
                let multi_out = Self::linear_to_multi(i, &out_shape.0);
                let mut multi_in = vec![0; out_shape.rank()];
                for d in 0..out_shape.rank() {
                    multi_in[d] = if padded_in[d] == 1 { 0 } else { multi_out[d] };
                }
                let in_idx = Self::multi_to_linear(&multi_in, &in_strides);
                result[i] = in_slice[in_idx];
            }
            self.from_slice::<T>(out_shape, &result, dtype)
        })
    }

    fn sum_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, in_shape, dtype, DType::F32)?;
            let res_f32 = self.sum_to_shape(&s_f32, in_shape, out_shape, DType::F32)?;
            return self.cast(&res_f32, out_shape, DType::F32, dtype);
        }

        dispatch_dtype!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut result = vec![T::zero(); out_shape.num_elements()];

            let padded_out = Self::pad_shape_left(&out_shape.0, in_shape.rank());
            let out_strides = Self::compute_strides(&padded_out);

            for i in 0..in_slice.len() {
                let multi_in = Self::linear_to_multi(i, &in_shape.0);
                let mut multi_out = vec![0; in_shape.rank()];
                for d in 0..in_shape.rank() {
                    multi_out[d] = if padded_out[d] == 1 { 0 } else { multi_in[d] };
                }
                let out_idx = Self::multi_to_linear(&multi_out, &out_strides);
                result[out_idx] = result[out_idx].add(in_slice[i]);
            }
            self.from_slice::<T>(out_shape, &result, dtype)
        })
    }

    fn reshape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        _dtype: DType,
    ) -> Result<Self::Storage> {
        if in_shape.num_elements() != out_shape.num_elements() {
            return Err(OrcaError::ShapeMismatch {
                op: "reshape",
                expected: format!("{} elements", in_shape.num_elements()),
                got: format!("{} elements", out_shape.num_elements()),
            });
        }
        Ok(storage.clone()) // Since CPU storage is a flat contiguous Vec<u8> right now
    }

    fn exp(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.exp(&s_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let result: Vec<T> = in_slice
                .par_iter()
                .map(|val| T::from_f32(val.to_f32().exp()))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn log(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.log(&s_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let result: Vec<T> = in_slice
                .par_iter()
                .map(|val| T::from_f32((val.to_f32().max(1e-7)).ln()))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn exp_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let o_f32 = self.cast(out_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.exp_backward(&g_f32, &o_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let out_slice = out_primal.as_slice::<T>();
            let result: Vec<T> = grad_slice
                .par_iter()
                .zip(out_slice.par_iter())
                .map(|(g, y)| g.mul(*y))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn log_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let i_f32 = self.cast(in_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.log_backward(&g_f32, &i_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let in_slice = in_primal.as_slice::<T>();
            let result: Vec<T> = grad_slice
                .par_iter()
                .zip(in_slice.par_iter())
                .map(|(g, x)| g.div(*x))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn div(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let lhs_f32 = self.cast(lhs, shape, dtype, DType::F32)?;
            let rhs_f32 = self.cast(rhs, shape, dtype, DType::F32)?;
            let res_f32 = self.div(&lhs_f32, &rhs_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let lhs_slice = lhs.as_slice::<T>();
            let rhs_slice = rhs.as_slice::<T>();
            let result: Vec<T> = lhs_slice
                .par_iter()
                .zip(rhs_slice.par_iter())
                .map(|(a, b)| a.div(*b))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sqrt(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let s_f32 = self.cast(storage, shape, dtype, DType::F32)?;
            let res_f32 = self.sqrt(&s_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let result: Vec<T> = in_slice
                .par_iter()
                .map(|val| T::from_f32(val.to_f32().sqrt()))
                .collect();
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn div_backward_lhs(
        &self,
        grad_out: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let r_f32 = self.cast(rhs_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.div_backward_lhs(&g_f32, &r_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let rhs_slice = rhs_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                result.push(grad_slice[i].div(rhs_slice[i]));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn div_backward_rhs(
        &self,
        grad_out: &Self::Storage,
        lhs_primal: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let l_f32 = self.cast(lhs_primal, shape, dtype, DType::F32)?;
            let r_f32 = self.cast(rhs_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.div_backward_rhs(&g_f32, &l_f32, &r_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let lhs_slice = lhs_primal.as_slice::<T>();
            let rhs_slice = rhs_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                let grad_f32 = grad_slice[i].to_f32();
                let l_f32 = lhs_slice[i].to_f32();
                let r_f32 = rhs_slice[i].to_f32();
                result.push(T::from_f32(grad_f32 * (-l_f32 / (r_f32 * r_f32))));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn sqrt_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let g_f32 = self.cast(grad_out, shape, dtype, DType::F32)?;
            let o_f32 = self.cast(out_primal, shape, dtype, DType::F32)?;
            let res_f32 = self.sqrt_backward(&g_f32, &o_f32, shape, DType::F32)?;
            return self.cast(&res_f32, shape, DType::F32, dtype);
        }

        dispatch_float!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let out_slice = out_primal.as_slice::<T>();
            let mut result = Vec::with_capacity(grad_slice.len());
            for i in 0..grad_slice.len() {
                let grad_f32 = grad_slice[i].to_f32();
                let out_f32 = out_slice[i].to_f32();
                result.push(T::from_f32(grad_f32 / (2.0 * out_f32)));
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn accumulate_grad(&self, lhs: &Self::Storage, rhs: &Self::Storage) -> Result<Self::Storage> {
        let lhs_f32 =
            unsafe { std::slice::from_raw_parts(lhs.as_bytes().as_ptr() as *const f32, lhs.len()) };
        let rhs_f32 =
            unsafe { std::slice::from_raw_parts(rhs.as_bytes().as_ptr() as *const f32, rhs.len()) };

        let mut result = Vec::with_capacity(lhs.len());
        for i in 0..lhs.len() {
            result.push(lhs_f32[i] + rhs_f32[i]);
        }

        let dummy_shape = Shape::new(vec![lhs.len()]);
        self.from_slice::<f32>(&dummy_shape, &result, DType::F32)
    }

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
    ) -> Result<Self::Storage> {
        if dtype != DType::F32 {
            return Err(OrcaError::UnsupportedDType {
                op: "conv2d",
                dtype,
            });
        }

        let n = in_shape[0];
        let c_in = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let in_c_per_g = c_in / groups;
        let out_c_per_g = c_out / groups;

        let in_f32 = unsafe {
            std::slice::from_raw_parts(input.as_bytes().as_ptr() as *const f32, input.len())
        };
        let w_f32 = unsafe {
            std::slice::from_raw_parts(weight.as_bytes().as_ptr() as *const f32, weight.len())
        };
        let b_f32 = bias.map(|b| unsafe {
            std::slice::from_raw_parts(b.as_bytes().as_ptr() as *const f32, b.len())
        });

        let mut result = vec![0.0; n * c_out * h_out * w_out];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0;
                        let g = oc / out_c_per_g;
                        for ic in (g * in_c_per_g)..((g + 1) * in_c_per_g) {
                            let ic_w = ic % in_c_per_g;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride) as isize - padding as isize
                                        + (kh * dilation) as isize;
                                    let iw = (ow * stride) as isize - padding as isize
                                        + (kw * dilation) as isize;

                                    if ih >= 0
                                        && ih < h_in as isize
                                        && iw >= 0
                                        && iw < w_in as isize
                                    {
                                        let in_idx = b * (c_in * h_in * w_in)
                                            + ic * (h_in * w_in)
                                            + (ih as usize) * w_in
                                            + (iw as usize);
                                        let w_idx = oc * (in_c_per_g * k_h * k_w)
                                            + ic_w * (k_h * k_w)
                                            + kh * k_w
                                            + kw;
                                        sum += in_f32[in_idx] * w_f32[w_idx];
                                    }
                                }
                            }
                        }
                        if let Some(bias_slice) = b_f32 {
                            sum += bias_slice[oc];
                        }
                        let out_idx =
                            b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        result[out_idx] = sum;
                    }
                }
            }
        }

        let out_shape = Shape::new(vec![n, c_out, h_out, w_out]);
        self.from_f32_slice(&out_shape, &result)
    }

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
        _dtype: DType,
    ) -> Result<Self::Storage> {
        let n = in_shape[0];
        let c_in = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let in_c_per_g = c_in / groups;
        let out_c_per_g = c_out / groups;

        let go_f32 = unsafe {
            std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len())
        };
        let w_f32 = unsafe {
            std::slice::from_raw_parts(weight.as_bytes().as_ptr() as *const f32, weight.len())
        };

        let mut grad_in = vec![0.0; n * c_in * h_in * w_in];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx =
                            b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        let g_val = go_f32[go_idx];
                        let g = oc / out_c_per_g;
                        for ic in (g * in_c_per_g)..((g + 1) * in_c_per_g) {
                            let ic_w = ic % in_c_per_g;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride) as isize - padding as isize
                                        + (kh * dilation) as isize;
                                    let iw = (ow * stride) as isize - padding as isize
                                        + (kw * dilation) as isize;

                                    if ih >= 0
                                        && ih < h_in as isize
                                        && iw >= 0
                                        && iw < w_in as isize
                                    {
                                        let in_idx = b * (c_in * h_in * w_in)
                                            + ic * (h_in * w_in)
                                            + (ih as usize) * w_in
                                            + (iw as usize);
                                        let w_idx = oc * (in_c_per_g * k_h * k_w)
                                            + ic_w * (k_h * k_w)
                                            + kh * k_w
                                            + kw;
                                        grad_in[in_idx] += g_val * w_f32[w_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.from_f32_slice(in_shape, &grad_in)
    }

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
        _dtype: DType,
    ) -> Result<Self::Storage> {
        let n = in_shape[0];
        let c_in = in_shape[1];
        let h_in = in_shape[2];
        let w_in = in_shape[3];

        let c_out = weight_shape[0];
        let k_h = weight_shape[2];
        let k_w = weight_shape[3];

        let h_out = (h_in + 2 * padding - dilation * (k_h - 1) - 1) / stride + 1;
        let w_out = (w_in + 2 * padding - dilation * (k_w - 1) - 1) / stride + 1;

        let in_c_per_g = c_in / groups;
        let out_c_per_g = c_out / groups;

        let go_f32 = unsafe {
            std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len())
        };
        let in_f32 = unsafe {
            std::slice::from_raw_parts(input.as_bytes().as_ptr() as *const f32, input.len())
        };

        let mut grad_w = vec![0.0; c_out * c_in * k_h * k_w];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx =
                            b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        let g_val = go_f32[go_idx];

                        let g = oc / out_c_per_g;
                        for ic in (g * in_c_per_g)..((g + 1) * in_c_per_g) {
                            let ic_w = ic % in_c_per_g;
                            for kh in 0..k_h {
                                for kw in 0..k_w {
                                    let ih = (oh * stride) as isize - padding as isize
                                        + (kh * dilation) as isize;
                                    let iw = (ow * stride) as isize - padding as isize
                                        + (kw * dilation) as isize;

                                    if ih >= 0
                                        && ih < h_in as isize
                                        && iw >= 0
                                        && iw < w_in as isize
                                    {
                                        let in_idx = b * (c_in * h_in * w_in)
                                            + ic * (h_in * w_in)
                                            + (ih as usize) * w_in
                                            + (iw as usize);
                                        let w_idx = oc * (in_c_per_g * k_h * k_w)
                                            + ic_w * (k_h * k_w)
                                            + kh * k_w
                                            + kw;
                                        grad_w[w_idx] += g_val * in_f32[in_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        self.from_f32_slice(weight_shape, &grad_w)
    }

    fn conv2d_backward_bias(
        &self,
        grad_out: &Self::Storage,
        out_shape: &Shape,
        _dtype: DType,
    ) -> Result<Self::Storage> {
        let n = out_shape[0];
        let c_out = out_shape[1];
        let h_out = out_shape[2];
        let w_out = out_shape[3];

        let go_f32 = unsafe {
            std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len())
        };
        let mut grad_b = vec![0.0; c_out];

        for b in 0..n {
            for oc in 0..c_out {
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let go_idx =
                            b * (c_out * h_out * w_out) + oc * (h_out * w_out) + oh * w_out + ow;
                        grad_b[oc] += go_f32[go_idx];
                    }
                }
            }
        }
        self.from_f32_slice(&Shape::new(vec![c_out]), &grad_b)
    }

    fn cast(
        &self,
        storage: &Self::Storage,
        _shape: &Shape,
        current_dtype: DType,
        target_dtype: DType,
    ) -> Result<Self::Storage> {
        if current_dtype == target_dtype {
            return Ok(storage.clone());
        }

        if current_dtype == DType::F32 && target_dtype == DType::F64 {
            let input = storage.as_slice::<f32>();
            let output: Vec<f64> = input.iter().map(|&value| value as f64).collect();
            return self.from_slice::<f64>(_shape, &output, target_dtype);
        }

        if current_dtype == DType::F64 && target_dtype == DType::F32 {
            let input = storage.as_slice::<f64>();
            let output: Vec<f32> = input.iter().map(|&value| value as f32).collect();
            return self.from_slice::<f32>(_shape, &output, target_dtype);
        }

        // F32 -> F16
        if current_dtype == DType::F32 && target_dtype == DType::F16 {
            let in_slice = storage.as_slice::<f32>();
            let mut out = CpuByteStorage::new(storage.len() * 2, storage.len(), 2)?;
            let out_slice = out.as_mut_slice::<u16>();
            for (i, &val) in in_slice.iter().enumerate() {
                out_slice[i] = DType::f32_to_f16(val);
            }
            return Ok(out);
        }

        // F32 -> BF16
        if current_dtype == DType::F32 && target_dtype == DType::BF16 {
            let in_slice = storage.as_slice::<f32>();
            let mut out = CpuByteStorage::new(storage.len() * 2, storage.len(), 2)?;
            let out_slice = out.as_mut_slice::<u16>();
            for (i, &val) in in_slice.iter().enumerate() {
                out_slice[i] = DType::f32_to_bf16(val);
            }
            return Ok(out);
        }

        // F16 -> F32
        if current_dtype == DType::F16 && target_dtype == DType::F32 {
            let in_slice = storage.as_slice::<u16>();
            let mut out = CpuByteStorage::new(storage.len() * 4, storage.len(), 4)?;
            let out_slice = out.as_mut_slice::<f32>();
            for (i, &val) in in_slice.iter().enumerate() {
                out_slice[i] = DType::f16_to_f32(val);
            }
            return Ok(out);
        }

        // BF16 -> F32
        if current_dtype == DType::BF16 && target_dtype == DType::F32 {
            let in_slice = storage.as_slice::<u16>();
            let mut out = CpuByteStorage::new(storage.len() * 4, storage.len(), 4)?;
            let out_slice = out.as_mut_slice::<f32>();
            for (i, &val) in in_slice.iter().enumerate() {
                out_slice[i] = DType::bf16_to_f32(val);
            }
            return Ok(out);
        }

        // F16 -> BF16 (via F32)
        if current_dtype == DType::F16 && target_dtype == DType::BF16 {
            let f32_storage = self.cast(storage, _shape, DType::F16, DType::F32)?;
            return self.cast(&f32_storage, _shape, DType::F32, DType::BF16);
        }

        // BF16 -> F16 (via F32)
        if current_dtype == DType::BF16 && target_dtype == DType::F16 {
            let f32_storage = self.cast(storage, _shape, DType::BF16, DType::F32)?;
            return self.cast(&f32_storage, _shape, DType::F32, DType::F16);
        }

        if current_dtype == DType::I32 && target_dtype == DType::F32 {
            let in_i32 = storage.as_slice::<i32>();
            let mut out = CpuByteStorage::new(storage.len() * 4, storage.len(), 4)?;
            let out_f32 = out.as_mut_slice::<f32>();
            for (i, &val) in in_i32.iter().enumerate() {
                out_f32[i] = val as f32;
            }
            return Ok(out);
        } else if current_dtype == DType::F32 && target_dtype == DType::I32 {
            let in_f32 = storage.as_slice::<f32>();
            let mut out = CpuByteStorage::new(storage.len() * 4, storage.len(), 4)?;
            let out_i32 = out.as_mut_slice::<i32>();
            for (i, &val) in in_f32.iter().enumerate() {
                out_i32[i] = val as i32;
            }
            return Ok(out);
        }

        Err(OrcaError::UnsupportedDType {
            op: "cast",
            dtype: target_dtype,
        })
    }

    fn scatter(
        &self,
        storage: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        src: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let base_slice = storage.as_slice::<T>();
            let src_slice = src.as_slice::<T>();
            // Assume index is stored as I32 internally for now (or F32 casted)
            let idx_slice = index.as_slice::<f32>(); // In our engine everything is f32 internally except when explicitly casted
                                                     // Wait, index is likely F32 internally if they didn't cast to I32!

            let mut result = base_slice.to_vec();
            let base_strides = Self::compute_strides(shape);

            for i in 0..index_shape.num_elements() {
                let multi = Self::linear_to_multi(i, index_shape);
                let idx_val = idx_slice[i] as usize;
                let mut base_multi = multi.clone();
                base_multi[dim] = idx_val;

                let base_idx = Self::multi_to_linear(&base_multi, &base_strides);
                result[base_idx] = src_slice[i];
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn gather(
        &self,
        storage: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        dispatch_dtype!(dtype, T, {
            let src_slice = storage.as_slice::<T>();
            let idx_slice = index.as_slice::<f32>();

            let mut result = vec![T::zero(); index_shape.num_elements()];
            let in_strides = Self::compute_strides(shape);

            for i in 0..index_shape.num_elements() {
                let mut multi = Self::linear_to_multi(i, index_shape);
                let idx_val = idx_slice[i] as usize;
                multi[dim] = idx_val;

                let src_idx = Self::multi_to_linear(&multi, &in_strides);
                result[i] = src_slice[src_idx];
            }
            self.from_slice::<T>(index_shape, &result, dtype)
        })
    }

    fn scatter_backward_src(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        // Gradient of scatter wrt src is just gather!
        // We gather the gradients from grad_out at the same indices.
        self.gather(grad_out, dim, index, shape, index_shape, dtype)
    }

    fn scatter_backward_base(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        // Gradient of scatter wrt base is grad_out, but 0 at the scattered indices.
        dispatch_dtype!(dtype, T, {
            let grad_slice = grad_out.as_slice::<T>();
            let idx_slice = index.as_slice::<f32>();

            let mut result = grad_slice.to_vec();
            let base_strides = Self::compute_strides(shape);

            for i in 0..index_shape.num_elements() {
                let multi = Self::linear_to_multi(i, index_shape);
                let idx_val = idx_slice[i] as usize;
                let mut base_multi = multi.clone();
                base_multi[dim] = idx_val;

                let base_idx = Self::multi_to_linear(&base_multi, &base_strides);
                result[base_idx] = T::zero(); // Zero out gradients for elements that were overwritten by src
            }
            self.from_slice::<T>(shape, &result, dtype)
        })
    }

    fn gather_backward(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        _dtype: DType,
    ) -> Result<Self::Storage> {
        let grad_out_f32 = unsafe {
            std::slice::from_raw_parts(grad_out.as_bytes().as_ptr() as *const f32, grad_out.len())
        };
        let index_f32 = unsafe {
            std::slice::from_raw_parts(index.as_bytes().as_ptr() as *const f32, index.len())
        };

        let mut grad_in = vec![0.0; shape.num_elements()];
        let _out_strides = Self::compute_strides(&index_shape.0);
        let in_strides = Self::compute_strides(&shape.0);

        for i in 0..index_shape.num_elements() {
            let out_multi = Self::linear_to_multi(i, &index_shape.0);
            let mut in_multi = out_multi.clone();
            if in_multi.len() > dim {
                in_multi[dim] = index_f32[i] as usize;
            }
            let in_idx = Self::multi_to_linear(&in_multi, &in_strides);
            grad_in[in_idx] += grad_out_f32[i];
        }

        self.from_f32_slice(shape, &grad_in)
    }

    fn from_bytes(&self, shape: &Shape, bytes: &[u8], dtype: DType) -> Result<Self::Storage> {
        let expected_len = shape.num_elements() * dtype.element_size();
        if bytes.len() != expected_len {
            return Err(OrcaError::InternalError(format!(
                "from_bytes: expected {} bytes, got {}",
                expected_len,
                bytes.len()
            )));
        }
        let mut storage =
            CpuByteStorage::new(expected_len, shape.num_elements(), dtype.element_size())?;
        storage.as_mut_bytes().copy_from_slice(bytes);
        Ok(storage)
    }

    fn to_bytes(&self, storage: &Self::Storage) -> Result<Vec<u8>> {
        let bytes = storage.as_bytes();
        Ok(bytes.to_vec())
    }

    fn has_nan_or_inf(&self, storage: &Self::Storage, dtype: DType) -> Result<bool> {
        if dtype == DType::F16 || dtype == DType::BF16 {
            let shape = Shape::new(vec![storage.len()]);
            let f32_storage = self.cast(storage, &shape, dtype, DType::F32)?;
            return self.has_nan_or_inf(&f32_storage, DType::F32);
        }
        dispatch_float!(dtype, T, {
            let slice = storage.as_slice::<T>();
            for &val in slice {
                if !CpuNumeric::is_finite(val) {
                    return Ok(true);
                }
            }
            Ok(false)
        })
    }

    fn max_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let in_slice = storage.as_slice::<T>();
            let mut out_data = vec![T::min_value(); out_shape.num_elements()];

            let mut padded_out_shape = out_shape.0.clone();
            if out_shape.rank() < in_shape.rank() {
                padded_out_shape = Self::pad_shape_left(&out_shape.0, in_shape.rank());
            }

            let _in_strides = Self::compute_strides(&in_shape.0);
            let out_strides = Self::compute_strides(&padded_out_shape);

            for i in 0..in_shape.num_elements() {
                let in_multi = Self::linear_to_multi(i, &in_shape.0);
                let mut out_multi = vec![0; in_shape.rank()];

                for j in 0..in_shape.rank() {
                    out_multi[j] = if padded_out_shape[j] == 1 {
                        0
                    } else {
                        in_multi[j]
                    };
                }

                let out_idx = Self::multi_to_linear(&out_multi, &out_strides);
                let val = in_slice[i];
                if val > out_data[out_idx] {
                    out_data[out_idx] = val;
                }
            }
            let mut out_storage = CpuByteStorage::new(
                out_shape.num_elements() * std::mem::size_of::<T>(),
                out_shape.num_elements(),
                std::mem::size_of::<T>(),
            )?;
            out_storage.as_mut_slice::<T>().copy_from_slice(&out_data);
            Ok(out_storage)
        })
    }

    fn max_to_shape_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        out_primal: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        dispatch_float!(dtype, T, {
            let grad_out_slice = grad_out.as_slice::<T>();
            let in_primal_slice = in_primal.as_slice::<T>();
            let out_primal_slice = out_primal.as_slice::<T>();

            let mut grad_in = vec![T::zero(); in_shape.num_elements()];

            let mut padded_out_shape = out_shape.0.clone();
            if out_shape.rank() < in_shape.rank() {
                padded_out_shape = Self::pad_shape_left(&out_shape.0, in_shape.rank());
            }

            let _in_strides = Self::compute_strides(&in_shape.0);
            let out_strides = Self::compute_strides(&padded_out_shape);

            for i in 0..in_shape.num_elements() {
                let in_multi = Self::linear_to_multi(i, &in_shape.0);
                let mut out_multi = vec![0; in_shape.rank()];
                for j in 0..in_shape.rank() {
                    out_multi[j] = if padded_out_shape[j] == 1 {
                        0
                    } else {
                        in_multi[j]
                    };
                }

                let out_idx = Self::multi_to_linear(&out_multi, &out_strides);

                // If it matches the max value, it gets the gradient
                if in_primal_slice[i] == out_primal_slice[out_idx] {
                    grad_in[i] = grad_in[i].add(grad_out_slice[out_idx]);
                }
            }
            let mut grad_in_storage = CpuByteStorage::new(
                in_shape.num_elements() * std::mem::size_of::<T>(),
                in_shape.num_elements(),
                std::mem::size_of::<T>(),
            )?;
            grad_in_storage
                .as_mut_slice::<T>()
                .copy_from_slice(&grad_in);
            Ok(grad_in_storage)
        })
    }
}
