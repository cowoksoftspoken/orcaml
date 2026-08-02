use orca_autograd::{tensor::AutogradTensorExt, Autodiff};
use orca_backend_cpu::CpuBackend;
use orca_backend_gpu::GpuBackend;
use orca_core::{DType, Device, Shape};
use orca_serialize::{load_tensors as rust_load, save_tensors as rust_save};
use orca_tensor::Tensor;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(FromPyObject)]
pub enum Operand<'a> {
    Scalar(f32),
    Tensor(PyRef<'a, PyTensor>),
}

fn global_backend_cpu() -> Autodiff<CpuBackend> {
    static BACKEND: OnceLock<Autodiff<CpuBackend>> = OnceLock::new();
    BACKEND.get_or_init(|| Autodiff::new(CpuBackend)).clone()
}

fn global_backend_gpu() -> PyResult<Autodiff<GpuBackend>> {
    static BACKEND: OnceLock<Result<Autodiff<GpuBackend>, String>> = OnceLock::new();
    match BACKEND.get_or_init(|| {
        GpuBackend::new()
            .map(Autodiff::new)
            .map_err(|err| err.to_string())
    }) {
        Ok(backend) => Ok(backend.clone()),
        Err(message) => Err(PyRuntimeError::new_err(message.clone())),
    }
}

#[derive(Clone)]
pub enum PyTensorInner {
    Cpu(Tensor<Autodiff<CpuBackend>>),
    Gpu(Tensor<Autodiff<GpuBackend>>),
}

/// Python wrapper for Orca's DType
#[pyclass(name = "DType", module = "orca.core")]
#[derive(Clone)]
pub struct PyDType(DType);

#[pymethods]
impl PyDType {
    #[classattr]
    const FLOAT32: PyDType = PyDType(DType::F32);
    #[classattr]
    const FLOAT64: PyDType = PyDType(DType::F64);
    #[classattr]
    const FLOAT16: PyDType = PyDType(DType::F16);
    #[classattr]
    const BFLOAT16: PyDType = PyDType(DType::BF16);
    #[classattr]
    const INT32: PyDType = PyDType(DType::I32);
    #[classattr]
    const INT64: PyDType = PyDType(DType::I64);
    #[classattr]
    const UINT8: PyDType = PyDType(DType::U8);
    #[classattr]
    const BOOL: PyDType = PyDType(DType::Bool);

    fn __richcmp__(&self, other: &Self, op: pyo3::pyclass::CompareOp) -> PyResult<bool> {
        match op {
            pyo3::pyclass::CompareOp::Eq => Ok(self.0 == other.0),
            pyo3::pyclass::CompareOp::Ne => Ok(self.0 != other.0),
            _ => Err(pyo3::exceptions::PyTypeError::new_err(
                "Only Eq and Ne are supported",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!("orca.{}", self.0)
    }
}

/// Python wrapper for Orca's Device
#[pyclass(name = "Device", module = "orca.core")]
#[derive(Clone)]
pub struct PyDevice(Device);

#[pymethods]
impl PyDevice {
    #[new]
    fn new(device_str: &str) -> PyResult<Self> {
        match device_str {
            "cpu" => Ok(Self(Device::Cpu)),
            s if s.starts_with("cuda") || s.starts_with("gpu") => {
                let idx: usize = s.split(':').nth(1).unwrap_or("0").parse().unwrap_or(0);
                Ok(Self(Device::Gpu(idx)))
            }
            _ => Err(PyValueError::new_err(format!(
                "Unknown device: {}",
                device_str
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!("device(type='{}')", self.0)
    }
}

/// A multi-dimensional array with optional automatic differentiation.
#[pyclass(name = "Tensor", module = "orca.core", subclass)]
pub struct PyTensor(pub PyTensorInner);

macro_rules! dispatch_val {
    ($self:expr, $inner:ident, $expr:expr) => {
        match &$self.0 {
            PyTensorInner::Cpu($inner) => $expr,
            PyTensorInner::Gpu($inner) => $expr,
        }
    };
}

macro_rules! dispatch_tensor {
    ($self:expr, $inner:ident, $expr:expr) => {
        match &$self.0 {
            PyTensorInner::Cpu($inner) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            }
            PyTensorInner::Gpu($inner) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
        }
    };
}

macro_rules! dispatch_binary {
    ($self:expr, $other:expr, $inner:ident, $other_inner:ident, $expr:expr) => {
        match (&$self.0, &$other.0) {
            (PyTensorInner::Cpu($inner), PyTensorInner::Cpu($other_inner)) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            }
            (PyTensorInner::Gpu($inner), PyTensorInner::Gpu($other_inner)) => {
                let res = $expr.map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
            _ => Err(PyValueError::new_err(
                "Cannot operate on tensors from different devices",
            )),
        }
    };
}

#[pymethods]
impl PyTensor {
    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    fn zeros(
        shape: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::zeros(global_backend_gpu()?, shape, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::zeros(global_backend_cpu(), shape, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, dtype=None, device=None, requires_grad=false))]
    fn ones(
        shape: Vec<usize>,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::ones(global_backend_gpu()?, shape, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::ones(global_backend_cpu(), shape, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (value, dtype=None, device=None, requires_grad=false))]
    fn scalar(
        value: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::scalar(global_backend_gpu()?, value, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::scalar(global_backend_cpu(), value, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, mean=0.0, std=1.0, dtype=None, device=None, requires_grad=false))]
    fn randn(
        shape: Vec<usize>,
        mean: f32,
        std: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::randn(global_backend_gpu()?, shape, mean, std, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::randn(global_backend_cpu(), shape, mean, std, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, low, high, dtype=None, device=None, requires_grad=false))]
    fn rand_uniform(
        shape: Vec<usize>,
        low: f32,
        high: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
        requires_grad: bool,
    ) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::rand_uniform(global_backend_gpu()?, shape, low, high, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::rand_uniform(global_backend_cpu(), shape, low, high, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    #[staticmethod]
    #[pyo3(signature = (shape, p, dtype=None, device=None))]
    fn rand_dropout_mask(
        shape: Vec<usize>,
        p: f32,
        dtype: Option<PyDType>,
        device: Option<PyDevice>,
    ) -> PyResult<Self> {
        let dtype = dtype.map(|d| d.0).unwrap_or(DType::F32);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_)),
            None => false,
        };

        if is_gpu {
            let tensor = Tensor::rand_dropout_mask(global_backend_gpu()?, shape, p, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let tensor = Tensor::rand_dropout_mask(global_backend_cpu(), shape, p, dtype)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }
    #[staticmethod]
    #[pyo3(signature = (equation, inputs))]
    fn einsum(equation: String, inputs: Vec<Py<PyTensor>>) -> PyResult<Self> {
        Python::with_gil(|py| {
            if inputs.is_empty() {
                return Err(PyValueError::new_err("Einsum requires at least one input"));
            }

            // Check devices and gather references
            let mut all_cpu = true;
            let mut all_gpu = true;

            for input in inputs.iter() {
                let cell = input.bind(py);
                let t = cell.borrow();
                match &t.0 {
                    PyTensorInner::Cpu(_) => all_gpu = false,
                    PyTensorInner::Gpu(_) => all_cpu = false,
                }
            }

            if !all_cpu && !all_gpu {
                return Err(PyValueError::new_err(
                    "Cannot mix CPU and GPU tensors in einsum",
                ));
            }

            if all_gpu {
                let mut gpu_refs = Vec::new();
                // We need to keep borrows alive.
                let guards: Vec<_> = inputs.iter().map(|i| i.bind(py).borrow()).collect();
                for g in &guards {
                    if let PyTensorInner::Gpu(t) = &g.0 {
                        gpu_refs.push(t);
                    }
                }
                let tensor = Tensor::einsum(&equation, &gpu_refs)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(tensor)))
            } else {
                let mut cpu_refs = Vec::new();
                let guards: Vec<_> = inputs.iter().map(|i| i.bind(py).borrow()).collect();
                for g in &guards {
                    if let PyTensorInner::Cpu(t) = &g.0 {
                        cpu_refs.push(t);
                    }
                }
                let tensor = Tensor::einsum(&equation, &cpu_refs)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(tensor)))
            }
        })
    }

    #[staticmethod]
    #[pyo3(signature = (data, shape=None, requires_grad=false, device=None))]
    fn from_list(
        data: Vec<f32>,
        shape: Option<Vec<usize>>,
        requires_grad: bool,
        device: Option<PyDevice>,
    ) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let is_gpu = match device {
            Some(d) => matches!(d.0, Device::Gpu(_) | Device::Cuda(_)),
            None => false,
        };

        if is_gpu {
            let mut tensor = Tensor::from_f32_slice(global_backend_gpu()?, &data, shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Gpu(tensor)))
        } else {
            let mut tensor = Tensor::from_f32_slice(global_backend_cpu(), &data, shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if requires_grad {
                tensor.require_grad();
            }
            Ok(Self(PyTensorInner::Cpu(tensor)))
        }
    }

    fn to(&self, device_str: &str) -> PyResult<Self> {
        let is_gpu = device_str.starts_with("cuda") || device_str.starts_with("gpu");
        let current_is_gpu = matches!(self.0, PyTensorInner::Gpu(_));

        if is_gpu == current_is_gpu {
            return Ok(PyTensor(self.0.clone()));
        }

        let data = self.to_list()?;
        let shape = self.shape();
        let requires_grad = self.requires_grad();

        Self::from_list(
            data,
            Some(shape),
            requires_grad,
            Some(PyDevice::new(device_str)?),
        )
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        dispatch_val!(self, t, t.shape().to_vec())
    }

    #[getter]
    fn dtype(&self) -> PyDType {
        dispatch_val!(self, t, PyDType(t.dtype()))
    }

    #[getter]
    fn device(&self) -> PyDevice {
        dispatch_val!(self, t, PyDevice(t.device()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Tensor(shape={:?}, dtype={}, device={})",
            self.shape(),
            self.dtype().0,
            self.device().0
        )
    }

    fn to_list(&self) -> PyResult<Vec<f32>> {
        let res = match &self.0 {
            PyTensorInner::Cpu(t) => {
                let f32_t = if t.dtype() != DType::F32 {
                    t.to_dtype(DType::F32)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    t.clone()
                };
                f32_t
                    .to_f32_vec()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            }
            PyTensorInner::Gpu(t) => {
                let f32_t = if t.dtype() != DType::F32 {
                    t.to_dtype(DType::F32)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    t.clone()
                };
                f32_t
                    .to_f32_vec()
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            }
        };
        Ok(res)
    }

    fn resolve_operand(&self, other: Operand) -> PyResult<PyTensor> {
        match other {
            Operand::Scalar(s) => {
                let is_gpu = matches!(self.0, PyTensorInner::Gpu(_));
                if is_gpu {
                    let tensor = Tensor::scalar(global_backend_gpu()?, s, self.dtype().0)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(PyTensor(PyTensorInner::Gpu(tensor)))
                } else {
                    let tensor = Tensor::scalar(global_backend_cpu(), s, self.dtype().0)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?;
                    Ok(PyTensor(PyTensorInner::Cpu(tensor)))
                }
            }
            Operand::Tensor(t) => Ok(PyTensor(t.0.clone())),
        }
    }

    fn __add__(&self, operand: Operand) -> PyResult<Self> {
        let other = self.resolve_operand(operand)?;
        dispatch_binary!(self, other, s, o, {
            let out_shape = s
                .shape()
                .broadcast(o.shape())
                .ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b + &o_b
        })
    }
    fn __radd__(&self, operand: Operand) -> PyResult<Self> {
        self.__add__(operand)
    }

    fn __sub__(&self, operand: Operand) -> PyResult<Self> {
        let other = self.resolve_operand(operand)?;
        dispatch_binary!(self, other, s, o, {
            let out_shape = s
                .shape()
                .broadcast(o.shape())
                .ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b - &o_b
        })
    }
    fn __rsub__(&self, operand: Operand) -> PyResult<Self> {
        let other = self.resolve_operand(operand)?;
        dispatch_binary!(other, self, s, o, {
            let out_shape = s
                .shape()
                .broadcast(o.shape())
                .ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b - &o_b
        })
    }

    fn __mul__(&self, operand: Operand) -> PyResult<Self> {
        match operand {
            Operand::Scalar(s_val) => dispatch_tensor!(self, t, t.mul_scalar(s_val)),
            Operand::Tensor(other) => dispatch_binary!(self, other, s, o, {
                let out_shape = s
                    .shape()
                    .broadcast(o.shape())
                    .ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
                let s_b = s
                    .expand(&out_shape)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                let o_b = o
                    .expand(&out_shape)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                &s_b * &o_b
            }),
        }
    }
    fn __rmul__(&self, operand: Operand) -> PyResult<Self> {
        self.__mul__(operand)
    }

    fn __matmul__(&self, other: &Self) -> PyResult<Self> {
        dispatch_binary!(self, other, s, o, s.matmul(o))
    }

    fn __truediv__(&self, operand: Operand) -> PyResult<Self> {
        let other = self.resolve_operand(operand)?;
        dispatch_binary!(self, other, s, o, {
            let out_shape = s
                .shape()
                .broadcast(o.shape())
                .ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b / &o_b
        })
    }
    fn __rtruediv__(&self, operand: Operand) -> PyResult<Self> {
        let other = self.resolve_operand(operand)?;
        dispatch_binary!(other, self, s, o, {
            let out_shape = s
                .shape()
                .broadcast(o.shape())
                .ok_or_else(|| PyValueError::new_err("Shapes are not broadcastable"))?;
            let s_b = s
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let o_b = o
                .expand(&out_shape)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            &s_b / &o_b
        })
    }

    fn relu(&self) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.relu())
    }
    fn sqrt(&self) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.sqrt())
    }
    fn sigmoid(&self) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.sigmoid())
    }
    fn exp(&self) -> PyResult<Self> {
        dispatch_tensor!(self, inner, inner.exp())
    }

    fn log(&self) -> PyResult<Self> {
        dispatch_tensor!(self, inner, inner.log())
    }

    fn __neg__(&self) -> PyResult<Self> {
        // -x is exactly x * -1.0
        match &self.0 {
            PyTensorInner::Cpu(inner) => {
                let res = inner
                    .mul_scalar(-1.0)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            }
            PyTensorInner::Gpu(inner) => {
                let res = inner
                    .mul_scalar(-1.0)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
        }
    }

    fn sum(&self) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.sum())
    }

    fn mean(&self) -> PyResult<Self> {
        dispatch_tensor!(self, t, {
            t.sum()
                .and_then(|sum| sum.mul_scalar(1.0 / t.shape().num_elements() as f32))
        })
    }

    #[pyo3(signature = (weight, bias=None, padding=0, stride=1, dilation=1, groups=1))]
    fn conv2d(
        &self,
        weight: &Self,
        bias: Option<&Self>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> PyResult<Self> {
        match (&self.0, &weight.0) {
            (PyTensorInner::Cpu(o), PyTensorInner::Cpu(w)) => {
                let res = match bias {
                    Some(b) => {
                        if let PyTensorInner::Cpu(b_in) = &b.0 {
                            o.conv2d(w, Some(b_in), padding, stride, dilation, groups)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?
                        } else {
                            return Err(PyValueError::new_err(
                                "Cannot operate on tensors from different devices",
                            ));
                        }
                    }
                    None => o
                        .conv2d(w, None, padding, stride, dilation, groups)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                };
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            }
            (PyTensorInner::Gpu(o), PyTensorInner::Gpu(w)) => {
                let res = match bias {
                    Some(b) => {
                        if let PyTensorInner::Gpu(b_in) = &b.0 {
                            o.conv2d(w, Some(b_in), padding, stride, dilation, groups)
                                .map_err(|e| PyValueError::new_err(e.to_string()))?
                        } else {
                            return Err(PyValueError::new_err(
                                "Cannot operate on tensors from different devices",
                            ));
                        }
                    }
                    None => o
                        .conv2d(w, None, padding, stride, dilation, groups)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?,
                };
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
            _ => Err(PyValueError::new_err(
                "Cannot operate on tensors from different devices",
            )),
        }
    }

    fn expand(&self, shape: Vec<usize>) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.expand(&Shape::new(shape)))
    }
    fn reshape(&self, shape: Vec<usize>) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.reshape(&Shape::new(shape)))
    }
    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.transpose(dim0, dim1))
    }
    fn gather(&self, dim: usize, index: &PyTensor) -> PyResult<Self> {
        match (&self.0, &index.0) {
            (PyTensorInner::Cpu(t), PyTensorInner::Cpu(i)) => {
                let res = t
                    .gather(dim, i)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            }
            (PyTensorInner::Gpu(t), PyTensorInner::Gpu(i)) => {
                let res = t
                    .gather(dim, i)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
            _ => Err(PyValueError::new_err(
                "Cannot operate on tensors from different devices",
            )),
        }
    }

    fn scatter(&self, dim: usize, index: &PyTensor, src: &PyTensor) -> PyResult<Self> {
        match (&self.0, &index.0, &src.0) {
            (PyTensorInner::Cpu(t), PyTensorInner::Cpu(i), PyTensorInner::Cpu(s)) => {
                let res = t
                    .scatter(dim, i, s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Cpu(res)))
            }
            (PyTensorInner::Gpu(t), PyTensorInner::Gpu(i), PyTensorInner::Gpu(s)) => {
                let res = t
                    .scatter(dim, i, s)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                Ok(PyTensor(PyTensorInner::Gpu(res)))
            }
            _ => Err(PyValueError::new_err(
                "Cannot operate on tensors from different devices",
            )),
        }
    }
    fn sum_to_shape(&self, shape: Vec<usize>) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.sum_to_shape(&Shape::new(shape)))
    }
    fn max_to_shape(&self, shape: Vec<usize>) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.max_to_shape(&Shape::new(shape)))
    }

    fn has_nan_or_inf(&self) -> PyResult<bool> {
        dispatch_val!(
            self,
            t,
            t.has_nan_or_inf()
                .map_err(|e| PyValueError::new_err(e.to_string()))
        )
    }
    #[getter]
    fn requires_grad(&self) -> bool {
        dispatch_val!(self, t, t.storage().node_id.is_some())
    }

    fn require_grad(&mut self) -> PyResult<()> {
        match &mut self.0 {
            PyTensorInner::Cpu(t) => t.require_grad(),
            PyTensorInner::Gpu(t) => t.require_grad(),
        }
        Ok(())
    }

    fn zero_grad(&self) -> PyResult<()> {
        dispatch_val!(
            self,
            t,
            t.zero_grad()
                .map_err(|e| PyValueError::new_err(e.to_string()))
        )
    }

    fn backward(&self) -> PyResult<()> {
        dispatch_val!(
            self,
            t,
            t.backward()
                .map_err(|e| PyValueError::new_err(e.to_string()))
        )
    }

    fn grad(&self) -> PyResult<Option<Self>> {
        match &self.0 {
            PyTensorInner::Cpu(t) => Ok(t
                .grad()
                .map(|grad_tensor| PyTensor(PyTensorInner::Cpu(grad_tensor)))),
            PyTensorInner::Gpu(t) => Ok(t
                .grad()
                .map(|grad_tensor| PyTensor(PyTensorInner::Gpu(grad_tensor)))),
        }
    }

    fn detach(&self) -> PyResult<Self> {
        match &self.0 {
            PyTensorInner::Cpu(t) => Ok(PyTensor(PyTensorInner::Cpu(t.detach()))),
            PyTensorInner::Gpu(t) => Ok(PyTensor(PyTensorInner::Gpu(t.detach()))),
        }
    }

    fn to_dtype(&self, dtype: PyDType) -> PyResult<Self> {
        dispatch_tensor!(self, t, t.to_dtype(dtype.0))
    }

    fn set_grad(&self, grad: &Self) -> PyResult<()> {
        match (&self.0, &grad.0) {
            (PyTensorInner::Cpu(t), PyTensorInner::Cpu(g)) => t
                .set_grad(g)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            (PyTensorInner::Gpu(t), PyTensorInner::Gpu(g)) => t
                .set_grad(g)
                .map_err(|e| PyValueError::new_err(e.to_string())),
            _ => Err(PyValueError::new_err(
                "Cannot set_grad across different devices",
            )),
        }
    }
}

#[pyfunction]
fn set_grad_enabled(enabled: bool) {
    orca_autograd::set_grad_enabled(enabled);
}

#[pyfunction]
fn is_grad_enabled() -> bool {
    orca_autograd::is_grad_enabled()
}

#[pyfunction]
fn save_tensors(path: &str, tensors_dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
    let mut hm = HashMap::new();
    for (k, v) in tensors_dict {
        let key_str: String = k.extract()?;
        let py_tensor: PyRef<PyTensor> = v.extract()?;
        match &py_tensor.0 {
            PyTensorInner::Cpu(t) => {
                let inner = t.primal();
                hm.insert(key_str, inner);
            }
            PyTensorInner::Gpu(_) => {
                let cpu_tensor = py_tensor.to("cpu")?;
                if let PyTensorInner::Cpu(t) = cpu_tensor.0 {
                    let inner = t.primal();
                    hm.insert(key_str, inner);
                } else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "Failed to convert GPU tensor to CPU",
                    ));
                }
            }
        }
    }

    rust_save(&hm, path).map_err(|e| PyRuntimeError::new_err(format!("Save failed: {:?}", e)))?;
    Ok(())
}

#[pyfunction]
fn load_tensors(path: &str) -> PyResult<HashMap<String, PyTensor>> {
    let backend = global_backend_cpu();
    let loaded = rust_load(backend.clone(), path)
        .map_err(|e| PyRuntimeError::new_err(format!("Load failed: {:?}", e)))?;

    let mut out = HashMap::new();
    for (k, v) in loaded {
        let py_inner = PyTensorInner::Cpu(v);
        let py_tensor = PyTensor(py_inner);
        out.insert(k, py_tensor);
    }
    Ok(out)
}

#[pymodule]
fn orca_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDType>()?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(save_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(load_tensors, m)?)?;
    m.add_function(wrap_pyfunction!(set_grad_enabled, m)?)?;
    m.add_function(wrap_pyfunction!(is_grad_enabled, m)?)?;
    Ok(())
}
