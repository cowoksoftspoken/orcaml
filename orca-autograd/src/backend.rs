use crate::tape::{BackwardOp, Gradients, NodeId, Tape};
use orca_core::{DType, Device, OrcaError, Result, Shape};
use orca_tensor::{Backend, Storage};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Storage wrapper that holds the primal data and its tape NodeId.
#[derive(Clone, Debug)]
pub struct AutodiffStorage<S: Storage> {
    pub primal: S,
    pub node_id: Option<NodeId>,
}

impl<S: Storage> Storage for AutodiffStorage<S> {
    fn len(&self) -> usize {
        self.primal.len()
    }
}

/// The Autodiff Backend wrapper.
#[derive(Clone, Debug)]
pub struct Autodiff<B: Backend> {
    inner: B,
    tape: Arc<Mutex<Tape<B>>>,
    _phantom: PhantomData<B>,
}

impl<B: Backend> Autodiff<B> {
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            tape: Arc::new(Mutex::new(Tape::new())),
            _phantom: PhantomData,
        }
    }

    pub fn inner(&self) -> &B {
        &self.inner
    }

    pub fn tape(&self) -> Arc<Mutex<Tape<B>>> {
        Arc::clone(&self.tape)
    }
}

impl<B: Backend> Backend for Autodiff<B> {
    type Storage = AutodiffStorage<B::Storage>;

    fn device(&self) -> Device {
        self.inner.device()
    }

    fn zeros(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let primal = self.inner.zeros(shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn from_f32_slice(&self, shape: &Shape, data: &[f32]) -> Result<Self::Storage> {
        let primal = self.inner.from_f32_slice(shape, data)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None, // Will be set by python wrapper if requires_grad=True
        })
    }

    fn to_f32_vec(&self, storage: &Self::Storage) -> Result<Vec<f32>> {
        self.inner.to_f32_vec(&storage.primal)
    }

    fn add(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.add(&lhs.primal, &rhs.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        // If either input has a node_id, it means they require grad, so this operation requires grad.
        if lhs.node_id.is_some() || rhs.node_id.is_some() {
            let lhs_id = lhs.node_id;
            let rhs_id = rhs.node_id;
            let _out_shape = shape.clone();

            tape.push_node(Box::new(AddBackward {
                out_id,
                lhs_id,
                rhs_id,
            }));

            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn matmul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        lhs_shape: &Shape,
        rhs_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .matmul(&lhs.primal, &rhs.primal, lhs_shape, rhs_shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if lhs.node_id.is_some() || rhs.node_id.is_some() {
            tape.push_node(Box::new(MatMulBackward {
                out_id,
                lhs_id: lhs.node_id,
                rhs_id: rhs.node_id,
                lhs_primal: lhs.primal.clone(),
                rhs_primal: rhs.primal.clone(),
                lhs_shape: lhs_shape.clone(),
                rhs_shape: rhs_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }
    fn transpose(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dim0: usize,
        dim1: usize,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .transpose(&storage.primal, shape, dim0, dim1, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(TransposeBackward {
                out_id,
                in_id,
                dim0,
                dim1,
                shape: shape.clone(),
                dtype,
                _phantom: PhantomData,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }
    fn sub(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.sub(&lhs.primal, &rhs.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if lhs.node_id.is_some() || rhs.node_id.is_some() {
            tape.push_node(Box::new(SubBackward {
                out_id,
                lhs_id: lhs.node_id,
                rhs_id: rhs.node_id,
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn mul_scalar(
        &self,
        storage: &Self::Storage,
        scalar: f32,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .mul_scalar(&storage.primal, scalar, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(MulScalarBackward {
                out_id,
                in_id,
                scalar,
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn mul(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.mul(&lhs.primal, &rhs.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if lhs.node_id.is_some() || rhs.node_id.is_some() {
            tape.push_node(Box::new(MulBackward {
                out_id,
                lhs_id: lhs.node_id,
                rhs_id: rhs.node_id,
                lhs_primal: lhs.primal.clone(),
                rhs_primal: rhs.primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn div(
        &self,
        lhs: &Self::Storage,
        rhs: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.div(&lhs.primal, &rhs.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if lhs.node_id.is_some() || rhs.node_id.is_some() {
            tape.push_node(Box::new(DivBackward {
                out_id,
                lhs_id: lhs.node_id,
                rhs_id: rhs.node_id,
                lhs_primal: lhs.primal.clone(),
                rhs_primal: rhs.primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn relu(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let primal = self.inner.relu(&storage.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(ReluBackward {
                out_id,
                in_id,
                in_primal: storage.primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn sqrt(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let primal = self.inner.sqrt(&storage.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(SqrtBackwardOp {
                out_id,
                in_id,
                out_primal: primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn sigmoid(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.sigmoid(&storage.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            // Optimization: capture the output primal (y) instead of input (x).
            // y is exactly `primal`.
            tape.push_node(Box::new(SigmoidBackward {
                out_id,
                in_id,
                out_primal: primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn relu_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .relu_backward(&grad_out.primal, &in_primal.primal, shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        }) // Higher order gradients not supported yet
    }

    fn sigmoid_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal =
            self.inner
                .sigmoid_backward(&grad_out.primal, &out_primal.primal, shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        }) // Higher order gradients not supported yet
    }

    fn expand(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .expand(&storage.primal, in_shape, out_shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(ExpandBackward {
                out_id,
                in_id,
                in_shape: in_shape.clone(),
                out_shape: out_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn sum_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .sum_to_shape(&storage.primal, in_shape, out_shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(SumToShapeBackward {
                out_id,
                in_id,
                in_shape: in_shape.clone(),
                out_shape: out_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn reshape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .reshape(&storage.primal, in_shape, out_shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(ReshapeBackward {
                out_id,
                in_id,
                in_shape: in_shape.clone(),
                out_shape: out_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn exp(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let primal = self.inner.exp(&storage.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(ExpBackwardOp {
                out_id,
                in_id,
                out_primal: primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn log(&self, storage: &Self::Storage, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let primal = self.inner.log(&storage.primal, shape, dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(LogBackwardOp {
                out_id,
                in_id,
                in_primal: storage.primal.clone(),
                shape: shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn exp_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .exp_backward(&grad_out.primal, &out_primal.primal, shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn log_backward(
        &self,
        grad_out: &Self::Storage,
        in_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let inner_grad =
            self.inner
                .log_backward(&grad_out.primal, &in_primal.primal, shape, dtype)?;
        Ok(AutodiffStorage {
            primal: inner_grad,
            node_id: None,
        })
    }

    fn div_backward_lhs(
        &self,
        grad_out: &Self::Storage,
        rhs_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal =
            self.inner
                .div_backward_lhs(&grad_out.primal, &rhs_primal.primal, shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
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
        let primal = self.inner.div_backward_rhs(
            &grad_out.primal,
            &lhs_primal.primal,
            &rhs_primal.primal,
            shape,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn sqrt_backward(
        &self,
        grad_out: &Self::Storage,
        out_primal: &Self::Storage,
        shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal =
            self.inner
                .sqrt_backward(&grad_out.primal, &out_primal.primal, shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn accumulate_grad(&self, lhs: &Self::Storage, rhs: &Self::Storage) -> Result<Self::Storage> {
        let primal = self.inner.accumulate_grad(&lhs.primal, &rhs.primal)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
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
        let primal = self.inner.conv2d(
            &input.primal,
            &weight.primal,
            bias.map(|b| &b.primal),
            in_shape,
            weight_shape,
            padding,
            stride,
            dilation,
            groups,
            dtype,
        )?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if input.node_id.is_some()
            || weight.node_id.is_some()
            || bias.map(|b| b.node_id.is_some()).unwrap_or(false)
        {
            tape.push_node(Box::new(Conv2dBackward {
                out_id,
                in_id: input.node_id,
                weight_id: weight.node_id,
                bias_id: bias.and_then(|b| b.node_id),
                in_primal: input.primal.clone(),
                weight_primal: weight.primal.clone(),
                in_shape: in_shape.clone(),
                weight_shape: weight_shape.clone(),
                padding,
                stride,
                dilation,
                groups,
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
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
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.conv2d_backward_input(
            &grad_out.primal,
            &weight.primal,
            in_shape,
            weight_shape,
            padding,
            stride,
            dilation,
            groups,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
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
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.conv2d_backward_weight(
            &grad_out.primal,
            &input.primal,
            in_shape,
            weight_shape,
            padding,
            stride,
            dilation,
            groups,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn conv2d_backward_bias(
        &self,
        grad_out: &Self::Storage,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .conv2d_backward_bias(&grad_out.primal, out_shape, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn cast(
        &self,
        storage: &Self::Storage,
        shape: &Shape,
        current_dtype: DType,
        target_dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .cast(&storage.primal, shape, current_dtype, target_dtype)?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let Some(in_id) = storage.node_id else {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        };

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();
        tape.push_node(Box::new(CastBackward {
            out_id,
            in_id,
            shape: shape.clone(),
            current_dtype,
            target_dtype,
        }));

        Ok(AutodiffStorage {
            primal,
            node_id: Some(out_id),
        })
    }

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
    ) -> Result<Self::Storage> {
        let primal = self.inner.scatter(
            &storage.primal,
            dim,
            &index.primal,
            &src.primal,
            shape,
            index_shape,
            dtype,
        )?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if storage.node_id.is_some() || src.node_id.is_some() {
            tape.push_node(Box::new(ScatterBackward {
                out_id,
                base_id: storage.node_id,
                src_id: src.node_id,
                index_primal: index.primal.clone(),
                dim,
                shape: shape.clone(),
                index_shape: index_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
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
        let primal = self.inner.gather(
            &storage.primal,
            dim,
            &index.primal,
            shape,
            index_shape,
            dtype,
        )?;

        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(GatherBackwardOp {
                out_id,
                in_id,
                index_primal: index.primal.clone(),
                dim,
                shape: shape.clone(),
                index_shape: index_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
    }

    fn gather_backward(
        &self,
        grad_out: &Self::Storage,
        dim: usize,
        index: &Self::Storage,
        shape: &Shape,
        index_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self.inner.gather_backward(
            &grad_out.primal,
            dim,
            &index.primal,
            shape,
            index_shape,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    // Phase 1-2 Production Hardening
    fn from_bytes(&self, shape: &Shape, bytes: &[u8], dtype: DType) -> Result<Self::Storage> {
        let primal = self.inner.from_bytes(shape, bytes, dtype)?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }

    fn to_bytes(&self, storage: &Self::Storage) -> Result<Vec<u8>> {
        self.inner.to_bytes(&storage.primal)
    }

    fn has_nan_or_inf(&self, storage: &Self::Storage, dtype: DType) -> Result<bool> {
        self.inner.has_nan_or_inf(&storage.primal, dtype)
    }

    fn max_to_shape(
        &self,
        storage: &Self::Storage,
        in_shape: &Shape,
        out_shape: &Shape,
        dtype: DType,
    ) -> Result<Self::Storage> {
        let primal = self
            .inner
            .max_to_shape(&storage.primal, in_shape, out_shape, dtype)?;
        if !crate::is_grad_enabled() {
            return Ok(AutodiffStorage {
                primal,
                node_id: None,
            });
        }

        let mut tape = self
            .tape
            .lock()
            .map_err(|_| OrcaError::InternalError("Mutex poisoned".into()))?;
        let out_id = tape.generate_id();

        if let Some(in_id) = storage.node_id {
            tape.push_node(Box::new(MaxToShapeBackward {
                out_id,
                in_id,
                in_primal: storage.primal.clone(),
                out_primal: primal.clone(),
                in_shape: in_shape.clone(),
                out_shape: out_shape.clone(),
                dtype,
            }));
            Ok(AutodiffStorage {
                primal,
                node_id: Some(out_id),
            })
        } else {
            Ok(AutodiffStorage {
                primal,
                node_id: None,
            })
        }
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
        let primal = self.inner.max_to_shape_backward(
            &grad_out.primal,
            &in_primal.primal,
            &out_primal.primal,
            in_shape,
            out_shape,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
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
        let primal = self.inner.scatter_backward_src(
            &grad_out.primal,
            dim,
            &index.primal,
            shape,
            index_shape,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
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
        let primal = self.inner.scatter_backward_base(
            &grad_out.primal,
            dim,
            &index.primal,
            shape,
            index_shape,
            dtype,
        )?;
        Ok(AutodiffStorage {
            primal,
            node_id: None,
        })
    }
}

#[derive(Debug)]
struct CastBackward {
    out_id: NodeId,
    in_id: NodeId,
    shape: Shape,
    current_dtype: DType,
    target_dtype: DType,
}

impl<B: Backend> BackwardOp<B> for CastBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let Some(out_grad) = grads.get(self.out_id) else {
            return Ok(());
        };

        let grad = backend.cast(out_grad, &self.shape, self.target_dtype, self.current_dtype)?;
        grads.accumulate(self.in_id, grad, backend)
    }
}

/// Backward operation for Gather
#[derive(Debug)]
struct GatherBackwardOp<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    index_primal: B::Storage,
    dim: usize,
    shape: Shape,
    index_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for GatherBackwardOp<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = match grads.get(self.out_id) {
            Some(g) => g.clone(),
            None => return Ok(()),
        };
        let grad_in = backend.gather_backward(
            &out_grad,
            self.dim,
            &self.index_primal,
            &self.shape,
            &self.index_shape,
            self.dtype,
        )?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Scatter
#[derive(Debug)]
struct ScatterBackward<B: Backend> {
    out_id: NodeId,
    base_id: Option<NodeId>,
    src_id: Option<NodeId>,
    index_primal: B::Storage,
    dim: usize,
    shape: Shape,
    index_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for ScatterBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = match grads.get(self.out_id) {
            Some(g) => g.clone(),
            None => return Ok(()),
        };
        if let Some(b_id) = self.base_id {
            let grad_base = backend.scatter_backward_base(
                &out_grad,
                self.dim,
                &self.index_primal,
                &self.shape,
                &self.index_shape,
                self.dtype,
            )?;
            grads.accumulate(b_id, grad_base, backend)?;
        }
        if let Some(s_id) = self.src_id {
            let grad_src = backend.scatter_backward_src(
                &out_grad,
                self.dim,
                &self.index_primal,
                &self.shape,
                &self.index_shape,
                self.dtype,
            )?;
            grads.accumulate(s_id, grad_src, backend)?;
        }
        Ok(())
    }
}

/// Backward operation for Addition
#[derive(Debug)]
struct AddBackward {
    out_id: NodeId,
    lhs_id: Option<NodeId>,
    rhs_id: Option<NodeId>,
}

impl<B: Backend> BackwardOp<B> for AddBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        if let Some(out_grad) = grads.get(self.out_id) {
            let out_grad_cloned = out_grad.clone();
            let out_grad_cloned_2 = out_grad.clone();

            if let Some(lhs) = self.lhs_id {
                grads.accumulate(lhs, out_grad_cloned, backend)?;
            }
            if let Some(rhs) = self.rhs_id {
                grads.accumulate(rhs, out_grad_cloned_2, backend)?;
            }
        }
        Ok(())
    }
}

/// Backward operation for max_to_shape
#[derive(Debug)]
struct MaxToShapeBackward<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    in_primal: B::Storage,
    out_primal: B::Storage,
    in_shape: Shape,
    out_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for MaxToShapeBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = match grads.get(self.out_id) {
            Some(g) => g.clone(),
            None => return Ok(()),
        };
        let grad_in = backend.max_to_shape_backward(
            &out_grad,
            &self.in_primal,
            &self.out_primal,
            &self.in_shape,
            &self.out_shape,
            self.dtype,
        )?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for MatMul
#[derive(Debug)]
struct MatMulBackward<B: Backend> {
    out_id: NodeId,
    lhs_id: Option<NodeId>,
    rhs_id: Option<NodeId>,
    lhs_primal: B::Storage,
    rhs_primal: B::Storage,
    lhs_shape: Shape,
    rhs_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for MatMulBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        let mut out_shape_vec = self.lhs_shape.to_vec();
        let lhs_rank = self.lhs_shape.rank();
        let rhs_rank = self.rhs_shape.rank();
        if lhs_rank >= 2 && rhs_rank >= 2 {
            out_shape_vec[lhs_rank - 1] = self.rhs_shape[rhs_rank - 1];
        }
        let out_shape = Shape::new(out_shape_vec);

        if let Some(lhs) = self.lhs_id {
            // grad_x = grad_z @ y^T
            let rhs_rank = self.rhs_shape.rank();
            let rhs_t = backend.transpose(
                &self.rhs_primal,
                &self.rhs_shape,
                rhs_rank - 2,
                rhs_rank - 1,
                self.dtype,
            )?;
            let mut rhs_t_shape_vec = self.rhs_shape.to_vec();
            rhs_t_shape_vec.swap(rhs_rank - 2, rhs_rank - 1);
            let rhs_t_shape = Shape::new(rhs_t_shape_vec);
            let grad_lhs =
                backend.matmul(&out_grad, &rhs_t, &out_shape, &rhs_t_shape, self.dtype)?;
            grads.accumulate(lhs, grad_lhs, backend)?;
        }
        if let Some(rhs) = self.rhs_id {
            // grad_y = x^T @ grad_z
            let lhs_rank = self.lhs_shape.rank();
            let lhs_t = backend.transpose(
                &self.lhs_primal,
                &self.lhs_shape,
                lhs_rank - 2,
                lhs_rank - 1,
                self.dtype,
            )?;
            let mut lhs_t_shape_vec = self.lhs_shape.to_vec();
            lhs_t_shape_vec.swap(lhs_rank - 2, lhs_rank - 1);
            let lhs_t_shape = Shape::new(lhs_t_shape_vec);
            let grad_rhs =
                backend.matmul(&lhs_t, &out_grad, &lhs_t_shape, &out_shape, self.dtype)?;
            grads.accumulate(rhs, grad_rhs, backend)?;
        }
        Ok(())
    }
}

/// Backward operation for Transpose
#[derive(Debug)]
struct TransposeBackward<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    dim0: usize,
    dim1: usize,
    shape: Shape,
    dtype: DType,
    _phantom: PhantomData<B>,
}

impl<B: Backend> BackwardOp<B> for TransposeBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        // grad_in = (grad_out)^T
        let mut out_shape_vec = self.shape.to_vec();
        out_shape_vec.swap(self.dim0, self.dim1);
        let out_shape = Shape::new(out_shape_vec);
        let grad_in = backend.transpose(&out_grad, &out_shape, self.dim0, self.dim1, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Subtraction
#[derive(Debug)]
struct SubBackward {
    out_id: NodeId,
    lhs_id: Option<NodeId>,
    rhs_id: Option<NodeId>,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for SubBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        if let Some(lhs) = self.lhs_id {
            // grad_x = grad_z
            grads.accumulate(lhs, out_grad.clone(), backend)?;
        }
        if let Some(rhs) = self.rhs_id {
            // grad_y = -grad_z
            // Implement -grad_z as mul_scalar(-1.0)
            let grad_rhs = backend.mul_scalar(&out_grad, -1.0, &self.shape, self.dtype)?;
            grads.accumulate(rhs, grad_rhs, backend)?;
        }
        Ok(())
    }
}

/// Backward operation for Scalar Multiplication
#[derive(Debug)]
struct MulScalarBackward {
    out_id: NodeId,
    in_id: NodeId,
    scalar: f32,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for MulScalarBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        // grad_in = grad_out * scalar
        let grad_in = backend.mul_scalar(&out_grad, self.scalar, &self.shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Element-wise Multiplication
#[derive(Debug)]
struct MulBackward<B: Backend> {
    out_id: NodeId,
    lhs_id: Option<NodeId>,
    rhs_id: Option<NodeId>,
    lhs_primal: B::Storage,
    rhs_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for MulBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        if let Some(lhs) = self.lhs_id {
            // grad_x = grad_z * y
            let grad_lhs = backend.mul(&out_grad, &self.rhs_primal, &self.shape, self.dtype)?;
            grads.accumulate(lhs, grad_lhs, backend)?;
        }
        if let Some(rhs) = self.rhs_id {
            // grad_y = grad_z * x
            let grad_rhs = backend.mul(&out_grad, &self.lhs_primal, &self.shape, self.dtype)?;
            grads.accumulate(rhs, grad_rhs, backend)?;
        }
        Ok(())
    }
}

/// Backward operation for ReLU
#[derive(Debug)]
struct ReluBackward<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    in_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for ReluBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        // grad_in = grad_out * (in_primal > 0)
        let grad_in = backend.relu_backward(&out_grad, &self.in_primal, &self.shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Sigmoid
#[derive(Debug)]
struct SigmoidBackward<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    out_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for SigmoidBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        // grad_in = grad_out * y * (1 - y)
        let grad_in =
            backend.sigmoid_backward(&out_grad, &self.out_primal, &self.shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Expand (Broadcast)
#[derive(Debug)]
struct ExpandBackward {
    out_id: NodeId,
    in_id: NodeId,
    in_shape: Shape,
    out_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for ExpandBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        // grad_in = sum_to_shape(grad_out, in_shape)
        let grad_in =
            backend.sum_to_shape(&out_grad, &self.out_shape, &self.in_shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for SumToShape
#[derive(Debug)]
struct SumToShapeBackward {
    out_id: NodeId,
    in_id: NodeId,
    in_shape: Shape,
    out_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for SumToShapeBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        // grad_in = expand(grad_out, in_shape)
        let grad_in = backend.expand(&out_grad, &self.out_shape, &self.in_shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Reshape
#[derive(Debug)]
struct ReshapeBackward {
    out_id: NodeId,
    in_id: NodeId,
    in_shape: Shape,
    out_shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for ReshapeBackward {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };
        // grad_in = reshape(grad_out, in_shape)
        let grad_in = backend.reshape(&out_grad, &self.out_shape, &self.in_shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Exp
#[derive(Debug)]
struct ExpBackwardOp<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    out_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for ExpBackwardOp<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };
        let grad_in = backend.exp_backward(&out_grad, &self.out_primal, &self.shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Log
#[derive(Debug)]
struct LogBackwardOp<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    in_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for LogBackwardOp<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };
        let grad_in = backend.log_backward(&out_grad, &self.in_primal, &self.shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Element-wise Division
#[derive(Debug)]
struct DivBackward<B: Backend> {
    out_id: NodeId,
    lhs_id: Option<NodeId>,
    rhs_id: Option<NodeId>,
    lhs_primal: B::Storage,
    rhs_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for DivBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        if let Some(lhs) = self.lhs_id {
            // grad_x = grad_z / y
            let grad_lhs =
                backend.div_backward_lhs(&out_grad, &self.rhs_primal, &self.shape, self.dtype)?;
            grads.accumulate(lhs, grad_lhs, backend)?;
        }
        if let Some(rhs) = self.rhs_id {
            // grad_y = grad_z * (-x / y^2)
            let grad_rhs = backend.div_backward_rhs(
                &out_grad,
                &self.lhs_primal,
                &self.rhs_primal,
                &self.shape,
                self.dtype,
            )?;
            grads.accumulate(rhs, grad_rhs, backend)?;
        }
        Ok(())
    }
}

/// Backward operation for Sqrt
#[derive(Debug)]
struct SqrtBackwardOp<B: Backend> {
    out_id: NodeId,
    in_id: NodeId,
    out_primal: B::Storage,
    shape: Shape,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for SqrtBackwardOp<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };
        let grad_in =
            backend.sqrt_backward(&out_grad, &self.out_primal, &self.shape, self.dtype)?;
        grads.accumulate(self.in_id, grad_in, backend)?;
        Ok(())
    }
}

/// Backward operation for Conv2d
#[derive(Debug)]
struct Conv2dBackward<B: Backend> {
    out_id: NodeId,
    in_id: Option<NodeId>,
    weight_id: Option<NodeId>,
    bias_id: Option<NodeId>,
    in_primal: B::Storage,
    weight_primal: B::Storage,
    in_shape: Shape,
    weight_shape: Shape,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    dtype: DType,
}

impl<B: Backend> BackwardOp<B> for Conv2dBackward<B> {
    fn backward(&self, grads: &mut Gradients<B>, backend: &B) -> orca_core::Result<()> {
        let out_grad = if let Some(g) = grads.get(self.out_id) {
            g.clone()
        } else {
            return Ok(());
        };

        let out_h =
            (self.in_shape[2] + 2 * self.padding - self.dilation * (self.weight_shape[2] - 1) - 1)
                / self.stride
                + 1;
        let out_w =
            (self.in_shape[3] + 2 * self.padding - self.dilation * (self.weight_shape[3] - 1) - 1)
                / self.stride
                + 1;
        let out_shape = Shape::new(vec![self.in_shape[0], self.weight_shape[0], out_h, out_w]);

        if let Some(in_id) = self.in_id {
            let grad_in = backend.conv2d_backward_input(
                &out_grad,
                &self.weight_primal,
                &self.in_shape,
                &self.weight_shape,
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                self.dtype,
            )?;
            grads.accumulate(in_id, grad_in, backend)?;
        }
        if let Some(w_id) = self.weight_id {
            let grad_w = backend.conv2d_backward_weight(
                &out_grad,
                &self.in_primal,
                &self.in_shape,
                &self.weight_shape,
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
                self.dtype,
            )?;
            grads.accumulate(w_id, grad_w, backend)?;
        }
        if let Some(b_id) = self.bias_id {
            let grad_b = backend.conv2d_backward_bias(&out_grad, &out_shape, self.dtype)?;
            grads.accumulate(b_id, grad_b, backend)?;
        }
        Ok(())
    }
}
