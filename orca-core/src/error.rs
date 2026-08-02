use thiserror::Error;

/// The unified error type for the Orca framework.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum OrcaError {
    #[error("Shape mismatch during {op}: expected {expected}, got {got}")]
    ShapeMismatch {
        op: &'static str,
        expected: String,
        got: String,
    },

    #[error("Device mismatch: tensors must be on the same device. Got {0} and {1}")]
    DeviceMismatch(crate::Device, crate::Device),

    #[error("Unsupported dtype for {op}: {dtype}")]
    UnsupportedDType {
        op: &'static str,
        dtype: crate::DType,
    },

    #[error("Type mismatch during {op}: expected {expected:?}, got {got:?}")]
    DTypeMismatch {
        op: &'static str,
        expected: crate::DType,
        got: crate::DType,
    },

    #[error("Internal error: {0}")]
    InternalError(String),
}

/// A specialized Result type for Orca operations.
pub type Result<T> = std::result::Result<T, OrcaError>;
