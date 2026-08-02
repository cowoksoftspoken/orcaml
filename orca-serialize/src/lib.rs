use orca_core::{DType, OrcaError, Result, Shape};
use orca_tensor::{Backend, Tensor};
use safetensors::tensor::{Dtype as SafeDtype, SafeTensors, TensorView};
use std::collections::HashMap;

pub fn save_tensors<B: Backend>(tensors: &HashMap<String, Tensor<B>>, path: &str) -> Result<()> {
    let mut buffers: Vec<Vec<u8>> = Vec::new();

    for t in tensors.values() {
        let byte_data = t.to_bytes()?;
        buffers.push(byte_data);
    }

    let mut views = Vec::new();
    for (i, (name, t)) in tensors.iter().enumerate() {
        let shape: Vec<usize> = t.shape().to_vec();

        let safe_dtype = match t.dtype() {
            DType::F32 => SafeDtype::F32,
            DType::F64 => SafeDtype::F64,
            DType::F16 => SafeDtype::F16,
            DType::BF16 => SafeDtype::BF16,
            DType::I32 => SafeDtype::I32,
            DType::I64 => SafeDtype::I64,
            DType::U8 => SafeDtype::U8,
            DType::Bool => SafeDtype::BOOL,
        };

        let view = TensorView::new(safe_dtype, shape, &buffers[i])
            .map_err(|e| OrcaError::InternalError(format!("Safetensors error: {:?}", e)))?;
        views.push((name.clone(), view));
    }

    safetensors::serialize_to_file(views.into_iter(), &None, path.as_ref())
        .map_err(|e| OrcaError::InternalError(format!("Failed to save safetensors: {:?}", e)))?;

    Ok(())
}

pub fn load_tensors<B: Backend>(backend: B, path: &str) -> Result<HashMap<String, Tensor<B>>> {
    let buffer = std::fs::read(path)
        .map_err(|e| OrcaError::InternalError(format!("Failed to read safetensors file: {}", e)))?;

    let safe = SafeTensors::deserialize(&buffer)
        .map_err(|e| OrcaError::InternalError(format!("Safetensors deserialize error: {:?}", e)))?;

    let mut tensors = HashMap::new();
    for name in safe.names() {
        let view = safe.tensor(name).map_err(|e| {
            OrcaError::InternalError(format!(
                "Failed to retrieve tensor view for {}: {:?}",
                name, e
            ))
        })?;

        let dtype = match view.dtype() {
            SafeDtype::F32 => DType::F32,
            SafeDtype::F64 => DType::F64,
            SafeDtype::F16 => DType::F16,
            SafeDtype::BF16 => DType::BF16,
            SafeDtype::I32 => DType::I32,
            SafeDtype::I64 => DType::I64,
            SafeDtype::U8 => DType::U8,
            SafeDtype::BOOL => DType::Bool,
            _ => {
                return Err(OrcaError::InternalError(
                    "Unsupported safetensor dtype".into(),
                ))
            }
        };

        let data = view.data();
        let shape = Shape::new(view.shape().to_vec());
        let tensor = Tensor::from_bytes(backend.clone(), data, shape, dtype)?;
        tensors.insert(name.clone(), tensor);
    }

    Ok(tensors)
}
