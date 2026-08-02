//! Distributed execution primitives for Orca.

use orca_core::{DType, OrcaError, Result};
use orca_tensor::{Backend, Tensor};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Mutex;

fn floats_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_be_bytes())
        .collect()
}

fn receive_floats(stream: &mut TcpStream, expected_elements: usize) -> Result<Vec<f32>> {
    let mut count_bytes = [0u8; 8];
    stream.read_exact(&mut count_bytes).map_err(|error| {
        OrcaError::InternalError(format!("Failed to read element count: {error}"))
    })?;
    let element_count = u64::from_be_bytes(count_bytes) as usize;
    if element_count != expected_elements {
        return Err(OrcaError::ShapeMismatch {
            op: "all_reduce",
            expected: format!("{expected_elements} elements"),
            got: format!("{element_count} elements"),
        });
    }

    let mut buffer = vec![0u8; element_count * std::mem::size_of::<f32>()];
    stream.read_exact(&mut buffer).map_err(|error| {
        OrcaError::InternalError(format!("Failed to read tensor data: {error}"))
    })?;
    Ok(buffer
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

/// Distribute model parameters or tensors across devices/workers.
pub struct DistributedCommunicator {
    rank: usize,
    world_size: usize,
    master_addr: String,
    streams: Mutex<Option<Vec<TcpStream>>>,
}

impl DistributedCommunicator {
    pub fn new(rank: usize, world_size: usize, master_addr: &str) -> Self {
        Self {
            rank,
            world_size,
            master_addr: master_addr.to_string(),
            streams: Mutex::new(None),
        }
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    fn init_connections(&self) -> Result<()> {
        let mut guard = self
            .streams
            .lock()
            .map_err(|_| OrcaError::InternalError("Distributed stream mutex poisoned".into()))?;
        if guard.is_some() {
            return Ok(());
        }

        if self.world_size <= 1 {
            *guard = Some(vec![]);
            return Ok(());
        }

        if self.rank == 0 {
            let listener = TcpListener::bind(&self.master_addr).map_err(|e| {
                OrcaError::InternalError(format!("Failed to bind master addr: {}", e))
            })?;
            let mut accepted_streams = Vec::new();
            for _ in 1..self.world_size {
                let (stream, _) = listener.accept().map_err(|e| {
                    OrcaError::InternalError(format!("Failed to accept worker connection: {}", e))
                })?;
                stream.set_nodelay(true).map_err(|error| {
                    OrcaError::InternalError(format!("Failed to configure worker stream: {error}"))
                })?;
                accepted_streams.push(stream);
            }
            *guard = Some(accepted_streams);
        } else {
            let mut stream = None;
            for _ in 0..10 {
                if let Ok(s) = TcpStream::connect(&self.master_addr) {
                    s.set_nodelay(true).map_err(|error| {
                        OrcaError::InternalError(format!(
                            "Failed to configure master stream: {error}"
                        ))
                    })?;
                    stream = Some(s);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let s = stream.ok_or_else(|| {
                OrcaError::InternalError(format!(
                    "Worker failed to connect to master at {}",
                    self.master_addr
                ))
            })?;
            *guard = Some(vec![s]);
        }

        Ok(())
    }

    /// Performs an element-wise All-Reduce SUM across all worker nodes.
    pub fn all_reduce<B: Backend>(&self, tensor: &Tensor<B>) -> Result<Tensor<B>> {
        if self.world_size == 0 || self.rank >= self.world_size {
            return Err(OrcaError::InternalError(format!(
                "Invalid distributed rank {} for world size {}",
                self.rank, self.world_size
            )));
        }
        if self.world_size <= 1 {
            return Ok(tensor.clone());
        }

        self.init_connections()?;

        let local_data = if tensor.dtype() != DType::F32 {
            let f32_tensor = tensor.to_dtype(DType::F32)?;
            f32_tensor.to_f32_vec()?
        } else {
            tensor.to_f32_vec()?
        };

        let num_elements = local_data.len();
        let num_bytes = num_elements * 4;

        let mut guard = self
            .streams
            .lock()
            .map_err(|_| OrcaError::InternalError("Distributed stream mutex poisoned".into()))?;
        let streams = guard.as_mut().ok_or_else(|| {
            OrcaError::InternalError("Distributed streams are not initialized".into())
        })?;

        if self.rank == 0 {
            let mut global_sum = local_data.clone();

            for stream in streams.iter_mut() {
                let worker_data = receive_floats(stream, num_elements)?;

                for i in 0..num_elements {
                    global_sum[i] += worker_data[i];
                }
            }

            let sum_bytes = floats_to_bytes(&global_sum);

            for stream in streams.iter_mut() {
                stream.write_all(&sum_bytes).map_err(|e| {
                    OrcaError::InternalError(format!(
                        "Master failed to broadcast back to worker: {}",
                        e
                    ))
                })?;
            }

            let sum_tensor = Tensor::from_f32_slice(
                tensor.backend().clone(),
                &global_sum,
                tensor.shape().clone(),
            )?;
            if tensor.dtype() != DType::F32 {
                sum_tensor.to_dtype(tensor.dtype())
            } else {
                Ok(sum_tensor)
            }
        } else {
            let worker_stream = &mut streams[0];

            let local_bytes = floats_to_bytes(&local_data);
            worker_stream
                .write_all(&(num_elements as u64).to_be_bytes())
                .map_err(|e| {
                    OrcaError::InternalError(format!("Worker failed to write to master: {}", e))
                })?;
            worker_stream.write_all(&local_bytes).map_err(|e| {
                OrcaError::InternalError(format!("Worker failed to write tensor data: {}", e))
            })?;

            let mut buffer = vec![0u8; num_bytes];
            worker_stream.read_exact(&mut buffer).map_err(|e| {
                OrcaError::InternalError(format!("Worker failed to read from master: {}", e))
            })?;

            let summed_data: Vec<f32> = buffer
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let sum_tensor = Tensor::from_f32_slice(
                tensor.backend().clone(),
                &summed_data,
                tensor.shape().clone(),
            )?;
            if tensor.dtype() != DType::F32 {
                sum_tensor.to_dtype(tensor.dtype())
            } else {
                Ok(sum_tensor)
            }
        }
    }
}
