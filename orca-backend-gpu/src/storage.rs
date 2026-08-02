use orca_core::{OrcaError, Result};
use orca_tensor::Storage;
use std::sync::Arc;

/// A pooled WebGPU buffer wrapper that returns its buffer to the pool when dropped.
#[derive(Debug)]
pub struct PooledBuffer {
    pub buffer: Option<wgpu::Buffer>,
    pub size: wgpu::BufferAddress,
    pub usage: wgpu::BufferUsages,
    pub pool: Option<Arc<std::sync::Mutex<crate::pool::MemoryPool>>>,
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if let Some(pool) = &self.pool {
                if let Ok(mut pool_guard) = pool.lock() {
                    pool_guard.release(buffer, self.size, self.usage);
                }
            }
        }
    }
}

/// GPU storage backed by a reference-counted wgpu Buffer.
#[derive(Clone, Debug)]
pub struct GpuStorage {
    pub inner: Arc<PooledBuffer>,
    pub num_elements: usize,
    pub element_size: usize,
}

impl GpuStorage {
    /// Creates a new GPU storage, wrapping the buffer with default STORAGE | COPY_SRC | COPY_DST usages.
    pub fn new(
        buffer: wgpu::Buffer,
        num_elements: usize,
        element_size: usize,
        pool: Arc<std::sync::Mutex<crate::pool::MemoryPool>>,
    ) -> Self {
        let size = (num_elements * element_size) as wgpu::BufferAddress;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        Self::new_with_usage(buffer, num_elements, element_size, size, usage, pool)
    }

    /// Creates a new GPU storage, specifying exact size, usage, and pool.
    pub fn new_with_usage(
        buffer: wgpu::Buffer,
        num_elements: usize,
        element_size: usize,
        size: wgpu::BufferAddress,
        usage: wgpu::BufferUsages,
        pool: Arc<std::sync::Mutex<crate::pool::MemoryPool>>,
    ) -> Self {
        let inner = PooledBuffer {
            buffer: Some(buffer),
            size,
            usage,
            pool: Some(pool),
        };
        Self {
            inner: Arc::new(inner),
            num_elements,
            element_size,
        }
    }

    /// Returns the underlying wgpu buffer.
    ///
    /// # Safety Invariant  
    /// Buffer is always `Some` after construction — `PooledBuffer::drop` is the
    /// only consumer, and it runs after all references are gone.
    pub fn buffer(&self) -> Result<&wgpu::Buffer> {
        self.inner.buffer.as_ref().ok_or_else(|| {
            OrcaError::InternalError("GpuStorage invariant violated: missing buffer".into())
        })
    }
}

impl Storage for GpuStorage {
    fn len(&self) -> usize {
        self.num_elements
    }
}
