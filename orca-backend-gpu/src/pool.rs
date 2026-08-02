use std::collections::HashMap;

/// A production-grade caching allocator for [`wgpu::Buffer`] objects.
///
/// This memory pool groups buffers by both their exact size in bytes and their
/// required WebGPU usages. This prevents illegal usage access errors and minimizes
/// the overhead of synchronous GPU allocations/deallocations.
#[derive(Debug)]
pub struct MemoryPool {
    /// Maps the tuple `(size, usage)` to a vector of pooled buffers.
    cache: HashMap<(wgpu::BufferAddress, wgpu::BufferUsages), Vec<wgpu::Buffer>>,
}

impl Default for MemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryPool {
    /// Creates a new, empty memory pool.
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Allocates a buffer of the requested size and usage.
    ///
    /// If a buffer with matching size and usage is available in the pool,
    /// it is returned immediately. Otherwise, a new buffer is created on the device.
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        size: wgpu::BufferAddress,
        usage: wgpu::BufferUsages,
    ) -> wgpu::Buffer {
        let key = (size, usage);
        if let Some(list) = self.cache.get_mut(&key) {
            if let Some(buffer) = list.pop() {
                return buffer;
            }
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_buffer"),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Releases a buffer back into the pool for future reuse.
    ///
    /// The buffer's original size and usage must be supplied to ensure it is
    /// cached under the correct key.
    pub fn release(
        &mut self,
        buffer: wgpu::Buffer,
        size: wgpu::BufferAddress,
        usage: wgpu::BufferUsages,
    ) {
        let key = (size, usage);
        self.cache.entry(key).or_default().push(buffer);
    }

    /// Clears the memory pool, dropping all cached buffers.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
