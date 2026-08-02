use orca_core::{OrcaError, Result};
use orca_tensor::Storage;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

/// A robust, SIMD-aligned (64-byte) raw memory buffer for CPU tensors.
/// Provides safe slice casting for various numeric types without alignment issues (Undefined Behavior).
struct AlignedBuffer {
    ptr: *mut u8,
    layout: Layout,
    capacity_bytes: usize,
}

// SAFETY: The buffer owns a single allocation and only exposes raw slices that
// stay within that allocation. No aliasing is introduced beyond what `Arc`
// already permits.
unsafe impl Send for AlignedBuffer {}
// SAFETY: Shared references only read the owned allocation; mutable access is
// mediated through `Arc::make_mut`, which preserves uniqueness before writes.
unsafe impl Sync for AlignedBuffer {}

impl AlignedBuffer {
    fn new(layout: Layout, capacity_bytes: usize) -> Self {
        let align = 64;
        debug_assert_eq!(layout.align(), align);
        // SAFETY: `layout` is validated by the caller and uses a fixed
        // power-of-two alignment that matches the allocation strategy.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self {
            ptr,
            layout,
            capacity_bytes,
        }
    }

    fn as_bytes(&self) -> &[u8] {
        if self.capacity_bytes == 0 {
            return &[];
        }
        // SAFETY: `ptr` points to an allocation of at least `capacity_bytes`
        // bytes for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.capacity_bytes) }
    }

    fn as_mut_bytes(&mut self) -> &mut [u8] {
        if self.capacity_bytes == 0 {
            return &mut [];
        }
        // SAFETY: `ptr` points to an allocation of at least `capacity_bytes`
        // bytes and `&mut self` guarantees unique access here.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.capacity_bytes) }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.capacity_bytes > 0 && !self.ptr.is_null() {
            // SAFETY: `ptr` was allocated with this exact `layout` and has not
            // been freed yet.
            unsafe { dealloc(self.ptr, self.layout) };
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        let mut new_buf = AlignedBuffer::new(self.layout, self.capacity_bytes);
        new_buf.as_mut_bytes().copy_from_slice(self.as_bytes());
        new_buf
    }
}

impl Debug for AlignedBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("AlignedBuffer")
            .field("capacity_bytes", &self.capacity_bytes)
            .finish()
    }
}

/// CPU storage backed by a reference-counted, aligned byte array.
#[derive(Clone, Debug)]
pub struct CpuByteStorage {
    data: Arc<AlignedBuffer>,
    num_elements: usize,
}

impl CpuByteStorage {
    pub fn new(size_in_bytes: usize, num_elements: usize, element_size: usize) -> Result<Self> {
        let expected_size = num_elements.checked_mul(element_size).ok_or_else(|| {
            OrcaError::InternalError("CpuStorage size overflow while allocating".into())
        })?;

        if size_in_bytes != expected_size {
            return Err(OrcaError::InternalError(format!(
                "CpuStorage size mismatch: expected {} bytes, got {}",
                expected_size, size_in_bytes
            )));
        }

        let layout = Layout::from_size_align(size_in_bytes.max(1), 64)
            .map_err(|_| OrcaError::InternalError("Invalid layout for CpuStorage".into()))?;
        Ok(Self {
            data: Arc::new(AlignedBuffer::new(layout, size_in_bytes)),
            num_elements,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    pub fn as_mut_bytes(&mut self) -> &mut [u8] {
        Arc::make_mut(&mut self.data).as_mut_bytes()
    }

    pub fn as_slice<T>(&self) -> &[T] {
        if self.num_elements == 0 {
            return &[];
        }
        assert!(
            (self.data.ptr as usize).is_multiple_of(std::mem::align_of::<T>()),
            "Alignment mismatch for tensor type"
        );
        // SAFETY: Alignment is verified above and `num_elements` is bounded by
        // the allocated capacity for the buffer.
        unsafe { std::slice::from_raw_parts(self.data.ptr as *const T, self.num_elements) }
    }

    pub fn as_mut_slice<T>(&mut self) -> &mut [T] {
        if self.num_elements == 0 {
            return &mut [];
        }
        let buf = Arc::make_mut(&mut self.data);
        assert!(
            (buf.ptr as usize).is_multiple_of(std::mem::align_of::<T>()),
            "Alignment mismatch for tensor type"
        );
        // SAFETY: Alignment is verified above and `&mut self` ensures unique
        // access to the underlying allocation.
        unsafe { std::slice::from_raw_parts_mut(buf.ptr as *mut T, self.num_elements) }
    }
}

impl Storage for CpuByteStorage {
    fn len(&self) -> usize {
        self.num_elements
    }
}
