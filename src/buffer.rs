pub use ash::vk;

use generational_arena as ga;

/// The general memory 'domain' a buffer should be placed in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferUsageDomain {
    /// Memory which is located on the graphics device (i.e. GPU). Will be fast for
    /// access on the device, but the CPU cannot access this memory.
    GpuOnly,
    /// Memory which is located on the host (i.e. CPU), and which is accessible
    /// by the GPU, but said access is probably slow.
    ///
    /// This type of buffer is
    /// usually used as a 'staging' or 'upload' buffer to copy data into a faster
    /// final `GpuOnly` buffer.
    Upload,
    /// Memory which is mappable on the CPU and is also preferrably fast to access
    /// on the GPU.
    ///
    /// Intended for use with data that is relatively small and changes frequently on the CPU,
    /// such as uniform, vertex, or texture data which is updated on every frame or draw call.
    ///
    /// This domain will be effective mostly for integrated and mobile GPUs,
    /// and some AMD GPUs which present some memory types that are CPU mappable but are also
    /// fast to access on the GPU, while on other platforms it will fall back to regular `Upload`
    /// memory.
    Dynamic,
    /// Memory which is mappable on host and cached.
    ///
    /// Intended use is to suppor readback operations for data that was computed on the GPU.
    Readback,
}

/// Information needed to create a Buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferCreateInfo {
    /// The memory domain into which the buffer should be placed.
    pub domain: BufferUsageDomain,
    /// Size of the buffer to create.
    pub size: vk::DeviceSize,
    /// Usage of the buffer.
    pub usage: vk::BufferUsageFlags,
}

/// Handle to a GPU Buffer.
#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    idx: ga::Index,
}

/// An owned `vk::Buffer` and some associated information.
#[derive(Debug)]
pub struct OwnedBuffer {
    buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
    create_info: BufferCreateInfo,
}

impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        panic!("OwnedBuffer dropped: {:?}", self);
    }
}

/// Information needed to create a BufferView
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferViewCreateInfo {
    /// The format the interpret the data being viewed as.
    pub format: vk::Format,
    /// The offset into the buffer data where the view should begin, in bytes.
    pub offset: vk::DeviceSize,
    /// The amount of data to be viewed, in bytes.
    pub range: vk::DeviceSize,
}

/// Handle to a GPU BufferView.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferView {
    idx: ga::Index,
}

/// An owned `vk::BufferView` and some associated information.
#[derive(Debug)]
pub struct OwnedBufferView {
    buffer: Buffer,
    view: vk::BufferView,
    create_info: BufferViewCreateInfo,
}

impl Drop for OwnedBufferView {
    fn drop(&mut self) {
        panic!("OwnedBufferView dropped: {:?}", self);
    }
}

/// Get all possible `vk::PipelineStageFlags` given a set of `vk::BufferUsageFlags`.
pub fn possible_stages_from_usage(usage: vk::BufferUsageFlags) -> vk::PipelineStageFlags {
    let mut flags = vk::PipelineStageFlags::empty();

    if usage.contains(vk::BufferUsageFlags::TRANSFER_SRC)
        || usage.contains(vk::BufferUsageFlags::TRANSFER_DST)
    {
        flags |= vk::PipelineStageFlags::TRANSFER;
    }
    if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER)
        || usage.contains(vk::BufferUsageFlags::INDEX_BUFFER)
    {
        flags |= vk::PipelineStageFlags::VERTEX_INPUT;
    }
    if usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
        flags |= vk::PipelineStageFlags::DRAW_INDIRECT;
    }
    if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER)
        || usage.contains(vk::BufferUsageFlags::UNIFORM_TEXEL_BUFFER)
        || usage.contains(vk::BufferUsageFlags::STORAGE_TEXEL_BUFFER)
    {
        flags |= vk::PipelineStageFlags::COMPUTE_SHADER
            | vk::PipelineStageFlags::VERTEX_SHADER
            | vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        flags |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }

    flags
}

/// Get all possible `vk::AccessFlags` given a set of `vk::BufferUsageFlags`.
pub fn possible_accesses_from_usage(usage: vk::BufferUsageFlags) -> vk::AccessFlags {
    let mut access = vk::AccessFlags::empty();

    if usage.contains(vk::BufferUsageFlags::TRANSFER_SRC) {
        access |= vk::AccessFlags::TRANSFER_READ;
    }
    if usage.contains(vk::BufferUsageFlags::TRANSFER_DST) {
        access |= vk::AccessFlags::TRANSFER_WRITE;
    }
    if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
        access |= vk::AccessFlags::VERTEX_ATTRIBUTE_READ;
    }
    if usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
        access |= vk::AccessFlags::INDEX_READ;
    }
    if usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
        access |= vk::AccessFlags::INDIRECT_COMMAND_READ;
    }
    if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
        access |= vk::AccessFlags::UNIFORM_READ;
    }
    if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        access |= vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    }

    access
}
