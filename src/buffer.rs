pub use ash::vk;
use ash::version::DeviceV1_0;

use derivative::Derivative;


use std::ptr::NonNull;
use std::sync::Arc;

use crate::{Device, Tag, resource::*};

/// The general memory 'domain' a buffer should be placed in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BufferUsageDomain {
    /// Memory which is located on the graphics device (i.e. GPU). Will be fast for
    /// access on the device, but the CPU may not be able to access this memory directly.
    Device,
    /// Memory which is located on the graphics device (i.e. GPU). Prefers to be put in
    /// memory which is directly mappable from the host, but may not be.
    DeviceDynamic,
    /// Memory which is located on the host (i.e. CPU), and which is accessible
    /// by the GPU, but said access is probably slow.
    ///
    /// This type of buffer is usually used as a 'staging' or 'upload' buffer to copy data
    /// into a faster final `Device` buffer.
    Host,
    /// Memory which is mappable on host and cached.
    ///
    /// Intended use is to suppor readback operations for data that was computed on the GPU.
    Readback,
}

/// Information needed to create a buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferCreateInfo {
    /// The memory domain into which the buffer should be placed.
    pub domain: BufferUsageDomain,
    /// Size of the buffer to create.
    pub size: vk::DeviceSize,
    /// Usage of the buffer.
    pub usage: vk::BufferUsageFlags,
}

/// An owned `vk::Buffer` and some associated information.
///
/// Will be automatically destroyed on Drop, though it must not outlife the Device it was
/// created from.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) allocation_info: vk_mem::AllocationInfo,
    pub(crate) create_info: BufferCreateInfo,
    pub(crate) mapped_data: Option<NonNull<u8>>,
    pub(crate) tag: Option<Tag>,
    #[derivative(Debug = "ignore")]
    pub(crate) device: Arc<Device>,
}

impl Buffer {
    /// Create a new owned Buffer. You probably want `Device::create_buffer` instead.
    ///
    /// # Safety
    ///
    /// `device` must be an Arc to the `Device` that this buffer was allocated from.
    pub(crate) unsafe fn new(
        device: Arc<Device>,
        buffer: vk::Buffer,
        allocation: vk_mem::Allocation,
        allocation_info: vk_mem::AllocationInfo,
        create_info: BufferCreateInfo,
        mapped_data: Option<NonNull<u8>>,
        tag: Option<Tag>,
    ) -> Self {
        Self {
            buffer,
            allocation,
            allocation_info,
            create_info,
            mapped_data,
            tag,
            device,
        }
    }

    /// The raw `vk::Buffer`
    pub fn raw(&self) -> vk::Buffer {
        self.buffer
    }

    /// The raw `vk_mem::Allocation`
    pub fn allocation(&self) -> &vk_mem::Allocation {
        &self.allocation
    }

    /// The `vk_mem::AllocationInfo` used to create this buffer.
    pub fn allocation_info(&self) -> &vk_mem::AllocationInfo {
        &self.allocation_info
    }

    /// The BufferCreateInfo used to create this buffer.
    pub fn create_info(&self) -> BufferCreateInfo {
        self.create_info
    }

    /// A NonNull pointer to the CPU mapped data of this buffer, if
    /// it exists.
    pub fn mapped_data(&mut self) -> Option<&mut NonNull<u8>> {
        self.mapped_data.as_mut()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if let Err(e) = self.device.raw_allocator().destroy_buffer(self.buffer, &self.allocation) {
            if let Some(ref tag) = self.tag {
                panic!("OwnedBuffer with tag {} errored on destruction: {:#?}", tag, e);
            } else {
                panic!("Generic (untagged) Buffer errored on destruction: {:#?}", e);
            }
        }
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

/// An owned `vk::BufferView` and some associated information.
///
/// Will automatically be destroyed on Drop, though it must not outlive the
/// Device it was allocated from.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct BufferView {
    pub(crate) buffer: BufferHandle,
    pub(crate) view: vk::BufferView,
    pub(crate) create_info: BufferViewCreateInfo,
    pub(crate) tag: Option<Tag>,
    #[derivative(Debug = "ignore")]
    pub(crate) device: Arc<Device>,
}

impl BufferView {
    /// Create a new OwnedBufferView.
    pub(crate) unsafe fn new(
        device: Arc<Device>,
        buffer: BufferHandle,
        view: vk::BufferView,
        create_info: BufferViewCreateInfo,
        tag: Option<Tag>,
    ) -> Self {
        BufferView {
            buffer,
            view,
            create_info,
            tag,
            device,
        }
    }
}

impl Drop for BufferView {
    fn drop(&mut self) {
        // safe since we must guarantee upon creation that device is the one used to allocate
        // this resource on.
        unsafe { self.device.raw_device().destroy_buffer_view(self.view, None) };
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
