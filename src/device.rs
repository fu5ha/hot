use ash::vk;
use ash::version::InstanceV1_0;

use std::ops::{Deref};

use crate::*;
use resource::ResourcePool;
use buffer_block::{VertexBlock, IndexBlock, UniformBlock};

struct PerFrame {
    graphics_cmd_pools: Vec<CommandPool>,
    compute_cmd_pools: Vec<CommandPool>,
    transfer_cmd_pools: Vec<CommandPool>,

    used_vbo_blocks: Vec<VertexBlock>,
    used_ibo_blocks: Vec<IndexBlock>,
    used_ubo_blocks: Vec<UniformBlock>,
    used_staging_blocks: Vec<StagingBlock>,
}

/// The Device. Owns and manages resources, submission, etc.
pub struct Device {
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    allocator: vk_mem::Allocator,

    graphics_queue: vk::Queue,
    graphics_queue_family_index: u32,
    compute_queue: vk::Queue,
    compute_queue_family_index: u32,
    transfer_queue: vk::Queue,
    transfer_queue_family_index: u32,
    multiple_queue_families: bool,

    resources: ResourcePool,
    per_frame: Vec<PerFrame>,

    vbo_pool: BufferBlockPool,
    ibo_pool: BufferBlockPool,
    ubo_pool: BufferBlockPool,
    staging_pool: BufferBlockPool,
}

impl Device {
    /// Get the raw `vk_mem::Allocator`.
    pub fn raw_allocator(&self) -> &vk_mem::Allocator {
        &self.allocator
    }

    /// Get the `vk::PhysicalDeviceMemoryProperties` for the physical device of this Device.
    pub fn get_physical_device_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) }
    }

    /// Create a Buffer from a BufferCreateInfo
    pub fn create_buffer(
        &mut self,
        create_info: BufferCreateInfo,
        tag: Option<Tag>,
    ) -> Result<Buffer, vk_mem::Error> {
        let mut queue_family_indices = [0u32; 3];
        let buffer_info = self.raw_buffer_create_info(create_info, &mut queue_family_indices);
        let alloc_info = self.allocation_info_from_buffer_create_info(create_info);

        let (buffer, allocation, allocation_info) =
            self.allocator.create_buffer(&buffer_info, &alloc_info)?;

        let mapped_data = std::ptr::NonNull::new(allocation_info.get_mapped_data());

        Ok(Buffer {
            idx: self
                .resources
                .buffers
                .insert(OwnedBuffer::new(
                    buffer,
                    allocation,
                    allocation_info,
                    create_info,
                    mapped_data,
                    tag,
                )),
        })
    }

    /// A helper function to find a usable memory type index given an example BufferInfo for
    /// a buffer to be allocated.
    pub fn find_memory_type_index_for_buffer_info(
        &self,
        create_info: BufferCreateInfo,
    ) -> Result<u32, vk_mem::Error> {
        let mut queue_family_indices = [0u32; 3];
        let buffer_info = self.raw_buffer_create_info(create_info, &mut queue_family_indices);
        let alloc_info = self.allocation_info_from_buffer_create_info(create_info);

        self.allocator.find_memory_type_index_for_buffer_info(&buffer_info, &alloc_info)
    }

    /// Create a Buffer from a BufferCreateInfo into a specific pool
    pub fn create_buffer_in(
        &mut self,
        create_info: BufferCreateInfo,
        pool: vk_mem::AllocatorPool,
        tag: Option<Tag>,
    ) -> Result<Buffer, vk_mem::Error> {
        let mut queue_family_indices = [0u32; 3];
        let buffer_info = self.raw_buffer_create_info(create_info, &mut queue_family_indices);

        let alloc_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            pool: Some(pool),
            ..Default::default()
        };

        let (buffer, allocation, allocation_info) =
            self.allocator.create_buffer(&buffer_info, &alloc_info)?;

        let mapped_data = std::ptr::NonNull::new(allocation_info.get_mapped_data());

        Ok(Buffer {
            idx: self
                .resources
                .buffers
                .insert(OwnedBuffer::new(
                    buffer,
                    allocation,
                    allocation_info,
                    create_info,
                    mapped_data,
                    tag
                )),
        })
    }

    /// Get a shared reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer(&self, buffer: Buffer) -> Option<&OwnedBuffer> {
        self.resources
            .buffers
            .get(buffer.idx)
    }

    /// Get an exclusive reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer_mut(&mut self, buffer: Buffer) -> Option<&mut OwnedBuffer> {
        self.resources
            .buffers
            .get_mut(buffer.idx)
    }

    /// Create the corresponding `vk_mem::AllocationCreateInfo` for a specified `BufferCreateInfo`
    pub fn allocation_info_from_buffer_create_info(
        &self,
        create_info: BufferCreateInfo
    ) -> vk_mem::AllocationCreateInfo {
        vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::Unknown,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: match create_info.domain {
                BufferUsageDomain::Device => vk::MemoryPropertyFlags::DEVICE_LOCAL,
                BufferUsageDomain::DeviceDynamic => vk::MemoryPropertyFlags::DEVICE_LOCAL,
                BufferUsageDomain::Host => vk::MemoryPropertyFlags::HOST_VISIBLE,
                BufferUsageDomain::Readback => {
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED
                }
            },
            preferred_flags: match create_info.domain {
                BufferUsageDomain::DeviceDynamic => vk::MemoryPropertyFlags::HOST_VISIBLE,
                _ => vk::MemoryPropertyFlags::empty(),
            },
            ..Default::default()
        }
    }

    /// Create the corresonding `vk::BufferCreateInfoBuilder` for a given `BufferCreateInfo`
    ///
    /// # Parameters
    ///
    /// * `queue_family_indices` this array will be filled with the needed queue family indices
    /// and must live at least as long as the returned `vk::BufferCreateInfoBuilder`
    pub fn raw_buffer_create_info<'a>(
        &self,
        create_info: BufferCreateInfo,
        queue_family_indices: &'a mut [u32; 3],
    ) -> vk::BufferCreateInfoBuilder<'a> {
        let (sharing_mode, queue_family_index_count) = if self.multiple_queue_families {
            let mut count = 1;
            queue_family_indices[0] = self.graphics_queue_family_index;
            if self.graphics_queue_family_index != self.compute_queue_family_index {
                queue_family_indices[count] = self.compute_queue_family_index;
                count += 1;
            }
            if self.graphics_queue_family_index != self.transfer_queue_family_index
                && self.compute_queue_family_index != self.transfer_queue_family_index
            {
                queue_family_indices[count] = self.transfer_queue_family_index;
                count += 1;
            }
            (vk::SharingMode::CONCURRENT, count)
        } else {
            (vk::SharingMode::EXCLUSIVE, 0)
        };

        vk::BufferCreateInfo::builder()
            .size(create_info.size)
            .usage(create_info.usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(&queue_family_indices[0..queue_family_index_count])
    }
}


impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.device
    }
}
