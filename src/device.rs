use ash::vk;

use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};

use crate::resource::ResourcePool;
use crate::*;

struct PerFrame {
    graphics_cmd_pools: Vec<CommandPool>,
    compute_cmd_pools: Vec<CommandPool>,
    transfer_cmd_pools: Vec<CommandPool>,
}

/// The Device. Owns and manages resources, submission, etc.
pub struct Device {
    instance: vk::Instance,
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
}

impl Device {
    /// Create a Buffer from a BufferCreateInfo
    pub fn create_buffer(
        &mut self,
        create_info: BufferCreateInfo,
    ) -> Result<Buffer, vk_mem::Error> {
        let mut queue_family_indices = [0u32; 3];
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
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(create_info.size)
            .usage(create_info.usage)
            .sharing_mode(sharing_mode)
            .queue_family_indices(&queue_family_indices[0..queue_family_index_count]);

        let alloc_info = vk_mem::AllocationCreateInfo {
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
        };

        let (buffer, allocation, allocation_info) =
            self.allocator.create_buffer(&buffer_info, &alloc_info)?;

        let mapped_data = std::ptr::NonNull::new(allocation_info.get_mapped_data());

        Ok(Buffer {
            idx: self
                .resources
                .buffers
                .insert(ManuallyDrop::new(OwnedBuffer {
                    buffer,
                    allocation,
                    allocation_info,
                    create_info,
                    mapped_data,
                })),
        })
    }

    /// Get a shared reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer(&self, buffer: Buffer) -> Option<&OwnedBuffer> {
        self.resources
            .buffers
            .get(buffer.idx)
            .map(|buffer| buffer.deref())
    }

    /// Get an exclusive reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer_mut(&mut self, buffer: Buffer) -> Option<&mut OwnedBuffer> {
        self.resources
            .buffers
            .get_mut(buffer.idx)
            .map(|buffer| buffer.deref_mut())
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &ash::Device {
        &self.device
    }
}
