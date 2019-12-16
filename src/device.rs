use ash::vk;
use ash::version::InstanceV1_0;

use std::ops::{Deref};

use crate::*;
use resource::ResourcePool;

struct PerFrame {
    graphics_cmd_pools: Vec<CommandPool>,
    compute_cmd_pools: Vec<CommandPool>,
    transfer_cmd_pools: Vec<CommandPool>,

    used_vbo_blocks: Vec<BufferBlock>,
    used_ibo_blocks: Vec<BufferBlock>,
    used_ubo_blocks: Vec<BufferBlock>,
    used_staging_blocks: Vec<BufferBlock>,
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

    memory_properties: vk::PhysicalDeviceMemoryProperties,

    resources: ResourcePool,
    per_frame: Vec<PerFrame>,

    vbo_pool: BufferBlockPool,
    ibo_pool: BufferBlockPool,
    ubo_pool: BufferBlockPool,
    staging_pool: BufferBlockPool,
}

impl Device {
    /// Request a BufferBlock which will allocate buffers that may be used as vertex buffers.
    ///
    /// The BufferBlock will be automatically recycled or destroyed the next time this frame
    /// begins.
    pub fn request_vertex_block(
        &mut self,
        size: usize,
        tag: Option<Tag>
    ) -> Result<BufferBlock, vk_mem::Error> {
        self.vbo_pool.request_block(&self.allocator, size, tag)
    }

    /// Request a BufferBlock which will allocate buffers that may be used as index buffers.
    ///
    /// The BufferBlock will be automatically recycled or destroyed the next time this frame
    /// begins.
    pub fn request_index_block(
        &mut self,
        size: usize,
        tag: Option<Tag>
    ) -> Result<BufferBlock, vk_mem::Error> {
        self.ibo_pool.request_block(&self.allocator, size, tag)
    }

    /// Request a BufferBlock which will allocate buffers that may be used as uniform buffers.
    ///
    /// The BufferBlock will be automatically recycled or destroyed the next time this frame
    /// begins.
    pub fn request_uniform_block(
        &mut self,
        size: usize,
        tag: Option<Tag>
    ) -> Result<BufferBlock, vk_mem::Error> {
        self.ubo_pool.request_block(&self.allocator, size, tag)
    }

    /// Request a BufferBlock which will allocate buffers that may be used as staging buffers,
    /// i.e. buffers with TRANSFER_SRC usage whose data may be copied to a persistent GPU side
    /// buffer.
    ///
    /// The BufferBlock will be automatically recycled or destroyed the next time this frame
    /// begins.
    pub fn request_staging_block(
        &mut self,
        size: usize,
        tag: Option<Tag>
    ) -> Result<BufferBlock, vk_mem::Error> {
        self.staging_pool.request_block(&self.allocator, size, tag)
    }

    /// Get the raw `vk_mem::Allocator`.
    pub fn raw_allocator(&self) -> &vk_mem::Allocator {
        &self.allocator
    }

    /// Get the `vk::PhysicalDeviceMemoryProperties` for the physical device of this Device.
    pub fn get_physical_device_memory_properties(&self) -> vk::PhysicalDeviceMemoryProperties {
        unsafe { self.instance.get_physical_device_memory_properties(self.physical_device) }
    }

    /// Find whether a certain memory type index is visible to the cpu, i.e. able to be mapped.
    pub fn is_memory_type_host_visible(&self, type_index: u32) -> bool {
        let ty = self.memory_properties.memory_types[type_index as usize];

        ty.property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) 
    }

    /// Find whether a certain memory type index is device local, i.e. fast for on-device access.
    pub fn is_memory_type_device_local(&self, type_index: u32) -> bool {
        let ty = self.memory_properties.memory_types[type_index as usize];

        ty.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) 
    }


    /// Create a Buffer from a BufferCreateInfo and, optionally, upload some
    /// initial data to it.
    ///
    /// Depending on the type of memory that the buffer gets allocated in,
    /// the initial data will either be directly copied into the cpu-mappable
    /// buffer, or will be uploaded automatically via a staging buffer.
    ///
    /// If `initial_data` exists, `size_of::<T>` must be <= to `create_info.size`.
    pub fn create_buffer<T>(
        &mut self,
        mut create_info: BufferCreateInfo,
        tag: Option<Tag>,
        initial_data: Option<T>
    ) -> Result<Buffer, vk_mem::Error> {
        if initial_data.is_some() {
            assert!(core::mem::size_of::<T>() as vk::DeviceSize <= create_info.size);
        }

        if create_info.domain != BufferUsageDomain::Host {
            create_info.usage |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        let mut queue_family_indices = [0u32; 3];
        let buffer_info = self.raw_buffer_create_info(create_info, &mut queue_family_indices);
        let alloc_info = self.allocation_info_from_buffer_create_info(create_info);

        let (buffer, allocation, allocation_info) =
            self.allocator.create_buffer(&buffer_info, &alloc_info)?;

        let mapped_data = std::ptr::NonNull::new(allocation_info.get_mapped_data());

        let handle = Buffer {
            idx: self
                .resources
                .buffers
                .insert(OwnedBuffer::new(
                    buffer,
                    allocation,
                    allocation_info,
                    create_info,
                    mapped_data,
                    tag.clone(),
                )),
        };

        if let Some(initial_data) = initial_data {
            if let Some(mapped) = mapped_data {
                let mut mapped = mapped.cast::<T>();
                unsafe {
                    *mapped.as_mut() = initial_data;
                }
            }
        } else {
            let mut staging_info = create_info;
            staging_info.domain = BufferUsageDomain::Host;
            staging_info.usage &= !vk::BufferUsageFlags::TRANSFER_DST;
            staging_info.usage |= vk::BufferUsageFlags::TRANSFER_SRC;

            let staging_buffer = self.create_buffer(staging_info, tag.clone(), initial_data);

            // TODO
            // let cmd_buf = self.request_commad_buffer(CommandBuffer::Type::AsyncTransfer);
            // cmd_buf.copy_buffer(staging_buffer, handle);

            // self.submit_staging(cmd_buf, staging_info.usage, true);
            // self.used_staging_buffer(staging_buffer);
        }

        Ok(handle)
    }

    // pub fn used_staging_buffer(&mut self, buffer: Buffer) {

    // }

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
