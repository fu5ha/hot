use ash::vk;

use crate::{Buffer, BufferCreateInfo, BufferUsageDomain, Device};

/// Either a CPU-visible, GPU-owned buffer with data or a GPU-only buffer
/// and an associated CPU staging buffer for copying data into it.
pub struct BufferBlock {
    gpu: Buffer,
    cpu: Option<Buffer>,
    size: vk::DeviceSize,
}

/// A pool of BufferBlocks which are allocated on a linear allocator, and freed
/// in a FIFO order.
pub struct LinearBufferRing {
    allocator_pool: vk_mem::AllocatorPool,
    device_local: bool,
    usage: vk::BufferUsageFlags,
}

impl LinearBufferRing {
    /// Allocate a new BufferBlock from the pool.
    pub fn allocate_block(
        &mut self,
        device: &mut Device,
        size: vk::DeviceSize,
    ) -> Result<BufferBlock, vk_mem::Error> {
        let (domain, usage) = if self.device_local {
            (
                BufferUsageDomain::DeviceDynamic,
                self.usage | vk::BufferUsageFlags::TRANSFER_DST,
            )
        } else {
            (BufferUsageDomain::Host, self.usage)
        };
        let create_info = BufferCreateInfo {
            domain,
            size,
            usage,
        };
        let gpu_buffer_handle = device.create_buffer(create_info)?;
        let gpu_buffer = device.get_buffer_mut(gpu_buffer_handle).unwrap();

        let cpu_buffer_handle = if gpu_buffer.mapped_data().is_none() {
            let create_info = BufferCreateInfo {
                domain: BufferUsageDomain::Host,
                size,
                usage: self.usage | vk::BufferUsageFlags::TRANSFER_SRC,
            };
            Some(device.create_buffer(create_info)?)
        } else {
            None
        };

        Ok(BufferBlock {
            gpu: gpu_buffer_handle,
            cpu: cpu_buffer_handle,
            size,
        })
    }
}
