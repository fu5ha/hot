use crate::Device;

use ash::{prelude::*, version::DeviceV1_0, vk};

struct BuffersAndIndex {
    buffers: Vec<vk::CommandBuffer>,
    idx: usize,
}

impl Default for BuffersAndIndex {
    fn default() -> Self {
        BuffersAndIndex {
            buffers: Vec::new(),
            idx: 0,
        }
    }
}

/// A CommandPool and associated command buffers.
///
/// It is assumed that command buffers created will be short lived, i.e. re-recorded every frame
/// and all reset at once.
pub struct CommandPool {
    pool: vk::CommandPool,
    buffers: BuffersAndIndex,
    secondary_buffers: BuffersAndIndex,
}

impl CommandPool {
    /// # Safety
    /// * `queue_family_index` must be the index of a queue family available on the `Device`.
    pub unsafe fn new(device: &Device, queue_family_index: u32) -> Result<CommandPool, vk::Result> {
        let create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .build();

        let pool = device.create_command_pool(&create_info, None)?;

        Ok(Self {
            pool,
            buffers: Default::default(),
            secondary_buffers: Default::default(),
        })
    }

    /// # Safety
    /// * This CommandPool must have been allocated from `device`.
    /// * All command buffers allocated from this pool must not be in use, i.e. not part of a
    /// pending GPU execution.
    pub unsafe fn reset(&mut self, device: &Device) -> VkResult<()> {
        device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())
    }

    /// # Safety
    /// * This CommandPool must have been allocated from `device`.
    /// * All command buffers allocated from this pool must not be in use, i.e. not part of a
    /// pending GPU execution.
    pub unsafe fn destroy(self, device: &Device) {
        device.destroy_command_pool(self.pool, None);
    }
}
