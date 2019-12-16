use ash::vk;

use generational_arena as ga;

use thiserror::Error;

use crate::{OwnedBuffer, BufferCreateInfo, BufferUsageDomain, Device, NoDrop, Tag};

use std::sync::atomic::AtomicUsize;

static BUFFER_BLOCK_POOL_UUID: AtomicUsize = AtomicUsize::new(0);

/// A handle to a GPU Buffer allocated from a BufferBlock.
pub struct BufferBlockBuffer {
    block: BufferBlock,
    gpu_idx: ga::Index,
    cpu_idx: Option<ga::Index>,
}

/// An owned BufferBlock which contains the actual vk_mem::AllocatorPool(s) that back it,
/// as well as owns all the sub Buffers that have been allocated from it.
#[derive(Debug)]
pub struct OwnedBufferBlock {
    pub(crate) self_id: Option<BufferBlock>,
    pub(crate) gpu: vk_mem::AllocatorPool,
    pub(crate) cpu: Option<vk_mem::AllocatorPool>,
    pub(crate) allocated_buffers: ga::Arena<OwnedBuffer>,
    pub(crate) usage: vk::BufferUsageFlags,
    pub(crate) domain: BufferUsageDomain,
    pub(crate) size: usize,
    pub(crate) nodrop: NoDrop,
}

impl OwnedBufferBlock {
    /// Create a new OwnedBufferBlock.
    pub fn new(
        self_id: Option<BufferBlock>,
        gpu: vk_mem::AllocatorPool,
        cpu: Option<vk_mem::AllocatorPool>,
        allocated_buffers: ga::Arena<OwnedBuffer>,
        usage: vk::BufferUsageFlags,
        domain: BufferUsageDomain,
        size: usize,
        tag: Option<Tag>
    ) -> Self {
        Self {
            self_id,
            gpu,
            cpu,
            allocated_buffers,
            usage,
            domain,
            size,
            nodrop: if let Some(tag) = tag {
                NoDrop::new(tag)
            } else {
                NoDrop::from_str("Generic OwnedBufferBlock")
            },
        }
    }

    /// Get a shared reference to the GPU-side buffer referenced by a `BufferBlockBuffer` created from this `BufferBlock`.
    pub fn get_gpu_buffer(&self, buffer: BufferBlockBuffer) -> Option<&OwnedBuffer> {
        if buffer.block == self.self_id.unwrap() {
            return self.allocated_buffers.get(buffer.gpu_idx);
        }
        
        None
    }

    /// Get a mutable reference to the GPU-side buffer referenced by a `BufferBlockBuffer` created from this `BufferBlock`.
    pub fn get_gpu_buffer_mut(&mut self, buffer: BufferBlockBuffer) -> Option<&mut OwnedBuffer> {
        if buffer.block == self.self_id.unwrap() {
            return self.allocated_buffers.get_mut(buffer.gpu_idx);
        }
        
        None
    }

    /// Get a shared reference to the CPU-side buffer referenced by a `BufferBlockBuffer` created from this `BufferBlock`,
    /// if there is one.
    pub fn get_cpu_buffer(&self, buffer: BufferBlockBuffer) -> Option<&OwnedBuffer> {
        if buffer.block == self.self_id.unwrap() {
            if let Some(cpu_idx) = buffer.cpu_idx {
                return self.allocated_buffers.get(cpu_idx);
            }
        }
        
        None
    }

    /// Get a mutable reference to the CPU-side buffer referenced by a `BufferBlockBuffer` created from this `BufferBlock,
    /// if there is one.
    pub fn get_cpu_buffer_mut(&mut self, buffer: BufferBlockBuffer) -> Option<&mut OwnedBuffer> {
        if buffer.block == self.self_id.unwrap() {
            if let Some(cpu_idx) = buffer.cpu_idx {
                return self.allocated_buffers.get_mut(cpu_idx);
            }
        }
        
        None
    }

    /// Allocate a buffer from the block. The buffer is allocated in a linear fashion, making allocation very fast.
    pub fn allocate_buffer(
        &mut self,
        device: &Device,
        size: usize,
        tag: Option<Tag>,
    ) -> Result<BufferBlockBuffer, vk_mem::Error> {
        let create_info = 
            BufferCreateInfo {
                size: size as _,
                usage: self.usage,
                domain: self.domain,
            };

        let mut queue_family_indices = [0u32; 3];
        let buffer_info = device.raw_buffer_create_info(create_info, &mut queue_family_indices);

        let alloc_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            pool: Some(self.gpu.clone()),
            ..Default::default()
        };

        let (buffer, allocation, allocation_info) =
            device.raw_allocator().create_buffer(&buffer_info, &alloc_info)?;

        let mapped_data = std::ptr::NonNull::new(allocation_info.get_mapped_data());

        let gpu_idx = self
            .allocated_buffers
            .insert(OwnedBuffer::new(
                buffer,
                allocation,
                allocation_info,
                create_info,
                mapped_data,
                tag.clone(),
            ));

        
        let cpu_idx = if self.cpu.is_some() {
            let create_info = 
                BufferCreateInfo {
                    size: size as _,
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    domain: BufferUsageDomain::Host,
                };
            let buffer_info = device.raw_buffer_create_info(create_info, &mut queue_family_indices);

            let alloc_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                pool: self.cpu.clone(),
                ..Default::default()
            };

            let (buffer, allocation, allocation_info) =
                device.raw_allocator().create_buffer(&buffer_info, &alloc_info)?;

            let mapped_data = std::ptr::NonNull::new(allocation_info.get_mapped_data());

            Some(self
                .allocated_buffers
                .insert(OwnedBuffer::new(
                    buffer,
                    allocation,
                    allocation_info,
                    create_info,
                    mapped_data,
                    tag.clone()
                )))
            
        } else {
            None
        };

        Ok(BufferBlockBuffer {
            block: self.self_id.unwrap(),
            gpu_idx,
            cpu_idx
        })
    }

    /// Resets the block by destryoing all `BufferBlockBuffer`s that were allocated from the block.
    pub fn reset(&mut self, device: &Device) -> Result<(), vk_mem::Error> {
        for (_, owned_buffer) in self.allocated_buffers.drain() {
            owned_buffer.destroy(device)?;
        }
        Ok(())
    }
}

/// A block of Buffers which is intended to be basically disposable and used for only one
/// frame before being recycled. It is meant to provide ease of use for such operations,
/// and so supports CPU side upload as a first class concern.
///
/// Either a CPU-visible, GPU-owned buffer with data or a GPU-only buffer
/// and an associated CPU staging buffer for copying data into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferBlock {
    pool_uuid: usize,
    idx: ga::Index,
}

/// A pool of BufferBlocks with the same `vk::BufferUsageFlags`.
///
/// Blocks will attempt to be recycled and reused according to the description in `new`.
pub struct BufferBlockPool {
    uuid: usize,

    owned_blocks: ga::Arena<OwnedBufferBlock>,
    recycled_blocks: Vec<OwnedBufferBlock>,

    gpu_memory_type_index: u32,
    cpu_memory_type_index: Option<u32>,
    device_local: bool,
    block_size: usize,
    domain: BufferUsageDomain,
    usage: vk::BufferUsageFlags,
}

impl BufferBlockPool {
    /// Create a new BufferBlockPool.
    ///
    /// # Parameters
    ///
    /// * `block_size`: The size that each block in the pool should be allocated as. When blocks are requested from the pool,
    /// if they are requested as less than this size, they will be allocated as this size and are then able to be returned
    /// to the pool and recycled without actually allocating more memory on the device. If a block is requested with size larger
    /// than the pool's `block_size`, then a block will still be allocated, but it will need to be simply deallocated and not
    /// re-used.
    /// * `usage`: The `vk::BufferUsageFlags` that all blocks (and all and all buffers allocated from those blocks) created from
    /// this pool will have.
    /// * `requires_device_local_memory`: Whether this pool requires its memory to be on the GPU. If so, staging buffers may need
    /// to be used in order to copy data into the final GPU-side buffer.
    pub fn new(
        device: &Device,
        block_size: usize,
        usage: vk::BufferUsageFlags,
        requires_device_local_memory: bool,
    ) -> Result<Self, vk_mem::Error> {
        let uuid = BUFFER_BLOCK_POOL_UUID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let device_local = requires_device_local_memory;

        let (domain, usage) = if device_local {
            (
                BufferUsageDomain::DeviceDynamic,
                usage | vk::BufferUsageFlags::TRANSFER_DST,
            )
        } else {
            (BufferUsageDomain::Host, usage)
        };

        let create_info = BufferCreateInfo {
            domain,
            size: block_size as _,
            usage,
        };

        let gpu_memory_type_index = device.find_memory_type_index_for_buffer_info(create_info)?;

        let cpu_memory_type_index = if !device.is_memory_type_host_visible(gpu_memory_type_index) {
            let create_info = BufferCreateInfo {
                domain: BufferUsageDomain::Host,
                size: block_size as _,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
            };

            Some(device.find_memory_type_index_for_buffer_info(create_info)?)
        } else {
            None
        };

        Ok(Self {
            uuid,
            owned_blocks: ga::Arena::new(),
            recycled_blocks: Vec::new(),
            device_local,
            gpu_memory_type_index,
            cpu_memory_type_index,
            block_size,
            domain,
            usage,
        })
    }

    /// Get a shared reference to the `OwnedBufferBlock` referenced by a `BufferBlock`.
    pub fn get_block(&self, block: BufferBlock) -> Option<&OwnedBufferBlock> {
        if block.pool_uuid != self.uuid {
            return None;
        }

        self.owned_blocks.get(block.idx)
    }

    /// Get a mutable reference to the `OwnedBufferBlock` referenced by a `BufferBlock`.
    pub fn get_block_mut(&mut self, block: BufferBlock) -> Option<&mut OwnedBufferBlock> {
        if block.pool_uuid != self.uuid {
            return None;
        }

        self.owned_blocks.get_mut(block.idx)
    }

    /// Request a BufferBlock from the pool. Will attempt to reuse previously allocated recycled blocks
    /// before allocating new one(s).
    ///
    /// # Parameters
    ///
    /// * `min_size`: The minimum size that must be allocated for the block.
    pub fn request_block(
        &mut self,
        allocator: &vk_mem::Allocator,
        min_size: usize,
        tag: Option<Tag>,
    ) -> Result<BufferBlock, vk_mem::Error> {
        if min_size <= self.block_size {
            if let Some(block) = self.recycled_blocks.pop() {
                let block_idx = self.owned_blocks.insert(block);

                let block = BufferBlock {
                    pool_uuid: self.uuid,
                    idx: block_idx,
                };

                self.owned_blocks.get_mut(block_idx).unwrap().self_id = Some(block);

                return Ok(block);
            }
        }

        self.allocate_block(allocator, min_size, tag)
    }

    /// Allocate a new BufferBlock from the pool. Will not attempt to reuse a previously allocated recycled Block.
    ///
    /// # Parameters
    ///
    /// * `min_size`: The minimum size that must be allocated for the block.
    pub fn allocate_block(
        &mut self,
        allocator: &vk_mem::Allocator,
        min_size: usize,
        tag: Option<Tag>,
    ) -> Result<BufferBlock, vk_mem::Error> {
        let block_size = if min_size <= self.block_size {
            self.block_size
        } else {
            min_size
        };

        let mut pool_info = vk_mem::AllocatorPoolCreateInfo {
            memory_type_index: self.gpu_memory_type_index,
            flags: vk_mem::AllocatorPoolCreateFlags::LINEAR_ALGORITHM,
            block_size,
            min_block_count: 1,
            max_block_count: 1,
            ..Default::default()
        };

        let gpu = allocator.create_pool(&pool_info)?;

        let cpu = if let Some(cpu_memory_type_index) = self.cpu_memory_type_index {
            pool_info.memory_type_index = cpu_memory_type_index;

            Some(allocator.create_pool(&pool_info)?)
        } else {
            None
        };

        let block_idx = self.owned_blocks.insert(OwnedBufferBlock::new(
            None,
            gpu,
            cpu,
            ga::Arena::new(),
            self.usage,
            self.domain,
            block_size,
            tag,
        ));

        let block = BufferBlock {
            pool_uuid: self.uuid,
            idx: block_idx,
        };

        self.owned_blocks.get_mut(block_idx).unwrap().self_id = Some(block);

        Ok(block)
    }

    /// Attempt to recycle a block. 
    ///
    /// `block` must have been allocated from this pool, and must
    /// have the same size as the default block size as this pool. If one of these conditions is
    /// not met, the function will return an error. If a block is not successfully recycled, you must
    /// manually destroy it by calling `destroy_block` on the pool it was created from.
    pub fn recycle_block(&mut self, device: &Device, block: BufferBlock) -> Result<(), BlockRecycleError> {
        if block.pool_uuid != self.uuid {
            return Err(BlockRecycleError::WrongPool);
        }

        if let Some(owned_block) = self.owned_blocks.get(block.idx) {
            if owned_block.size != self.block_size {
                return Err(BlockRecycleError::WrongSize);
            }
        } else {
            return Err(BlockRecycleError::AlreadyFreed);
        }

        let mut owned_block = self.owned_blocks.remove(block.idx).unwrap();
        owned_block.reset(device)?;
        owned_block.self_id = None;
        self.recycled_blocks.push(owned_block);

        Ok(())
    }
}

/// An error that could occur when attempting to recycle a block.
#[derive(Error, Debug)]
pub enum BlockRecycleError {
    /// The block was not allocated from this pool.
    #[error("block was not allocated from this pool.")]
    WrongPool,
    /// The block does not have the same size as the block size of the pool.
    #[error("block does not have the same size as the block size of the pool.")]
    WrongSize,
    /// The block was already either recycled or deleted.
    #[error("block was already recycled or deleted")]
    AlreadyFreed,
    /// There was an error while destroying a contained buffer
    #[error("error destroying contained buffer")]
    DestructionError(vk_mem::Error)
}

impl From<vk_mem::Error> for BlockRecycleError {
    fn from(e: vk_mem::Error) -> Self {
        Self::DestructionError(e)
    }
}
