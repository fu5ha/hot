use generational_arena as ga;

use crate::*;

/// A set of persistent GPU resources.
pub struct ResourceSet {
    pub(crate) buffers: ga::Arena<Buffer>,
    pub(crate) buffer_views: ga::Arena<BufferView>,
    pub(crate) images: ga::Arena<Image>,
    pub(crate) image_views: ga::Arena<ImageView>,
}

impl ResourceSet {
    /// Get a shared reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer(&self, buffer: BufferHandle) -> Option<&Buffer> {
        self.buffers.get(buffer.idx)
    }

    /// Get an exclusive reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer_mut(&mut self, buffer: BufferHandle) -> Option<&mut Buffer> {
        self.buffers.get_mut(buffer.idx)
    }

    /// Get a shared reference to the owned buffer view behind a given handle, if
    /// it still exists.
    pub fn get_buffer_view(&self, buffer_view: BufferViewHandle) -> Option<&BufferView> {
        self.buffer_views.get(buffer_view.idx)
    }

    /// Get an exclusive reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_buffer_view_mut(&mut self, buffer_view: BufferViewHandle) -> Option<&mut BufferView> {
        self.buffer_views.get_mut(buffer_view.idx)
    }

    /// Get a shared reference to the owned image behind a given handle, if
    /// it still exists.
    pub fn get_image(&self, image: ImageHandle) -> Option<&Image> {
        self.images.get(image.idx)
    }

    /// Get an exclusive reference to the owned buffer behind a given handle, if
    /// it still exists.
    pub fn get_image_mut(&mut self, image: ImageHandle) -> Option<&mut Image> {
        self.images.get_mut(image.idx)
    }
}

/// Handle to a GPU buffer.
#[derive(Clone, Copy, Debug)]
pub struct BufferHandle {
    pub(crate) idx: ga::Index,
}

impl BufferHandle {
    pub(crate) fn new(idx: ga::Index) -> Self {
       BufferHandle { idx }
    }
}

/// Handle to a GPU buffer view.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct BufferViewHandle {
    pub(crate) idx: ga::Index,
}

impl BufferViewHandle {
    pub(crate) fn new(idx: ga::Index) -> Self {
        BufferViewHandle { idx }
    }
}

/// Handle to a GPU image.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ImageHandle {
    pub(crate) idx: ga::Index,
}

impl ImageHandle {
    pub(crate) fn new(idx: ga::Index) -> Self {
        ImageHandle { idx }
    }
}

/// A set of BufferBlockPools, for different usages.
pub struct BufferBlockSet {
    pub(crate) vbo_pool: BufferBlockPool,
    pub(crate) ibo_pool: BufferBlockPool,
    pub(crate) ubo_pool: BufferBlockPool,
    pub(crate) staging_pool: BufferBlockPool,
}

impl BufferBlockSet {
    /// Get a reference to a vertex buffer block, if it exists.
    pub fn get_vertex_block(&self, block: BufferBlockHandle) -> Option<&BufferBlock> {
        self.vbo_pool.get_block(block)
    }

    /// Get a reference to a vertex buffer block, if it exists.
    pub fn get_vertex_block_mut(&mut self, block: BufferBlockHandle) -> Option<&mut BufferBlock> {
        self.vbo_pool.get_block_mut(block)
    }

    /// Get a reference to a uniform buffer block, if it exists.
    pub fn get_uniform_block(&self, block: BufferBlockHandle) -> Option<&BufferBlock> {
        self.ubo_pool.get_block(block)
    }

    /// Get a reference to a uniform buffer block, if it exists.
    pub fn get_uniform_block_mut(&mut self, block: BufferBlockHandle) -> Option<&mut BufferBlock> {
        self.ubo_pool.get_block_mut(block)
    }

    /// Get a reference to a index buffer block, if it exists.
    pub fn get_index_block(&self, block: BufferBlockHandle) -> Option<&BufferBlock> {
        self.ibo_pool.get_block(block)
    }

    /// Get a reference to a index buffer block, if it exists.
    pub fn get_index_block_mut(&mut self, block: BufferBlockHandle) -> Option<&mut BufferBlock> {
        self.ibo_pool.get_block_mut(block)
    }

    /// Get a reference to a staging buffer block, if it exists.
    pub fn get_staging_block(&self, block: BufferBlockHandle) -> Option<&BufferBlock> {
        self.staging_pool.get_block(block)
    }

    /// Get a reference to a staging buffer block, if it exists.
    pub fn get_staging_block_mut(&mut self, block: BufferBlockHandle) -> Option<&mut BufferBlock> {
        self.staging_pool.get_block_mut(block)
    }
}

// /// A struct used when syncing a ThreadedResourcePools into a main ResourcePool
// pub struct ResourceHandles {
//     pub buffers: Vec<Buffer>,
//     pub buffer_views: Vec<BufferView>,
//     pub images: Vec<Image>,
//     pub image_views: Vec<ImageView>,
// }

// impl ResourcePool {
//     pub(crate) fn sync_threaded_pools<I>(&mut self, pools: I) -> Vec<ResourceHandles>
//         where I: IntoIterator<Item = ThreadedResourcePool>
//     {
//         pools.into_iter().map(|mut pool| {
//             let buffers = pool.buffers.drain(..).map(|buffer| {
//                 Buffer::new(self.buffers.insert(buffer))
//             }).collect::<Vec<_>>();

//             let buffer_views = pool.buffer_views.drain(..).map(|buffer_view| {
//                 BufferView::new(self.buffer_views.insert(buffer_view))
//             }).collect::<Vec<_>>();

//             let images = pool.images.drain(..).map(|image| {
//                 Image::new(self.images.insert(image))
//             }).collect::<Vec<_>>();

//             let image_views = pool.image_views.drain(..).map(|image_view| {
//                 ImageView::new(self.image_views.insert(image_view))
//             }).collect::<Vec<_>>();

//             ResourceHandles {
//                 buffers,
//                 buffer_views,
//                 images,
//                 image_views
//             }
//         })
//         .collect::<Vec<_>>()
//     }
// }

// pub struct ThreadedResourcePool {
//     pub(crate) buffers: Vec<OwnedBuffer>,
//     pub(crate) buffer_views: Vec<OwnedBufferView>,
//     pub(crate) images: Vec<OwnedImage>,
//     pub(crate) image_views: Vec<OwnedImageView>,
// }
