use generational_arena as ga;

use std::mem::ManuallyDrop;

use crate::{OwnedBuffer, OwnedBufferView};

/// Owns and manages GPU resources.
pub(crate) struct ResourcePool {
    pub(crate) buffers: ga::Arena<ManuallyDrop<OwnedBuffer>>,
    pub(crate) buffer_views: ga::Arena<ManuallyDrop<OwnedBufferView>>,
}
