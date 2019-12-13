use generational_arena as ga;

use crate::{OwnedBuffer, OwnedBufferView};

/// Owns and manages GPU resources.
pub(crate) struct ResourcePool {
    pub(crate) buffers: ga::Arena<OwnedBuffer>,
    pub(crate) buffer_views: ga::Arena<OwnedBufferView>,
}
