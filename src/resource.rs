use generational_arena as ga;

use std::mem::ManuallyDrop;

use crate::{OwnedBuffer, OwnedBufferView};

/// Owns and manages GPU resources.
pub struct ResourceManager {
    buffers: ga::Arena<ManuallyDrop<OwnedBuffer>>,
    buffer_views: ga::Arena<ManuallyDrop<OwnedBufferView>>,
}
