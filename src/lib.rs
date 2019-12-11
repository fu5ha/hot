//! A mid-level Vulkan abstraction library for the experts and the masses.
#![allow(dead_code)]
#![deny(missing_docs)]

pub use ash;

/// CommandPool abstraction.
pub mod command_pool;
pub use command_pool::*;

/// Buffers and BufferViews.
pub mod buffer;
pub use buffer::*;

/// A group of Buffers.
pub mod buffer_block;
pub use buffer_block::*;

/// Images and ImageViews.
pub mod image;
pub use image::*;

/// Resource management.
pub mod resource;

/// Utilities for working with Vulkan Formats.
pub mod format;

/// A Device wrapper, the central type which creates, owns, and manages other resources.
pub mod device;
pub use device::*;
