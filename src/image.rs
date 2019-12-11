use ash::vk;
use bitflags::bitflags;
use generational_arena as ga;

/// Initial data for an Image.
pub struct InitialImageData<'a> {
    /// The raw data.
    pub data: &'a [u8],
    /// Length of a row in pixels.
    pub row_length: usize,
    /// Height of the image in pixels.
    pub image_height: usize,
}

/// The general memory 'domain' an image should be placed in.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ImageUsageDomain {
    /// Physical which is located in the graphics device's memory(i.e. GPU). Should be used for
    /// persistent images which are used in multiple render passes.
    Physical,
    /// A transient image which exists only in GPU cache. Should be used for transient images
    /// which only exist within one physical RenderPass.
    ///
    /// For example, a depth buffer which is never read in a subsequent pass.
    Transient,
}

bitflags! {
    /// Miscellaneous options for an image.
    pub struct MiscImageFlags: u16 {
        /// Automatically generate mipmaps for the image.
        const GENERATE_MIPS = 0b0001;
    }
}

/// Info necessary to create an Image.
pub struct ImageCreateInfo {
    /// Domain to place the image into.
    pub domain: ImageUsageDomain,
    /// Width of the image in pixels.
    pub width: usize,
    /// Height of hte image in pixels.
    pub height: usize,
    /// Depth of the image in pixels.
    pub depth: usize,
    /// Number of mip levels for the image.
    pub levels: usize,
    /// The format of the image.
    pub format: vk::Format,
    /// The type of image.
    pub image_type: vk::ImageType,
    /// The image usage.
    pub usage: vk::ImageUsageFlags,
    /// The number of samples.
    pub sample_count: vk::SampleCountFlags,
    /// Vulkan creation flags for the image.
    pub create_flags: vk::ImageCreateFlags,
    /// Miscelaneous options for the image.
    pub misc_flags: MiscImageFlags,
    /// The component swizzle.
    pub swizzle: vk::ComponentMapping,
}

/// Handle to a GPU image.
pub struct Image {
    idx: ga::Index,
}

/// An owned Image and associated data. Must be manually destroyed and not be dropped.
pub struct OwnedImage {
    image: vk::Image,
    alloc: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
    create_info: ImageCreateInfo,
}

/// Get the number of possible mip levels for an image given its extent.
pub fn mip_levels_from_extent(extent: vk::Extent3D) -> u32 {
    let mut largest = extent.width.max(extent.height).max(extent.depth);

    let mut levels = 0;

    loop {
        if largest == 0 {
            break;
        }
        levels += 1;
        largest >>= 1;
    }

    levels
}

/// Extract vk::FormatFeatureFlag from given vk::ImageUsageFlags
pub fn image_usage_to_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    let mut flags = vk::FormatFeatureFlags::empty();

    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        flags |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
    }
    if usage.contains(vk::ImageUsageFlags::STORAGE) {
        flags |= vk::FormatFeatureFlags::STORAGE_IMAGE;
    }
    if usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT) {
        flags |= vk::FormatFeatureFlags::COLOR_ATTACHMENT;
    }
    if usage.contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT) {
        flags |= vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT;
    }

    flags
}

/// Get all possible vk::PipelineStageFlags from a given vk::ImageUsageFlags
pub fn image_usage_to_possible_stages(usage: vk::ImageUsageFlags) -> vk::PipelineStageFlags
{
    let mut flags = vk::PipelineStageFlags::empty();

    if !(usage & (vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)).is_empty() {
        flags |= vk::PipelineStageFlags::TRANSFER;
    }
    if !(usage & vk::ImageUsageFlags::SAMPLED).is_empty() {
        flags |= vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::VERTEX_SHADER |
                 vk::PipelineStageFlags::FRAGMENT_SHADER;
    }
    if !(usage & vk::ImageUsageFlags::STORAGE).is_empty() {
        flags |= vk::PipelineStageFlags::COMPUTE_SHADER;
    }
    if !(usage & vk::ImageUsageFlags::COLOR_ATTACHMENT).is_empty() {
        flags |= vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
    }
    if !(usage & vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT).is_empty() {
        flags |= vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
    }
    if !(usage & vk::ImageUsageFlags::INPUT_ATTACHMENT).is_empty() {
        flags |= vk::PipelineStageFlags::FRAGMENT_SHADER;
    }

    if !(usage & vk::ImageUsageFlags::TRANSIENT_ATTACHMENT).is_empty()
    {
        let mut possible = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT |
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS |
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;

        if !(usage & vk::ImageUsageFlags::INPUT_ATTACHMENT).is_empty() {
            possible |= vk::PipelineStageFlags::FRAGMENT_SHADER;
        }

        flags &= possible;
    }

    return flags;
}

/// Get all possible vk::AccessFlags from a given vk::ImageLayout
pub fn image_layout_to_possible_access(layout: vk::ImageLayout) -> vk::AccessFlags
{
    match layout
    {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL =>
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::INPUT_ATTACHMENT_READ,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL =>
            vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL =>
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::ImageLayout::DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL =>
            vk::AccessFlags::INPUT_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL =>
            vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL =>
            vk::AccessFlags::TRANSFER_WRITE,
        _ => vk::AccessFlags::empty()
    }
}

/// Get possible vk::AccessFlags from a given vk::ImageUsageFlags
pub fn image_usage_to_possible_access(usage: vk::ImageUsageFlags) -> vk::AccessFlags
{
    let mut flags = vk::AccessFlags::empty();

    if !(usage & (vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST)).is_empty() {
        flags |= vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE;
    }
    if !(usage & vk::ImageUsageFlags::SAMPLED).is_empty() {
        flags |= vk::AccessFlags::SHADER_READ;
    }
    if !(usage & vk::ImageUsageFlags::STORAGE).is_empty() {
        flags |= vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE;
    }
    if !(usage & vk::ImageUsageFlags::COLOR_ATTACHMENT).is_empty() {
        flags |= vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE;
    }
    if !(usage & vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT).is_empty() {
        flags |= vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if !(usage & vk::ImageUsageFlags::INPUT_ATTACHMENT).is_empty() {
        flags |= vk::AccessFlags::INPUT_ATTACHMENT_READ;
    }

    // Transient attachments can only be attachments, and never other resources.
    if !(usage & vk::ImageUsageFlags::TRANSIENT_ATTACHMENT).is_empty()
    {
        flags &= vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE |
                 vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE |
                 vk::AccessFlags::INPUT_ATTACHMENT_READ;
    }

    flags
}
