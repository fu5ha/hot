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

/// Extract format features from image usage.
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

/*
static inline VkPipelineStageFlags image_usage_to_possible_stages(VkImageUsageFlags usage)
{
    VkPipelineStageFlags flags = 0;

    if (usage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        flags |= VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
        flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
        flags |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        flags |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        flags |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    if (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
        flags |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

    if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
    {
        VkPipelineStageFlags possible = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;

        if (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
            possible |= VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

        flags &= possible;
    }

    return flags;
}

static inline VkAccessFlags image_layout_to_possible_access(VkImageLayout layout)
{
    switch (layout)
    {
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        return VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
        return VK_ACCESS_INPUT_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return VK_ACCESS_TRANSFER_READ_BIT;
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return VK_ACCESS_TRANSFER_WRITE_BIT;
    default:
        return ~0u;
    }
}

static inline VkAccessFlags image_usage_to_possible_access(VkImageUsageFlags usage)
{
    VkAccessFlags flags = 0;

    if (usage & (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        flags |= VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
        flags |= VK_ACCESS_SHADER_READ_BIT;
    if (usage & VK_IMAGE_USAGE_STORAGE_BIT)
        flags |= VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
    if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
        flags |= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
    if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
        flags |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    if (usage & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)
        flags |= VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;

    // Transient attachments can only be attachments, and never other resources.
    if (usage & VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT)
    {
        flags &= VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                 VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                 VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    }

    return flags;
}
*/
