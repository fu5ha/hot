use ash::vk;
use bitflags::bitflags;
use generational_arena as ga;

use crate::format::format_has_depth_or_stencil_aspect;

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

/// Info necessary to create an ImageView.
///
/// Does not require an aspects bitmask because `hot` will create
/// create multiple raw `vk::ImageViews` with different aspects for you.
#[derive(Clone, Copy, Debug)]
pub struct ImageViewCreateInfo {
    /// The image being viewed.
    pub image: Image,
    /// The format to interpret the image as.
    pub format: vk::Format,
    /// The base mip level to view.
    pub base_mip_level: usize,
    /// The number of mip levels to view.
    pub mip_levels: usize,
    /// The base array layer to view.
    pub base_array_layer: usize,
    /// The number of array layers to view.
    pub array_layers: usize,
    /// The image view type.
    pub view_type: vk::ImageViewType,
    /// The component mapping to use for the view.
    pub swizzle: vk::ComponentMapping,
}

/// A handle to an image view.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct ImageView {
    idx: ga::Index,
}

/// An owned ImageView and associated data. Must be manually destroyed and not be dropped.
#[derive(Debug)]
pub struct OwnedImageView {
    view: vk::ImageView,
    render_target_views: Vec<vk::ImageView>,
    depth_view: vk::ImageView,
    stencil_view: vk::ImageView,
    unorm_view: vk::ImageView,
    srgb_view: vk::ImageView,
    create_info: ImageViewCreateInfo,
}

impl Drop for OwnedImageView {
    fn drop(&mut self) {
        panic!("OwnedImage dropped: {:?}", self);
    }
}

/// Info necessary to create an Image.
#[derive(Clone, Copy, Debug)]
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
    /// Number of image layers.
    pub layers: usize,
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
    /// The initial layout that the image should be created with.
    pub initial_layout: vk::ImageLayout,
    /// The component swizzle.
    pub swizzle: vk::ComponentMapping,
}

impl Default for ImageCreateInfo {
    fn default() -> Self {
        ImageCreateInfo {
            domain: ImageUsageDomain::Physical,
            width: 0,
            height: 0,
            depth: 0,
            levels: 1,
            layers: 1,
            format: vk::Format::UNDEFINED,
            image_type: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::empty(),
            sample_count: vk::SampleCountFlags::TYPE_1,
            create_flags: vk::ImageCreateFlags::empty(),
            misc_flags: MiscImageFlags::empty(),
            initial_layout: vk::ImageLayout::GENERAL,
            swizzle: vk::ComponentMapping::builder()
                .r(vk::ComponentSwizzle::R)
                .g(vk::ComponentSwizzle::G)
                .b(vk::ComponentSwizzle::B)
                .a(vk::ComponentSwizzle::A)
                .build(),
        }
    }
}

impl ImageCreateInfo {
    /// Make an ImageCreateInfo suitable for an immutable, 2d image
    /// using sensible defaults.
    pub fn immutable_2d_image(
        width: usize,
        height: usize,
        format: vk::Format,
        generate_mips: bool,
    ) -> Self {
        Self {
            width,
            height,
            depth: 1,
            levels: if generate_mips { 0 } else { 1 },
            usage: vk::ImageUsageFlags::SAMPLED,
            format,
            misc_flags: if generate_mips {
                MiscImageFlags::GENERATE_MIPS
            } else {
                MiscImageFlags::empty()
            },
            ..Default::default()
        }
    }

    /// Make an ImageCreateInfo suitable for an immutable, 3d image
    /// using sensible defaults.
    pub fn immutable_3d_image(
        width: usize,
        height: usize,
        depth: usize,
        format: vk::Format,
        generate_mips: bool,
    ) -> Self {
        let mut info = Self::immutable_2d_image(width, height, format, generate_mips);
        info.depth = depth;
        info.image_type = vk::ImageType::TYPE_3D;
        info
    }

    /// Make an ImageCreateInfo suitable for a render target using sensible defaults.
    pub fn render_target(width: usize, height: usize, format: vk::Format, transient: bool) -> Self {
        let mut usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        if format_has_depth_or_stencil_aspect(format) {
            usage |= vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
        } else {
            usage |= vk::ImageUsageFlags::COLOR_ATTACHMENT;
        }
        Self {
            domain: if transient {
                ImageUsageDomain::Transient
            } else {
                ImageUsageDomain::Physical
            },
            width,
            height,
            depth: 1,
            format,
            usage,
            initial_layout: if transient {
                vk::ImageLayout::UNDEFINED
            } else {
                vk::ImageLayout::GENERAL
            },
            ..Default::default()
        }
    }
}

/// Handle to a GPU image.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Image {
    idx: ga::Index,
}

/// The type of layout that this image is in. Can either be the optimal
/// layout for a given access type or the General layout which is usable for
/// any access.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ImageLayoutType {
    /// An Optimal layout, i.e. someting like `vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL`
    /// or `vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL`.
    Optimal,
    /// `vk::ImageLayout::GENERAL`
    General,
}

impl ImageLayoutType {
    /// Get the actual layout given a specific optimal layout.
    pub fn layout(self, optimal_layout: vk::ImageLayout) -> vk::ImageLayout {
        if self == ImageLayoutType::Optimal {
            optimal_layout
        } else {
            vk::ImageLayout::GENERAL
        }
    }
}

/// An owned Image and associated data. Must be manually destroyed and not be dropped.
#[derive(Debug)]
pub struct OwnedImage {
    image: vk::Image,
    alloc: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
    create_info: ImageCreateInfo,
    view: Option<ImageView>,
    layout_type: ImageLayoutType,
    stage_flags: vk::PipelineStageFlags,
    access_flags: vk::AccessFlags,
    swapchain_layout: vk::ImageLayout,
}

impl Drop for OwnedImage {
    fn drop(&mut self) {
        panic!("OwnedImage dropped: {:?}", self);
    }
}

impl OwnedImage {
    /// Get the width of this image.
    pub fn width(&self) -> usize {
        self.create_info.width
    }

    /// Get the width of this image at a given LOD (level of detail), also
    /// known as a mip level.
    pub fn width_lod(&self, lod: usize) -> usize {
        (self.create_info.width >> lod).max(1)
    }

    /// Get the height of this image.
    pub fn height(&self) -> usize {
        self.create_info.height
    }

    /// Get the height of this image at a given LOD (level of detail), also
    /// known as a mip level.
    pub fn height_lod(&self, lod: usize) -> usize {
        (self.create_info.height >> lod).max(1)
    }

    /// Get the depth of this image.
    pub fn depth(&self) -> usize {
        self.create_info.depth
    }

    /// Get the depth of this image at a given LOD (level of detail), also
    /// known as a mip level.
    pub fn depth_lod(&self, lod: usize) -> usize {
        (self.create_info.depth >> lod).max(1)
    }

    /// Get the ImageCreateInfo used to create this image.
    pub fn create_info(&self) -> ImageCreateInfo {
        self.create_info
    }

    /// Get the layout of this image given a concrete optimal layout
    pub fn layout(&self, optimal_layout: vk::ImageLayout) -> vk::ImageLayout {
        self.layout_type.layout(optimal_layout)
    }
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
