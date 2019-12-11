use ash::vk::Format;

/// Get whether a format is SRGB or not.
pub fn format_is_srgb(format: Format) -> bool {
    match format {
        Format::A8B8G8R8_SRGB_PACK32
        | Format::R8G8B8A8_SRGB
        | Format::B8G8R8A8_SRGB
        | Format::R8_SRGB
        | Format::R8G8_SRGB
        | Format::R8G8B8_SRGB
        | Format::B8G8R8_SRGB => true,
        _ => false,
    }
}

/// Get whether a format has a depth aspect.
pub fn format_has_depth_aspect(format: Format) -> bool {
    match format {
        Format::D16_UNORM
        | Format::D16_UNORM_S8_UINT
        | Format::D24_UNORM_S8_UINT
        | Format::D32_SFLOAT
        | Format::X8_D24_UNORM_PACK32
        | Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

/// Get whether a format has a stencil aspect.
pub fn format_has_stencil_aspect(format: Format) -> bool {
    match format {
        Format::D16_UNORM_S8_UINT
        | Format::D24_UNORM_S8_UINT
        | Format::D32_SFLOAT_S8_UINT
        | Format::S8_UINT => true,
        _ => false,
    }
}

/// Get whether a format has a depth or stencil aspect.
pub fn format_has_depth_or_stencil_aspect(format: Format) -> bool {
    format_has_depth_aspect(format) || format_has_stencil_aspect(format)
}

/*
static inline VkImageAspectFlags format_to_aspect_mask(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_UNDEFINED:
        return 0;

    case VK_FORMAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT;

    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT:
        return VK_IMAGE_ASPECT_STENCIL_BIT | VK_IMAGE_ASPECT_DEPTH_BIT;

    case VK_FORMAT_D16_UNORM:
    case VK_FORMAT_D32_SFLOAT:
    case VK_FORMAT_X8_D24_UNORM_PACK32:
        return VK_IMAGE_ASPECT_DEPTH_BIT;

    default:
        return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

static inline void format_align_dim(VkFormat format, uint32_t &width, uint32_t &height)
{
    uint32_t align_width, align_height;
    TextureFormatLayout::format_block_dim(format, align_width, align_height);
    width = ((width + align_width - 1) / align_width) * align_width;
    height = ((height + align_height - 1) / align_height) * align_height;
}

static inline void format_num_blocks(VkFormat format, uint32_t &width, uint32_t &height)
{
    uint32_t align_width, align_height;
    TextureFormatLayout::format_block_dim(format, align_width, align_height);
    width = (width + align_width - 1) / align_width;
    height = (height + align_height - 1) / align_height;
}

static inline VkDeviceSize format_get_layer_size(VkFormat format, unsigned width, unsigned height, unsigned depth)
{
    uint32_t blocks_x = width;
    uint32_t blocks_y = height;
    format_num_blocks(format, blocks_x, blocks_y);
    format_align_dim(format, width, height);

    VkDeviceSize size = TextureFormatLayout::format_block_size(format) * depth * blocks_x * blocks_y;
    return size;
}
*/
