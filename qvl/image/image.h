#pragma once

#include <contracts>
#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

#include <lighter/types.hpp>

namespace qvl::image {

using namespace lighter::types;

/// 8-bit RGB, row-major, no padding.
struct Rgb {
    u32 width = 0;
    u32 height = 0;
    std::vector<u8> pixels;
};

/// Decodes JPEG, PNG, GIF (first frame), BMP and similar. Transparency is
/// composited onto white, the way Qwen's reference pastes RGBA images.
std::expected<Rgb, std::string> decode(std::span<const std::byte> bytes);

/// Catmull-Rom resampling (close to Pillow's bicubic), in gamma space as Pillow does.
Rgb resize(const Rgb &image, u32 width, u32 height) pre(width > 0 && height > 0);

/// Scales `image` to fit inside size x size, keeping its aspect ratio (up or
/// down), and centres it on white. Every result has the same shape, which keeps
/// the vision encoder to one input shape.
Rgb letterbox(const Rgb &image, u32 size) pre(size > 0);

struct Size {
    u32 width = 0;
    u32 height = 0;
};

/// Qwen's smart_resize: the nearest size whose sides are multiples of
/// `factor`, scaled into [min_pixels, max_pixels]. Fails on aspect ratios over 200.
std::expected<Size, std::string> smart_resize(u32 width, u32 height, u32 factor = 32, u64 min_pixels = 4 * 32 * 32,
                                              u64 max_pixels = 1800 * 32 * 32);

inline constexpr u32 k_patch_size = 16;
inline constexpr u32 k_merge_size = 2;
inline constexpr u32 k_temporal_patch_size = 2;
/// Values per patch: channels x temporal x patch x patch.
inline constexpr u32 k_patch_values = 3 * k_temporal_patch_size * k_patch_size * k_patch_size;

/// Qwen3-VL's pixel_values for one still image: grid_h x grid_w patches of
/// 16x16 pixels, each flattened as (channel, frame, y, x) with the frame
/// duplicated, rows ordered by 2x2 merge block. Values are (p - 127.5) / 127.5.
struct Patches {
    u32 grid_h = 0;
    u32 grid_w = 0;
    /// [grid_h * grid_w, k_patch_values]
    std::vector<float> values;
};

Patches patchify(const Rgb &image) pre(image.width % (k_patch_size * k_merge_size) == 0 &&
                                       image.height % (k_patch_size * k_merge_size) == 0 && image.width > 0 && image.height > 0);

} // namespace qvl::image
