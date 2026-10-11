#include "image.h"

#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

namespace qvl::image {

std::expected<Rgb, std::string> decode(std::span<const std::byte> bytes) {
    int width = 0;
    int height = 0;
    int channels = 0;
    auto *rgba = stbi_load_from_memory(reinterpret_cast<const stbi_uc *>(bytes.data()), static_cast<int>(bytes.size()), &width, &height,
                                       &channels, 4);
    if (!rgba) return std::unexpected(std::string("cannot decode the image: ") + stbi_failure_reason());

    Rgb out{.width = static_cast<u32>(width), .height = static_cast<u32>(height)};
    out.pixels.resize(static_cast<usize>(width) * static_cast<usize>(height) * 3);
    const usize count = static_cast<usize>(width) * static_cast<usize>(height);
    for (usize i = 0; i < count; ++i) {
        const u32 alpha = rgba[i * 4 + 3];
        for (usize c = 0; c < 3; ++c) {
            // Over white, rounded: (value * a + 255 * (255 - a)) / 255.
            const u32 value = rgba[i * 4 + c];
            out.pixels[i * 3 + c] = static_cast<u8>((value * alpha + 255 * (255 - alpha) + 127) / 255);
        }
    }
    stbi_image_free(rgba);
    return out;
}

Rgb resize(const Rgb &image, u32 width, u32 height) {
    if (image.width == width && image.height == height) return image;
    Rgb out{.width = width, .height = height};
    out.pixels.resize(static_cast<usize>(width) * height * 3);
    stbir_resize(image.pixels.data(), static_cast<int>(image.width), static_cast<int>(image.height), static_cast<int>(image.width * 3),
                 out.pixels.data(), static_cast<int>(width), static_cast<int>(height), static_cast<int>(width * 3), STBIR_RGB,
                 STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM);
    return out;
}

Rgb letterbox(const Rgb &image, u32 size) {
    const double scale = static_cast<double>(size) / std::max(image.width, image.height);
    const u32 width = std::clamp(static_cast<u32>(std::lround(image.width * scale)), 1u, size);
    const u32 height = std::clamp(static_cast<u32>(std::lround(image.height * scale)), 1u, size);
    const auto fitted = resize(image, width, height);

    Rgb out{.width = size, .height = size};
    out.pixels.assign(static_cast<usize>(size) * size * 3, 255);
    const u32 left = (size - width) / 2;
    const u32 top = (size - height) / 2;
    for (u32 y = 0; y < height; ++y) {
        std::copy_n(fitted.pixels.begin() + static_cast<std::ptrdiff_t>(y) * width * 3, width * 3,
                    out.pixels.begin() + (static_cast<std::ptrdiff_t>(top + y) * size + left) * 3);
    }
    return out;
}

std::expected<Size, std::string> smart_resize(u32 width, u32 height, u32 factor, u64 min_pixels, u64 max_pixels) {
    const double w = width;
    const double h = height;
    if (width == 0 || height == 0) return std::unexpected("empty image");
    if (std::max(w, h) / std::min(w, h) > 200) return std::unexpected("aspect ratio over 200");

    // Python's round() rounds half to even, as nearbyint does by default.
    const double f = factor;
    double h_bar = std::max(f, std::nearbyint(h / f) * f);
    double w_bar = std::max(f, std::nearbyint(w / f) * f);
    if (h_bar * w_bar > static_cast<double>(max_pixels)) {
        const double beta = std::sqrt(h * w / static_cast<double>(max_pixels));
        h_bar = std::max(f, std::floor(h / beta / f) * f);
        w_bar = std::max(f, std::floor(w / beta / f) * f);
    } else if (h_bar * w_bar < static_cast<double>(min_pixels)) {
        const double beta = std::sqrt(static_cast<double>(min_pixels) / (h * w));
        h_bar = std::ceil(h * beta / f) * f;
        w_bar = std::ceil(w * beta / f) * f;
    }
    return Size{.width = static_cast<u32>(w_bar), .height = static_cast<u32>(h_bar)};
}

Patches patchify(const Rgb &image) {
    Patches out{.grid_h = image.height / k_patch_size, .grid_w = image.width / k_patch_size};
    out.values.resize(static_cast<usize>(out.grid_h) * out.grid_w * k_patch_values);
    auto *row = out.values.data();
    for (u32 block_y = 0; block_y < out.grid_h / k_merge_size; ++block_y) {
        for (u32 block_x = 0; block_x < out.grid_w / k_merge_size; ++block_x) {
            for (u32 inner_y = 0; inner_y < k_merge_size; ++inner_y) {
                for (u32 inner_x = 0; inner_x < k_merge_size; ++inner_x) {
                    const u32 top = (block_y * k_merge_size + inner_y) * k_patch_size;
                    const u32 left = (block_x * k_merge_size + inner_x) * k_patch_size;
                    for (u32 c = 0; c < 3; ++c) {
                        for (u32 y = 0; y < k_patch_size; ++y) {
                            for (u32 x = 0; x < k_patch_size; ++x) {
                                const auto pixel = image.pixels[(static_cast<usize>(top + y) * image.width + left + x) * 3 + c];
                                const float value = (static_cast<float>(pixel) - 127.5f) / 127.5f;
                                // The still image fills both temporal frames.
                                for (u32 t = 0; t < k_temporal_patch_size; ++t) {
                                    row[(c * k_temporal_patch_size + t) * k_patch_size * k_patch_size + y * k_patch_size + x] = value;
                                }
                            }
                        }
                    }
                    row += k_patch_values;
                }
            }
        }
    }
    return out;
}

} // namespace qvl::image
