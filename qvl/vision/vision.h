#pragma once

#include <contracts>
#include <vector>

#include <lighter/types.hpp>

namespace qvl::vision {

using namespace lighter::types;

/// An image in patches (16 px each); both sides even, for the 2x2 merge.
struct Grid {
    u32 h = 0;
    u32 w = 0;

    u32 patches() const noexcept { return h * w; }
    /// Language-model tokens after the 2x2 merge.
    u32 tokens() const noexcept { return h * w / 4; }

    friend bool operator==(const Grid &, const Grid &) = default;
};

/// Side of the learned position-embedding table (48 x 48 entries).
inline constexpr u32 k_position_grid = 48;

/// Bilinear interpolation of the 48x48 position table onto `grid`, as
/// Qwen3VLVisionModel.fast_pos_embed_interpolate does: for each patch, four
/// table indices and their weights, rows already in merge-block order so the
/// weighted sum lines up with pixel_values.
struct PositionTable {
    /// [4, patches]: the corners floor/floor, floor/ceil, ceil/floor, ceil/ceil.
    std::vector<i64> indices;
    /// [4, patches], matching `indices`.
    std::vector<float> weights;
};

PositionTable position_table(Grid grid) pre(grid.h % 2 == 0 && grid.w % 2 == 0 && grid.h > 0 && grid.w > 0);

inline constexpr u32 k_rotary_dim = 32;

/// The vision rotary frequencies, as Qwen3VLVisionModel.rot_pos_emb: per patch
/// (merge-block order) 16 row and 16 column angles, theta 10000. [patches, 32]
std::vector<float> rotary(Grid grid) pre(grid.h % 2 == 0 && grid.w % 2 == 0 && grid.h > 0 && grid.w > 0);

} // namespace qvl::vision
