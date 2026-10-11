#include "vision.h"

#include <algorithm>
#include <cmath>

namespace qvl::vision {

namespace {

constexpr u32 k_merge = 2;

/// torch.linspace(start, end, steps) in float32: the first half counts up from
/// start, the second half down from end, which is not quite start + i * step.
std::vector<float> linspace(float start, float end, u32 steps) {
    std::vector<float> out(steps);
    if (steps == 1) {
        out[0] = start;
        return out;
    }
    const float step = (end - start) / static_cast<float>(steps - 1);
    const u32 halfway = steps / 2;
    for (u32 i = 0; i < steps; ++i) {
        out[i] = i < halfway ? start + step * static_cast<float>(i) : end - step * static_cast<float>(steps - i - 1);
    }
    return out;
}

/// Calls `fn(row, column)` for every patch in merge-block order.
template <typename Fn>
void for_each_merged(Grid grid, Fn &&fn) {
    for (u32 block_y = 0; block_y < grid.h / k_merge; ++block_y) {
        for (u32 block_x = 0; block_x < grid.w / k_merge; ++block_x) {
            for (u32 inner_y = 0; inner_y < k_merge; ++inner_y) {
                for (u32 inner_x = 0; inner_x < k_merge; ++inner_x) {
                    fn(block_y * k_merge + inner_y, block_x * k_merge + inner_x);
                }
            }
        }
    }
}

} // namespace

PositionTable position_table(Grid grid) {
    constexpr auto last = static_cast<float>(k_position_grid - 1);
    const auto ys = linspace(0, last, grid.h);
    const auto xs = linspace(0, last, grid.w);

    PositionTable out;
    const usize n = grid.patches();
    out.indices.resize(4 * n);
    out.weights.resize(4 * n);
    usize k = 0;
    for_each_merged(grid, [&](u32 row, u32 column) {
        const float y = ys[row];
        const float x = xs[column];
        const auto y0 = static_cast<i64>(y);
        const auto x0 = static_cast<i64>(x);
        const auto y1 = std::min<i64>(y0 + 1, k_position_grid - 1);
        const auto x1 = std::min<i64>(x0 + 1, k_position_grid - 1);
        const float dy = y - static_cast<float>(y0);
        const float dx = x - static_cast<float>(x0);
        const i64 indices[4] = {y0 * k_position_grid + x0, y0 * k_position_grid + x1, y1 * k_position_grid + x0, y1 * k_position_grid + x1};
        const float weights[4] = {(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx};
        for (usize corner = 0; corner < 4; ++corner) {
            out.indices[corner * n + k] = indices[corner];
            out.weights[corner * n + k] = weights[corner];
        }
        ++k;
    });
    return out;
}

std::vector<float> rotary(Grid grid) {
    constexpr u32 half = k_rotary_dim / 2;
    float inverse[half];
    for (u32 i = 0; i < half; ++i) {
        // 1 / theta^(2i / dim) in float32, as torch computes inv_freq.
        const float exponent = static_cast<float>(2 * i) / static_cast<float>(k_rotary_dim);
        inverse[i] = 1.0f / std::pow(10000.0f, exponent);
    }

    std::vector<float> out;
    out.reserve(static_cast<usize>(grid.patches()) * k_rotary_dim);
    for_each_merged(grid, [&](u32 row, u32 column) {
        for (u32 i = 0; i < half; ++i) out.push_back(static_cast<float>(row) * inverse[i]);
        for (u32 i = 0; i < half; ++i) out.push_back(static_cast<float>(column) * inverse[i]);
    });
    return out;
}

} // namespace qvl::vision
