#include <cmath>
#include <cstdio>
#include <exception>
#include <format>
#include <span>
#include <string>

#include <qvl/image/image.h>
#include <qvl/tests/support/fixtures.h>

namespace {

using namespace qvl::test;
namespace image = qvl::image;

image::Rgb load(std::string_view name) {
    const auto bytes = read_file(fixture_dir() / name);
    auto decoded = image::decode(std::as_bytes(std::span(bytes)));
    require(decoded.has_value(), decoded ? "" : decoded.error());
    return std::move(*decoded);
}

/// pixel_values of images that need no resize match the processor's.
void test_patchify(const Fixtures &fixtures) {
    for (const std::string name : {"wide-96x64", "small-64"}) {
        const auto picture = load(name + ".png");
        const auto patches = image::patchify(picture);
        std::vector<usize> shape;
        const auto expected = fixtures.array<float>("pixels-" + name, &shape);
        require(shape.size() == 2 && shape[1] == image::k_patch_values, "pixel fixture shape");
        require(patches.grid_h * patches.grid_w == shape[0],
                std::format("{}: {} patches, expected {}", name, patches.grid_h * patches.grid_w, shape[0]));
        const auto &thw = fixtures["grids"][name].get_array();
        require(patches.grid_h == thw[1].get_number() && patches.grid_w == thw[2].get_number(), name + ": grid differs");
        double worst = 0;
        for (usize i = 0; i < expected.size(); ++i) worst = std::max(worst, std::abs(double(expected[i]) - patches.values[i]));
        require(worst < 1e-5, std::format("{}: pixel_values differ by {}", name, worst));
    }
}

/// Transparency is composited on white as PIL's paste does (within rounding).
void test_decode(const Fixtures &fixtures) {
    const auto rgba = load("rgba-200x120.png");
    require(rgba.width == 200 && rgba.height == 120, "RGBA size");
    const auto expected = fixtures.array<i64>("decoded-rgba-200x120");
    i64 worst = 0;
    for (usize i = 0; i < expected.size(); ++i) worst = std::max(worst, std::abs(expected[i] - static_cast<i64>(rgba.pixels[i])));
    require(worst <= 1, std::format("RGBA composite differs by {}", worst));

    const auto gif = load("anim.gif");
    require(gif.width == 48 && gif.height == 32, "GIF size");
    require(gif.pixels[0] == 255 && gif.pixels[1] == 0 && gif.pixels[2] == 0, "a GIF decodes to its first frame");

    const std::string junk = "not an image";
    require(!image::decode(std::as_bytes(std::span(junk))).has_value(), "junk is refused");
}

void test_letterbox() {
    const auto wide = load("rgba-200x120.png");
    const auto boxed = image::letterbox(wide, 256);
    require(boxed.width == 256 && boxed.height == 256, "letterbox size");
    // 200x120 fits as 256x154, centred: rows 0..50 stay white.
    require(boxed.pixels[0] == 255 && boxed.pixels[(10 * 256 + 128) * 3] == 255, "letterbox margin is white");
    const auto same = image::letterbox(load("card-256.png"), 256);
    require(same.pixels == load("card-256.png").pixels, "a 256x256 image passes through unchanged");
}

void test_smart_resize() {
    struct Case {
        u32 width, height, out_width, out_height;
    };
    // Rounded to multiples of 32; a tiny image is scaled up to 4096 px; a huge
    // one down under 1800 * 32 * 32; 48 -> 32 or 64 by round-half-even.
    for (const auto c : {Case{300, 500, 288, 512}, Case{256, 256, 256, 256}, Case{20, 20, 64, 64}, Case{4000, 3000, 1536, 1152},
                         Case{48, 80, 64, 64}, Case{80, 48, 64, 64}}) {
        auto size = image::smart_resize(c.width, c.height);
        require(size.has_value(), size ? "" : size.error());
        require(size->width == c.out_width && size->height == c.out_height,
                std::format("smart_resize({}x{}) = {}x{}, expected {}x{}", c.width, c.height, size->width, size->height, c.out_width,
                            c.out_height));
    }
    require(!image::smart_resize(10, 3000).has_value(), "aspect ratios over 200 are refused");
}

} // namespace

int main() {
    try {
        const auto fixtures = Fixtures::load();
        test_patchify(fixtures);
        test_decode(fixtures);
        test_letterbox();
        test_smart_resize();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
