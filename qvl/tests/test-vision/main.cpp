#include <cmath>
#include <cstdio>
#include <exception>
#include <format>
#include <string>

#include <qvl/tests/support/fixtures.h>
#include <qvl/vision/vision.h>

namespace {

using namespace qvl::test;
using qvl::vision::Grid;

double relative_error(const std::vector<float> &expected, const std::vector<float> &actual) {
    double worst = 0;
    for (usize i = 0; i < expected.size(); ++i) {
        worst = std::max(worst, std::abs(double(expected[i]) - actual[i]) / std::max(1.0, std::abs(double(expected[i]))));
    }
    return worst;
}

/// Interpolation indices, weights and rotary angles match transformers'.
void test_tables(const Fixtures &fixtures) {
    for (const auto grid : {Grid{16, 16}, Grid{4, 6}, Grid{4, 4}, Grid{18, 32}}) {
        const auto name = std::format("{}x{}", grid.h, grid.w);
        const auto table = qvl::vision::position_table(grid);
        require(table.indices == fixtures.array<i64>("pos-idx-" + name), name + ": position indices differ");
        const auto weights = relative_error(fixtures.array<float>("pos-weight-" + name), table.weights);
        require(weights < 1e-6, std::format("{}: position weights differ by {}", name, weights));
        const auto rotary = relative_error(fixtures.array<float>("rotary-" + name), qvl::vision::rotary(grid));
        require(rotary < 1e-6, std::format("{}: rotary angles differ by {}", name, rotary));
    }
}

} // namespace

int main() {
    try {
        test_tables(Fixtures::load());
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
