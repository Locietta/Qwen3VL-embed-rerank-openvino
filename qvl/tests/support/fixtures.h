#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <glaze/json.hpp>

#include <lighter/types.hpp>

namespace qvl::test {

using namespace lighter::types;

inline void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

inline std::string read_file(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot read " + path.string());
    std::stringstream buffer;
    buffer << stream.rdbuf();
    return std::move(buffer).str();
}

/// qvl/tests/fixtures, written by tools/make_fixtures.py.
inline std::filesystem::path fixture_dir() { return std::filesystem::path(QVL_FIXTURE_DIR); }

/// The exported models (tools/export.py), from QVL_TEST_MODELS; tests that need
/// them are skipped without it.
inline std::optional<std::filesystem::path> model_dir(std::string_view name) {
    const char *root = std::getenv("QVL_TEST_MODELS");
    if (!root || !*root) {
        std::printf("skipped: QVL_TEST_MODELS is not set\n");
        return std::nullopt;
    }
    return std::filesystem::path(root) / name;
}

inline constexpr std::string_view k_embedding_model = "Qwen3-VL-Embedding-2B-int8";
inline constexpr std::string_view k_reranker_model = "Qwen3-VL-Reranker-2B-int8";

struct Fixtures {
    glz::generic index;

    static Fixtures load() {
        Fixtures out;
        const auto text = read_file(fixture_dir() / "index.json");
        auto ec = glz::read<glz::opts{.null_terminated = false}>(out.index, text);
        require(!ec, "invalid fixtures/index.json: " + glz::format_error(ec, text));
        return out;
    }

    const glz::generic &operator[](std::string_view key) const {
        require(index.contains(key), "fixtures lack " + std::string(key));
        return index[key];
    }

    template <typename T>
    std::vector<T> array(std::string_view name, std::vector<usize> *shape = nullptr) const {
        const auto &entry = (*this)["arrays"][name];
        const auto bytes = read_file(fixture_dir() / entry["file"].get_string());
        std::vector<T> out(bytes.size() / sizeof(T));
        std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
        if (shape) {
            shape->clear();
            for (const auto &dim : entry["shape"].get_array()) shape->push_back(static_cast<usize>(dim.get_number()));
        }
        return out;
    }
};

inline std::vector<i32> ids_of(const glz::generic &array) {
    std::vector<i32> out;
    for (const auto &id : array.get_array()) out.push_back(static_cast<i32>(id.get_number()));
    return out;
}

} // namespace qvl::test
