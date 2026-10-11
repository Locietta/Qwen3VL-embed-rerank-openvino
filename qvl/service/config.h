#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <lighter/types.hpp>

namespace qvl::service {

using namespace lighter::types;

struct ModelConfig {
    /// Directory name under models_dir (tools/export.py names them).
    std::string name;
    /// Names clients may ask for in "model", besides `name`.
    std::vector<std::string> aliases;
    std::string device = "GPU";
    /// Defaults to `device`.
    std::string vision_device;
};

/// ~/.config/qvl/config.json. Every field has a default; paths may start with "~/".
struct Config {
    std::string host = "127.0.0.1";
    i32 port = 8091;
    std::string models_dir = "~/services/qvl/models";
    /// OpenVINO's compiled-blob cache; empty disables it.
    std::string cache_dir = "~/.cache/qvl/openvino";
    std::optional<ModelConfig> embedding = ModelConfig{.name = "Qwen3-VL-Embedding-2B-int8", .aliases = {"Qwen3-VL-Embedding-2B"}};
    /// null serves no reranker.
    std::optional<ModelConfig> reranker = ModelConfig{.name = "Qwen3-VL-Reranker-2B-int8", .aliases = {"Qwen3-VL-Reranker-2B"}};
    /// Every image is letterboxed to this square (a multiple of 32).
    u32 image_size = 256;
    /// Directories whose files may be named by "image": {"path": ...}.
    std::vector<std::string> image_roots;
    std::vector<u32> length_buckets{64, 128, 256, 512, 1024, 2048};
    std::vector<u32> batch_buckets{1, 4, 16};
    u32 max_batch_tokens = 8192;
    bool dynamic_quantization = false;
    usize max_body_bytes = 32 * 1024 * 1024;
    usize max_inputs = 256;
    usize max_image_bytes = 16 * 1024 * 1024;
};

/// Reads and validates a config file.
std::expected<Config, std::string> load_config(const std::filesystem::path &path);

/// Expands a leading "~/" to $HOME.
std::filesystem::path expand_home(std::string_view path);

} // namespace qvl::service
