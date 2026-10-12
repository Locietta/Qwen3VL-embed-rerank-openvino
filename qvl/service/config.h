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
    std::vector<u32> length_buckets{32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048};
    std::vector<u32> batch_buckets{1, 2, 4, 8, 12, 16};
    u32 max_batch_tokens = 8192;
    /// Dynamic quantization group of the language model's activations (0: off).
    u64 dynamic_quantization_group = 0;
    /// Run each rerank request's shared prefix (system, instruction, query) once.
    bool rerank_prefix_cache = true;
    /// Vision-encoder outputs in memory (~2 MiB per picture).
    usize image_cache_bytes = usize{512} << 20;
    /// ... and on disk, surviving restarts; empty disables the disk cache.
    std::string image_cache_dir = "~/.cache/qvl/vision";
    u64 image_cache_disk_bytes = u64{4} << 30;
    usize max_body_bytes = 32 * 1024 * 1024;
    usize max_inputs = 256;
    usize max_image_bytes = 16 * 1024 * 1024;
};

/// Reads and validates a config file.
std::expected<Config, std::string> load_config(const std::filesystem::path &path);

/// Expands a leading "~/" to $HOME.
std::filesystem::path expand_home(std::string_view path);

} // namespace qvl::service
