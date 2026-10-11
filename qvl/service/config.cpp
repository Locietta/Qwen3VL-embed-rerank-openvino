#include "config.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <glaze/json.hpp>

namespace qvl::service {

std::filesystem::path expand_home(std::string_view path) {
    if (path == "~" || path.starts_with("~/")) {
        const char *home = std::getenv("HOME");
        return std::filesystem::path(home ? home : "") / std::string(path.substr(std::min<usize>(path.size(), 2)));
    }
    return std::filesystem::path(std::string(path));
}

std::expected<Config, std::string> load_config(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return std::unexpected("cannot read " + path.string());
    std::stringstream buffer;
    buffer << stream.rdbuf();
    const auto text = std::move(buffer).str();

    Config config;
    if (auto ec = glz::read<glz::opts{.null_terminated = false, .comments = true}>(config, text)) {
        return std::unexpected("invalid config: " + glz::format_error(ec, text));
    }
    if (config.port <= 0 || config.port > 65535) return std::unexpected("port must be 1-65535");
    if (config.image_size == 0 || config.image_size % 32 != 0) return std::unexpected("image_size must be a positive multiple of 32");
    if (config.length_buckets.empty() || config.batch_buckets.empty()) return std::unexpected("buckets must not be empty");
    if (!config.embedding && !config.reranker) return std::unexpected("neither an embedding model nor a reranker is configured");
    for (auto *model : {config.embedding ? &*config.embedding : nullptr, config.reranker ? &*config.reranker : nullptr}) {
        if (!model) continue;
        if (model->name.empty()) return std::unexpected("a model without a name");
        if (model->vision_device.empty()) model->vision_device = model->device;
    }
    return config;
}

} // namespace qvl::service
