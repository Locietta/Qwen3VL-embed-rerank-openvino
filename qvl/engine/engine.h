#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>
#include <qvl/image/image.h>
#include <qvl/tokenizer/tokenizer.h>

namespace qvl::engine {

using namespace lighter::types;

enum struct Kind { EMBEDDING, RERANKER };

struct Options {
    /// The model directory written by tools/export.py.
    std::filesystem::path dir;
    /// OpenVINO devices for the language model (and token embeddings) and for the vision encoder.
    std::string language_device = "GPU";
    std::string vision_device = "GPU";
    /// Compiled-blob cache; empty for none.
    std::filesystem::path cache_dir;
    /// Sequence lengths a batch is padded to. The model only ever sees these;
    /// finer buckets waste less on padding and cost no memory.
    std::vector<u32> length_buckets{32, 64, 96, 128, 160, 192, 256, 320, 384, 512, 768, 1024, 1536, 2048};
    /// Batch sizes a batch is padded to.
    std::vector<u32> batch_buckets{1, 2, 4, 8, 12, 16};
    /// Largest batch x length the model runs at once.
    u32 max_batch_tokens = 8192;
    /// Side of the square every image is letterboxed to (multiple of 32).
    u32 image_size = 256;
    /// OpenVINO's activation quantization for int8 weights. Off by default: the
    /// int8 vision merger is numerically unstable with it.
    bool dynamic_quantization = false;
    /// Run one text and one image at load time, so the first requests are fast.
    bool warm_up = true;
    /// Vision-encoder outputs kept by image key (about 2 MiB per 256x256 image);
    /// a picture seen again skips the encoder. 0 disables the cache.
    usize image_cache_bytes = usize{512} << 20;
};

/// One picture of a prompt.
struct Image {
    /// Identifies the picture for the encoder cache (e.g. a hash of its file);
    /// empty: never cached.
    std::string key;
    /// Its patches; may be null when `encoded` is set.
    std::shared_ptr<const image::Patches> patches;
    /// The cached encoder output from Engine::lookup(key), held so eviction
    /// cannot take it away before the sequence runs.
    std::shared_ptr<const void> encoded;
};

/// One prompt ready for the model: its token ids and, for every run of
/// <|image_pad|> in order, the picture.
struct Sequence {
    std::vector<i32> ids;
    std::vector<Image> images;
};

/// Cumulative time per stage, for /metrics.
struct Timings {
    double vision_seconds = 0;
    double token_embedding_seconds = 0;
    double language_seconds = 0;
    /// Building inputs and copying tensors on the host.
    double host_seconds = 0;
    u64 images_encoded = 0;
    u64 images_cached = 0;
    u64 runs = 0;
};

/// A Qwen3-VL backbone on OpenVINO, up to the last hidden state of each
/// sequence. The language model is made stateless (an embedding pass needs no
/// KV cache) and gathers only the last real token in the graph; the vision
/// encoder is compiled for the one image shape. Not thread-safe: one caller at
/// a time (the service's worker thread).
struct Engine {
    Engine() noexcept;
    Engine(Engine &&) noexcept;
    Engine &operator=(Engine &&) noexcept;
    ~Engine();

    static std::expected<Engine, std::string> load(Kind kind, const Options &options);

    Kind kind() const noexcept;
    const Options &options() const noexcept;
    const tokenizer::Tokenizer &tokenizer() const noexcept;
    i32 image_token() const noexcept;
    /// Hidden size (2048 for the 2B models).
    usize hidden() const noexcept;
    /// The longest sequence a batch may hold.
    u32 max_length() const noexcept;
    /// The largest batch for sequences of `length` tokens.
    u32 max_batch(u32 length) const noexcept;
    /// lm_head["yes"] - lm_head["no"] for the reranker; empty for the embedder.
    std::span<const float> rerank_head() const noexcept;

    /// The cached encoder output of the picture with this key, or null; for Image::encoded.
    std::shared_ptr<const void> lookup(std::string_view key);

    Timings timings() const noexcept;

    /// [batch.size(), hidden()] row-major: each sequence's hidden state at its
    /// last token. All sequences must fit max_length(), and the batch max_batch().
    std::expected<std::vector<float>, std::string> last_hidden(std::span<const Sequence> batch);

    struct Self;

private:
    explicit Engine(std::unique_ptr<Self> self) noexcept;

    std::unique_ptr<Self> self;
};

} // namespace qvl::engine
