#pragma once

#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>

namespace qvl::tokenizer {

using namespace lighter::types;

/// Qwen2's byte-level BPE, configured from a Hugging Face tokenizer.json the
/// way the `tokenizers` library applies it: added tokens are split out first,
/// the rest is NFC-normalized, cut by the Split pre-tokenizer's regex, mapped
/// byte-level and merged by rank; the post-processor's template then adds its
/// special tokens (Qwen3-VL-Embedding appends <|endoftext|>, the reranker
/// nothing).
///
/// Immutable after load(), so encode() may run on any thread.
struct Tokenizer {
    Tokenizer() noexcept;
    Tokenizer(Tokenizer &&) noexcept;
    Tokenizer &operator=(Tokenizer &&) noexcept;
    ~Tokenizer();

    static std::expected<Tokenizer, std::string> load(const std::filesystem::path &tokenizer_json);

    /// Token ids with the post-processor's special tokens (add_special_tokens=True).
    std::vector<i32> encode(std::string_view text) const;

    /// Token ids without the post-processor.
    std::vector<i32> encode_plain(std::string_view text) const;

    /// The id of a vocabulary or added token, e.g. "<|image_pad|>".
    std::optional<i32> token_id(std::string_view token) const;

    struct Self;

private:
    explicit Tokenizer(std::unique_ptr<Self> self) noexcept;

    std::unique_ptr<Self> self;
};

} // namespace qvl::tokenizer
