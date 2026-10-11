#pragma once

#include <expected>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <lighter/types.hpp>
#include <qvl/vision/vision.h>

namespace qvl::prompt {

using namespace lighter::types;

inline constexpr std::string_view k_embedding_instruction = "Represent the user's input.";
inline constexpr std::string_view k_rerank_instruction = "Given a search query, retrieve relevant candidates that answer the query.";
inline constexpr std::string_view k_rerank_system =
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\".";

/// One side of an input: optional text and an optional image of `image_tokens`
/// placeholders (vision::Grid::tokens()).
struct Content {
    std::optional<std::string> text;
    u32 image_tokens = 0;
};

/// Qwen3-VL-Embedding's chat prompt, as its reference embedder renders it with
/// add_generation_prompt: the instruction as the system turn (a full stop
/// appended unless it ends in punctuation), then the image and the text as the
/// user turn ("NULL" when both are absent).
std::string embedding(std::string_view instruction, const Content &content);

/// Qwen3-VL-Reranker's chat prompt: the fixed judging system turn, then
/// "<Instruct>: ...<Query>:...\n<Document>:..." with images before text.
std::string rerank(std::string_view instruction, const Content &query, const Content &document);

/// Qwen3-VL's M-RoPE position ids for one sequence, as Qwen3VLModel.get_rope_index:
/// text counts up on all three rows; image k's tokens take (start, start + row,
/// start + column) on its merged grid, and the text after it resumes at
/// start + max(rows, columns). Every run of `image_token` is one image, matched
/// in order with `grids`. Returns [3, ids.size()] row-major.
std::expected<std::vector<i64>, std::string> position_ids(std::span<const i32> ids, i32 image_token, std::span<const vision::Grid> grids);

} // namespace qvl::prompt
