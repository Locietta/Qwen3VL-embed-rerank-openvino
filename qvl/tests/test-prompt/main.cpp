#include <array>
#include <cstdio>
#include <exception>
#include <format>
#include <optional>
#include <string>

#include <qvl/prompt/prompt.h>
#include <qvl/tests/support/fixtures.h>
#include <qvl/tokenizer/tokenizer.h>

namespace {

using namespace qvl::test;
using qvl::vision::Grid;

std::optional<std::string> optional_text(const glz::generic &value) {
    return value.is_string() ? std::optional(value.get_string()) : std::nullopt;
}

/// The embedder's prompts match the chat template; with the tokenizer, so do
/// the ids and the M-RoPE positions of image prompts.
void test_embedding_prompts(const Fixtures &fixtures) {
    const auto dir = model_dir(k_embedding_model);
    std::optional<qvl::tokenizer::Tokenizer> tokenizer;
    if (dir) {
        auto loaded = qvl::tokenizer::Tokenizer::load(*dir / "tokenizer.json");
        require(loaded.has_value(), loaded ? "" : loaded.error());
        tokenizer = std::move(*loaded);
    }

    for (const auto &line : fixtures["embedding_prompts"].get_array()) {
        const auto instruction = optional_text(line["instruction"]);
        const qvl::prompt::Content content{.text = optional_text(line["text"]),
                                           .image_tokens = static_cast<u32>(line["image_tokens"].get_number())};
        const auto prompt = qvl::prompt::embedding(instruction.value_or(""), content);
        require(prompt == line["prompt"].get_string(),
                std::format("prompt differs:\n  expected {:?}\n  actual   {:?}", line["prompt"].get_string(), prompt));
        if (!tokenizer) continue;

        const auto ids = tokenizer->encode(prompt);
        require(ids == ids_of(line["ids"]), "prompt ids differ for " + prompt.substr(0, 80));
        if (!line["image"].is_string()) continue;

        const auto &name = line["image"].get_string();
        const auto &thw = fixtures["grids"][name].get_array();
        const std::array grids{Grid{.h = static_cast<u32>(thw[1].get_number()), .w = static_cast<u32>(thw[2].get_number())}};
        auto positions = qvl::prompt::position_ids(ids, 151655, grids);
        require(positions.has_value(), positions ? "" : positions.error());
        require(*positions == fixtures.array<i64>("positions-" + name), "M-RoPE positions differ for " + name);
    }
}

void test_rerank_prompt() {
    const auto prompt = qvl::prompt::rerank("", {.text = "q"}, {.text = std::nullopt, .image_tokens = 2});
    const std::string expected =
        "<|im_start|>system\n" + std::string(qvl::prompt::k_rerank_system) +
        "<|im_end|>\n<|im_start|>user\n<Instruct>: " + std::string(qvl::prompt::k_rerank_instruction) +
        "<Query>:q\n<Document>:<|vision_start|><|image_pad|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n";
    require(prompt == expected, "rerank prompt: " + prompt);
    const auto empty = qvl::prompt::rerank("Find it", {}, {.text = "d"});
    require(empty.contains("<Instruct>: Find it<Query>:NULL\n<Document>:d<|im_end|>"), "empty query side: " + empty);
}

void test_positions_two_images() {
    // text(2) image(2x2 merged = 4 tokens) text(1) image(1x2 = 2 tokens) text(1)
    const std::vector<i32> ids{1, 2, 9, 9, 9, 9, 3, 9, 9, 4};
    const std::array grids{Grid{.h = 4, .w = 4}, Grid{.h = 2, .w = 4}};
    auto positions = qvl::prompt::position_ids(ids, 9, grids);
    require(positions.has_value(), positions ? "" : positions.error());
    const std::vector<i64> expected{
        0, 1, 2, 2, 2, 2, 4, 5, 5, 7, // t
        0, 1, 2, 2, 3, 3, 4, 5, 5, 7, // h
        0, 1, 2, 3, 2, 3, 4, 5, 6, 7, // w
    };
    require(*positions == expected, "two-image positions");
    require(!qvl::prompt::position_ids(ids, 9, std::span(grids).first(1)).has_value(), "a missing grid is refused");
}

} // namespace

int main() {
    try {
        const auto fixtures = Fixtures::load();
        test_embedding_prompts(fixtures);
        test_rerank_prompt();
        test_positions_two_images();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
