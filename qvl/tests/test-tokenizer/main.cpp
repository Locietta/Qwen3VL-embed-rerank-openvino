#include <cstdio>
#include <exception>
#include <format>
#include <string>

#include <qvl/tests/support/fixtures.h>
#include <qvl/tokenizer/tokenizer.h>

namespace {

using namespace qvl::test;

std::string show(const std::vector<i32> &ids) {
    std::string out;
    for (auto id : ids) out += std::format("{} ", id);
    return out;
}

/// Every corpus line encodes to the ids the `tokenizers` library produced.
void test_corpus(const Fixtures &fixtures, std::string_view model) {
    const auto dir = model_dir(model);
    if (!dir) return;
    auto tokenizer = qvl::tokenizer::Tokenizer::load(*dir / "tokenizer.json");
    require(tokenizer.has_value(), tokenizer ? "" : tokenizer.error());

    usize checked = 0;
    for (const auto &line : fixtures["tokenizer"][model].get_array()) {
        const auto &text = line["text"].get_string();
        const auto expected = ids_of(line["ids"]);
        const auto actual = tokenizer->encode(text);
        require(actual == expected, std::format("{}: {:?}\n  expected {}\n  actual   {}", model, text, show(expected), show(actual)));
        ++checked;
    }
    std::printf("%.*s: %zu lines match\n", static_cast<int>(model.size()), model.data(), checked);

    require(tokenizer->token_id("<|image_pad|>") == 151655, "<|image_pad|> id");
    require(tokenizer->token_id("<|endoftext|>") == 151643, "<|endoftext|> id");
}

} // namespace

int main() {
    try {
        const auto fixtures = Fixtures::load();
        test_corpus(fixtures, k_embedding_model);
        test_corpus(fixtures, k_reranker_model);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
