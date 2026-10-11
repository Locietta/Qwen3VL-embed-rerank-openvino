#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <format>
#include <optional>
#include <span>
#include <string>

#include <qvl/engine/engine.h>
#include <qvl/image/image.h>
#include <qvl/prompt/prompt.h>
#include <qvl/tests/support/fixtures.h>

namespace {

using namespace qvl::test;
namespace engine = qvl::engine;

std::optional<std::string> optional_text(const glz::generic &line, std::string_view key) {
    return line.contains(key) && line[key].is_string() ? std::optional(line[key].get_string()) : std::nullopt;
}

/// Fixture images, letterboxed as the service does. Only the 256x256 ones
/// match the reference input (the encoder takes one shape).
std::optional<qvl::image::Patches> fixture_image(const glz::generic &line, u32 size) {
    const auto name = optional_text(line, "image");
    if (!name) return std::nullopt;
    const auto bytes = read_file(fixture_dir() / (*name + ".png"));
    auto decoded = qvl::image::decode(std::as_bytes(std::span(bytes)));
    require(decoded.has_value(), "fixture image");
    return qvl::image::patchify(qvl::image::letterbox(*decoded, size));
}

bool usable(const glz::generic &line) {
    const auto name = optional_text(line, "image");
    return !name || name->ends_with("-256");
}

engine::Sequence embedding_sequence(engine::Engine &model, const glz::generic &line) {
    engine::Sequence out;
    auto image = fixture_image(line, model.options().image_size);
    const qvl::prompt::Content content{.text = optional_text(line, "text"), .image_tokens = image ? 256u * 256 / 32 / 32 : 0u};
    out.ids = model.tokenizer().encode(qvl::prompt::embedding(optional_text(line, "instruction").value_or(""), content));
    if (image) out.images.push_back(std::move(*image));
    return out;
}

std::vector<float> normalized(std::span<const float> row) {
    double norm = 0;
    for (float v : row) norm += double(v) * v;
    std::vector<float> out(row.begin(), row.end());
    for (auto &v : out) v = static_cast<float>(v / std::sqrt(norm));
    return out;
}

double dot(std::span<const float> a, std::span<const float> b) {
    double out = 0;
    for (usize i = 0; i < a.size(); ++i) out += double(a[i]) * b[i];
    return out;
}

engine::Engine load(engine::Kind kind, std::string_view name, std::string device) {
    const auto started = std::chrono::steady_clock::now();
    auto model =
        engine::Engine::load(kind, {.dir = *model_dir(name), .language_device = device, .vision_device = device, .warm_up = false});
    require(model.has_value(), model ? "" : model.error());
    std::printf("loaded %.*s on %s in %.1fs\n", static_cast<int>(name.size()), name.data(), device.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    return std::move(*model);
}

/// Embeddings match the OpenVINO reference (same IR); on CPU near-exactly,
/// which checks every host-side step. Batching with padding changes nothing.
void test_embeddings(const Fixtures &fixtures, const std::string &device, double text_floor, double image_floor) {
    auto model = load(engine::Kind::EMBEDDING, k_embedding_model, device);
    const auto reference = fixtures.array<float>("embeddings-ov");
    const auto ground_truth = fixtures.array<float>("embeddings-torch");
    const auto hidden = model.hidden();

    std::vector<engine::Sequence> sequences;
    std::vector<usize> cases;
    const auto &lines = fixtures["embeddings"].get_array();
    for (usize i = 0; i < lines.size(); ++i) {
        if (!usable(lines[i])) continue;
        sequences.push_back(embedding_sequence(model, lines[i]));
        cases.push_back(i);
    }

    std::vector<std::vector<float>> singles;
    for (usize k = 0; k < sequences.size(); ++k) {
        const auto started = std::chrono::steady_clock::now();
        auto hidden_state = model.last_hidden(std::span(sequences).subspan(k, 1));
        require(hidden_state.has_value(), hidden_state ? "" : hidden_state.error());
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        auto embedding = normalized(*hidden_state);
        const auto i = cases[k];
        const double ov = dot(embedding, std::span(reference).subspan(i * hidden, hidden));
        const double torch = dot(embedding, std::span(ground_truth).subspan(i * hidden, hidden));
        const bool image = !sequences[k].images.empty();
        std::printf("  %s case %zu (%s, %zu tokens): cosine %.5f vs OpenVINO reference, %.5f vs PyTorch, %.0f ms\n", device.c_str(), i,
                    image ? "image" : "text", sequences[k].ids.size(), ov, torch, ms);
        require(ov >= (image ? image_floor : text_floor), std::format("{} case {}: cosine {} vs the OpenVINO reference", device, i, ov));
        singles.push_back(std::move(embedding));
    }

    // Text cases as one padded batch.
    std::vector<engine::Sequence> texts;
    std::vector<usize> text_index;
    for (usize k = 0; k < sequences.size(); ++k) {
        if (sequences[k].images.empty()) {
            texts.push_back(sequences[k]);
            text_index.push_back(k);
        }
    }
    auto batched = model.last_hidden(texts);
    require(batched.has_value(), batched ? "" : batched.error());
    for (usize j = 0; j < texts.size(); ++j) {
        const double same = dot(normalized(std::span(*batched).subspan(j * hidden, hidden)), singles[text_index[j]]);
        require(same >= text_floor, std::format("{}: batched vs single cosine {}", device, same));
    }
    std::printf("  %s: a batch of %zu matches the single runs\n", device.c_str(), texts.size());
}

/// Rerank scores match the OpenVINO reference.
void test_rerank(const Fixtures &fixtures, const std::string &device, double tolerance) {
    auto model = load(engine::Kind::RERANKER, k_reranker_model, device);
    const auto &cases = fixtures["rerank"];
    const auto &expected = fixtures["rerank_scores_ov"].get_array();
    const auto &query = cases["query"];
    const auto &documents = cases["documents"].get_array();
    for (usize i = 0; i < documents.size(); ++i) {
        if (!usable(documents[i])) continue;
        engine::Sequence sequence;
        auto image = fixture_image(documents[i], model.options().image_size);
        const qvl::prompt::Content document{.text = optional_text(documents[i], "text"), .image_tokens = image ? 64u : 0u};
        sequence.ids = model.tokenizer().encode(qvl::prompt::rerank("", {.text = optional_text(query, "text")}, document));
        if (image) sequence.images.push_back(std::move(*image));
        auto hidden_state = model.last_hidden(std::span(&sequence, 1));
        require(hidden_state.has_value(), hidden_state ? "" : hidden_state.error());
        const double score = 1 / (1 + std::exp(-dot(*hidden_state, model.rerank_head())));
        const double want = expected[i].get_number();
        std::printf("  %s document %zu: score %.4f, reference %.4f\n", device.c_str(), i, score, want);
        require(std::abs(score - want) <= tolerance, std::format("{} document {}: score {} vs {}", device, i, score, want));
    }
}

} // namespace

int main() {
    try {
        if (!model_dir(k_embedding_model)) return 0;
        const auto fixtures = Fixtures::load();
        test_embeddings(fixtures, "CPU", 0.9999, 0.9995);
        test_rerank(fixtures, "CPU", 0.002);
        test_embeddings(fixtures, "GPU", 0.995, 0.98);
        test_rerank(fixtures, "GPU", 0.03);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
