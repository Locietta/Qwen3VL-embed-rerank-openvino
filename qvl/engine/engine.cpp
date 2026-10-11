#include "engine.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <map>
#include <numeric>

#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/util/read_value_base.hpp>
#include <openvino/openvino.hpp>

#include <qvl/prompt/prompt.h>
#include <qvl/vision/vision.h>

namespace qvl::engine {

namespace {

constexpr usize k_vision_hidden = 1024;
constexpr i32 k_pad_token = 151643;

/// Drops the KV cache: every ReadValue becomes its initializer and the Assigns
/// go, so attention runs over the current tokens only and nothing persists.
void make_stateless(const std::shared_ptr<ov::Model> &model) {
    const auto sinks = model->get_sinks(); // a copy: remove_sink() edits the model's list
    for (const auto &sink : sinks) model->remove_sink(sink);
    for (const auto &op : model->get_ordered_ops()) {
        if (auto read = ov::as_type_ptr<ov::op::util::ReadValueBase>(op)) {
            if (read->get_input_size() != 1) throw std::runtime_error("a KV-cache state without an initializer");
            read->output(0).replace(read->input_value(0));
        }
    }
    for (const auto &variable : model->get_variables()) model->remove_variable(variable);
    model->validate_nodes_and_infer_types();
}

/// Replaces the [batch, length, hidden] output with the rows at the new input
/// last_index [batch, 1]: only each sequence's last token leaves the device.
void gather_last(const std::shared_ptr<ov::Model> &model) {
    auto result = model->get_results().at(0);
    auto last = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1, 1});
    last->set_friendly_name("last_index");
    last->output(0).get_tensor().set_names({"last_index"});
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto gathered = std::make_shared<ov::op::v8::Gather>(result->input_value(0), last, axis, 1);
    gathered->output(0).get_tensor().set_names({"last_hidden"});
    result->input(0).replace_source_output(gathered->output(0));
    model->add_parameters({last});
    model->validate_nodes_and_infer_types();
}

ov::AnyMap compile_config(const Options &options, std::string_view device) {
    ov::AnyMap out{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};
    if (!options.dynamic_quantization) out.emplace(ov::hint::dynamic_quantization_group_size.name(), uint64_t{0});
    if (device == "CPU") out.emplace(ov::hint::inference_precision.name(), ov::element::f32);
    return out;
}

template <typename T>
ov::Tensor view(ov::element::Type type, ov::Shape shape, std::vector<T> &data) {
    return ov::Tensor(type, std::move(shape), data.data());
}

std::expected<std::vector<float>, std::string> read_floats(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return std::unexpected("cannot read " + path.string());
    const auto size = static_cast<usize>(stream.tellg());
    std::vector<float> out(size / sizeof(float));
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(out.data()), static_cast<std::streamsize>(out.size() * sizeof(float)));
    return out;
}

} // namespace

struct Engine::Self {
    Kind kind = Kind::EMBEDDING;
    Options options;
    tokenizer::Tokenizer tokenizer;
    i32 image_token = 0;
    usize hidden = 0;
    std::vector<float> head;

    ov::Core core;
    ov::InferRequest text;
    ov::InferRequest language;
    ov::InferRequest patch;
    ov::InferRequest merger;

    vision::Grid grid;
    /// The interpolated position embeddings of `grid`, [patches, 1024].
    std::vector<float> position_embeds;
    std::vector<float> rotary;
    std::vector<float> attention_mask;

    u32 bucket_length(usize length) const {
        for (auto bucket : options.length_buckets) {
            if (length <= bucket) return bucket;
        }
        return 0;
    }

    u32 bucket_batch(usize count) const {
        for (auto bucket : options.batch_buckets) {
            if (count <= bucket) return bucket;
        }
        return 0;
    }

    u32 max_batch(u32 length) const {
        const auto bucket = bucket_length(length);
        u32 out = 1;
        for (auto batch : options.batch_buckets) {
            if (static_cast<u64>(batch) * bucket <= options.max_batch_tokens) out = std::max(out, batch);
        }
        return out;
    }

    /// One image through the vision encoder: its language-model rows
    /// [tokens, hidden] and its deepstack rows [3, tokens, hidden].
    std::pair<std::vector<float>, std::vector<float>> encode_image(const image::Patches &patches) {
        const usize n = grid.patches();
        auto &values = const_cast<std::vector<float> &>(patches.values);
        patch.set_input_tensor(view(ov::element::f32, {n, image::k_patch_values}, values));
        patch.infer();
        const auto embedded = patch.get_output_tensor(0);
        std::vector<float> states(n * k_vision_hidden);
        const auto *source = embedded.data<float>();
        for (usize i = 0; i < states.size(); ++i) states[i] = source[i] + position_embeds[i];

        merger.set_tensor("hidden_states", view(ov::element::f32, {n, k_vision_hidden}, states));
        merger.set_tensor("attention_mask", view(ov::element::f32, {1, n, n}, attention_mask));
        merger.set_tensor("rotary_pos_emb", view(ov::element::f32, {n, vision::k_rotary_dim}, rotary));
        merger.infer();
        const auto rows = merger.get_output_tensor(0);
        const auto deep = merger.get_output_tensor(1);
        return {std::vector<float>(rows.data<float>(), rows.data<float>() + rows.get_size()),
                std::vector<float>(deep.data<float>(), deep.data<float>() + deep.get_size())};
    }

    std::vector<float> run(std::span<const Sequence> batch) {
        usize longest = 0;
        for (const auto &sequence : batch) longest = std::max(longest, sequence.ids.size());
        const usize length = bucket_length(longest);
        const usize rows = bucket_batch(batch.size());
        if (length == 0) throw std::runtime_error(std::format("a sequence of {} tokens is longer than the largest bucket", longest));
        if (rows == 0 || rows > max_batch(static_cast<u32>(longest))) {
            throw std::runtime_error(std::format("a batch of {} at {} tokens is too large", batch.size(), length));
        }

        std::vector<i64> ids(rows * length, k_pad_token);
        std::vector<i64> mask(rows * length, 0);
        std::vector<i64> positions(3 * rows * length, 1);
        std::vector<u8> visual(rows * length, 0);
        std::vector<i64> last(rows, 0);
        std::vector<i32> beams(rows);
        std::iota(beams.begin(), beams.end(), 0);

        // Image rows in batch order: row-major over (sequence, position), the
        // order the graph's deepstack cumsum reads them in.
        std::vector<std::vector<float>> image_rows;
        std::vector<std::vector<float>> deep_rows;
        for (usize b = 0; b < rows; ++b) {
            if (b >= batch.size()) {
                // Padding rows hold one token, so their attention is never empty.
                mask[b * length] = 1;
                for (usize r = 0; r < 3; ++r) positions[(r * rows + b) * length] = 0;
                continue;
            }
            const auto &sequence = batch[b];
            std::vector<vision::Grid> grids;
            for (const auto &image : sequence.images) {
                const vision::Grid seen{.h = image.grid_h, .w = image.grid_w};
                if (seen != grid)
                    throw std::runtime_error(
                        std::format("an image of {}x{} patches; the encoder takes {}x{}", seen.h, seen.w, grid.h, grid.w));
                grids.push_back(seen);
            }
            auto placed = prompt::position_ids(sequence.ids, image_token, grids);
            if (!placed) throw std::runtime_error(placed.error());
            const usize n = sequence.ids.size();
            for (usize t = 0; t < n; ++t) {
                ids[b * length + t] = sequence.ids[t];
                mask[b * length + t] = 1;
                visual[b * length + t] = sequence.ids[t] == image_token;
                for (usize r = 0; r < 3; ++r) positions[(r * rows + b) * length + t] = (*placed)[r * n + t];
            }
            last[b] = static_cast<i64>(n - 1);
            for (const auto &image : sequence.images) {
                auto [lm, deep] = encode_image(image);
                image_rows.push_back(std::move(lm));
                deep_rows.push_back(std::move(deep));
            }
        }

        text.set_input_tensor(view(ov::element::i64, {rows, length}, ids));
        text.infer();
        const auto token_embeds = text.get_output_tensor(0);
        std::vector<float> embeds(token_embeds.data<float>(), token_embeds.data<float>() + token_embeds.get_size());

        // Image rows replace their placeholders' token embeddings.
        usize image_tokens = 0;
        for (const auto &rows_of_image : image_rows) image_tokens += rows_of_image.size() / hidden;
        std::vector<float> deepstack;
        if (image_tokens > 0) {
            usize image = 0;
            usize within = 0;
            for (usize i = 0; i < rows * length; ++i) {
                if (!visual[i]) continue;
                std::memcpy(&embeds[i * hidden], &image_rows[image][within * hidden], hidden * sizeof(float));
                if (++within * hidden == image_rows[image].size()) {
                    ++image;
                    within = 0;
                }
            }
            deepstack.resize(3 * image_tokens * hidden);
            for (usize layer = 0; layer < 3; ++layer) {
                usize at = layer * image_tokens * hidden;
                for (const auto &deep : deep_rows) {
                    const usize count = deep.size() / 3;
                    std::memcpy(&deepstack[at], &deep[layer * count], count * sizeof(float));
                    at += count;
                }
            }
        } else {
            // The graph indexes row 0 even without images; an empty tensor yields NaN.
            deepstack.assign(3 * hidden, 0.0f);
        }

        language.set_tensor("attention_mask", view(ov::element::i64, {rows, length}, mask));
        language.set_tensor("position_ids", view(ov::element::i64, {3, rows, length}, positions));
        language.set_tensor("inputs_embeds", view(ov::element::f32, {rows, length, hidden}, embeds));
        language.set_tensor("visual_pos_masks", view(ov::element::boolean, {rows, length}, visual));
        language.set_tensor("deepstack_visual_embeds", view(ov::element::f32, {3, std::max<usize>(image_tokens, 1), hidden}, deepstack));
        language.set_tensor("beam_idx", view(ov::element::i32, {rows}, beams));
        language.set_tensor("last_index", view(ov::element::i64, {rows, 1}, last));
        language.infer();
        const auto out = language.get_output_tensor(0);
        const auto *data = out.data<float>();
        return std::vector<float>(data, data + batch.size() * hidden);
    }

    /// Runs one short text and one image, so the first requests do not pay for
    /// compiling the graphs. Other shapes compile in milliseconds on first use,
    /// and memory does not grow with their number (measured, 2026-10-11).
    void warm_up() {
        std::vector<Sequence> text_batch{Sequence{.ids = std::vector<i32>(options.length_buckets.front(), k_pad_token)}};
        run(text_batch);
        auto text = tokenizer.encode(prompt::embedding("", {.image_tokens = grid.tokens()}));
        image::Rgb blank{.width = options.image_size, .height = options.image_size};
        blank.pixels.assign(static_cast<usize>(blank.width) * blank.height * 3, 255);
        std::vector<Sequence> batch{Sequence{.ids = std::move(text), .images = {image::patchify(blank)}}};
        run(batch);
    }
};

Engine::Engine() noexcept = default;
Engine::Engine(Engine &&) noexcept = default;
Engine &Engine::operator=(Engine &&) noexcept = default;
Engine::~Engine() = default;
Engine::Engine(std::unique_ptr<Self> self) noexcept : self(std::move(self)) {}

std::expected<Engine, std::string> Engine::load(Kind kind, const Options &options) {
    if (options.image_size == 0 || options.image_size % 32 != 0) return std::unexpected("image_size must be a positive multiple of 32");
    auto self = std::make_unique<Self>();
    self->kind = kind;
    self->options = options;
    std::ranges::sort(self->options.length_buckets);
    std::ranges::sort(self->options.batch_buckets);

    auto tokenizer = tokenizer::Tokenizer::load(options.dir / "tokenizer.json");
    if (!tokenizer) return std::unexpected(tokenizer.error());
    self->tokenizer = std::move(*tokenizer);
    const auto image_token = self->tokenizer.token_id("<|image_pad|>");
    if (!image_token) return std::unexpected("the tokenizer has no <|image_pad|>");
    self->image_token = *image_token;

    try {
        auto &core = self->core;
        if (!options.cache_dir.empty()) core.set_property(ov::cache_dir(options.cache_dir.string()));
        auto path = [&](std::string_view name) { return (options.dir / std::format("openvino_{}_model.xml", name)).string(); };

        auto language = core.read_model(path("language"));
        make_stateless(language);
        gather_last(language);
        self->hidden = static_cast<usize>(language->output(0).get_partial_shape()[2].get_length());
        self->language =
            core.compile_model(language, options.language_device, compile_config(options, options.language_device)).create_infer_request();
        self->text = core.compile_model(path("text_embeddings"), options.language_device, compile_config(options, options.language_device))
                         .create_infer_request();

        // The vision encoder sees one shape, compiled statically.
        self->grid = {.h = options.image_size / image::k_patch_size, .w = options.image_size / image::k_patch_size};
        const usize n = self->grid.patches();
        auto patch = core.read_model(path("vision_embeddings"));
        patch->reshape(ov::PartialShape{static_cast<i64>(n), image::k_patch_values});
        self->patch =
            core.compile_model(patch, options.vision_device, compile_config(options, options.vision_device)).create_infer_request();
        auto merger = core.read_model(path("vision_embeddings_merger"));
        merger->reshape(std::map<std::string, ov::PartialShape>{
            {"hidden_states", ov::PartialShape{static_cast<i64>(n), static_cast<i64>(k_vision_hidden)}},
            {"attention_mask", ov::PartialShape{1, static_cast<i64>(n), static_cast<i64>(n)}},
            {"rotary_pos_emb", ov::PartialShape{static_cast<i64>(n), vision::k_rotary_dim}},
        });
        self->merger =
            core.compile_model(merger, options.vision_device, compile_config(options, options.vision_device)).create_infer_request();

        // Position embeddings depend only on the grid: interpolate them once.
        auto table = vision::position_table(self->grid);
        auto position = core.compile_model(path("vision_embeddings_pos"), "CPU", compile_config(options, "CPU")).create_infer_request();
        position.set_input_tensor(view(ov::element::i64, {4, n}, table.indices));
        position.infer();
        const auto corners = position.get_output_tensor(0);
        const auto *corner = corners.data<float>();
        self->position_embeds.assign(n * k_vision_hidden, 0.0f);
        for (usize c = 0; c < 4; ++c) {
            for (usize i = 0; i < n; ++i) {
                const float weight = table.weights[c * n + i];
                for (usize d = 0; d < k_vision_hidden; ++d) {
                    self->position_embeds[i * k_vision_hidden + d] += corner[(c * n + i) * k_vision_hidden + d] * weight;
                }
            }
        }
        self->rotary = vision::rotary(self->grid);
        self->attention_mask.assign(n * n, 0.0f);

        if (kind == Kind::RERANKER) {
            auto head = read_floats(options.dir / "rerank_head.f32");
            if (!head) return std::unexpected(head.error());
            if (head->size() != self->hidden) return std::unexpected("rerank_head.f32 does not match the hidden size");
            self->head = std::move(*head);
        }
        if (options.warm_up) self->warm_up();
    } catch (const std::exception &error) {
        return std::unexpected(std::string("OpenVINO: ") + error.what());
    }
    return Engine(std::move(self));
}

Kind Engine::kind() const noexcept { return self->kind; }
const Options &Engine::options() const noexcept { return self->options; }
const tokenizer::Tokenizer &Engine::tokenizer() const noexcept { return self->tokenizer; }
i32 Engine::image_token() const noexcept { return self->image_token; }
usize Engine::hidden() const noexcept { return self->hidden; }
u32 Engine::max_length() const noexcept { return self->options.length_buckets.back(); }
u32 Engine::max_batch(u32 length) const noexcept { return self->max_batch(length); }
std::span<const float> Engine::rerank_head() const noexcept { return self->head; }

std::expected<std::vector<float>, std::string> Engine::last_hidden(std::span<const Sequence> batch) {
    if (batch.empty()) return std::vector<float>();
    try {
        return self->run(batch);
    } catch (const std::exception &error) {
        return std::unexpected(error.what());
    }
}

} // namespace qvl::engine
