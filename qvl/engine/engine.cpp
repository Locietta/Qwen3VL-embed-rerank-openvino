#include "engine.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <list>
#include <unordered_map>
#include <exception>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>

#include <glaze/json.hpp>
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

/// `quantization_group`: OpenVINO's dynamic quantization of activations for
/// int8 weights (0: off; UINT64_MAX: per token).
ov::AnyMap compile_config(std::string_view device, u64 quantization_group) {
    ov::AnyMap out{{ov::hint::performance_mode.name(), ov::hint::PerformanceMode::LATENCY}};
    out.emplace(ov::hint::dynamic_quantization_group_size.name(), quantization_group);
    // The CPU plugin compresses the KV cache to u8 by default, which moves
    // reranker scores by ~0.01 when a prefix is reused; keep it exact.
    out.emplace(ov::hint::kv_cache_precision.name(), device == "CPU" ? ov::element::f32 : ov::element::f16);
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

    /// One picture through the vision encoder: its language-model rows
    /// [tokens, hidden] and its deepstack rows [3, tokens, hidden].
    struct Encoded {
        std::vector<float> rows;
        std::vector<float> deep;
        usize bytes() const { return (rows.size() + deep.size()) * sizeof(float); }
    };
    /// Most recently used first.
    std::list<std::pair<std::string, std::shared_ptr<const Encoded>>> lru;
    std::unordered_map<std::string, decltype(lru)::iterator> cache;
    usize cache_bytes = 0;

    /// The disk cache's directory (empty: none) and its running size.
    std::filesystem::path disk;
    u64 disk_bytes = 0;

    Timings timings;
    /// The language model keeps its KV cache (Options::prefix_cache).
    bool stateful = false;

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

    std::shared_ptr<const Encoded> encode_image(const image::Patches &patches) {
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
        return std::make_shared<const Encoded>(Encoded{
            .rows = std::vector<float>(rows.data<float>(), rows.data<float>() + rows.get_size()),
            .deep = std::vector<float>(deep.data<float>(), deep.data<float>() + deep.get_size()),
        });
    }

    /// Into the memory cache, evicting the least recently used.
    void remember(const std::string &key, std::shared_ptr<const Encoded> encoded) {
        if (encoded->bytes() > options.image_cache_bytes) return;
        lru.emplace_front(key, std::move(encoded));
        cache.emplace(key, lru.begin());
        cache_bytes += lru.front().second->bytes();
        while (cache_bytes > options.image_cache_bytes) {
            cache_bytes -= lru.back().second->bytes();
            cache.erase(lru.back().first);
            lru.pop_back();
        }
    }

    /// Keys become file names: only hash characters.
    static bool safe_key(std::string_view key) {
        return !key.empty() && key.size() <= 128 &&
               std::ranges::all_of(key, [](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_'; });
    }

    struct DiskHeader {
        char magic[8] = {'Q', 'V', 'L', 'V', 'I', 'S', '1', '\0'};
        u64 rows = 0;
        u64 deep = 0;
    };

    std::shared_ptr<const Encoded> load(const std::string &key) {
        if (disk.empty() || !safe_key(key)) return nullptr;
        const auto path = disk / (key + ".bin");
        std::ifstream in(path, std::ios::binary);
        if (!in) return nullptr;
        DiskHeader header;
        in.read(reinterpret_cast<char *>(&header), sizeof header);
        const usize rows = static_cast<usize>(grid.tokens()) * hidden;
        if (!in || std::memcmp(header.magic, DiskHeader{}.magic, sizeof header.magic) != 0 || header.rows != rows ||
            header.deep != 3 * rows) {
            return nullptr;
        }
        Encoded out{.rows = std::vector<float>(rows), .deep = std::vector<float>(3 * rows)};
        in.read(reinterpret_cast<char *>(out.rows.data()), static_cast<std::streamsize>(rows * sizeof(float)));
        in.read(reinterpret_cast<char *>(out.deep.data()), static_cast<std::streamsize>(3 * rows * sizeof(float)));
        if (!in) return nullptr;
        std::error_code ec;
        std::filesystem::last_write_time(path, std::filesystem::file_time_type::clock::now(), ec); // recently used
        return std::make_shared<const Encoded>(std::move(out));
    }

    /// Best effort: a write that fails only costs a later re-encode.
    void store(const std::string &key, const Encoded &encoded) {
        if (disk.empty() || !safe_key(key)) return;
        const auto path = disk / (key + ".bin");
        const auto partial = disk / (key + ".tmp");
        {
            std::ofstream out(partial, std::ios::binary | std::ios::trunc);
            const DiskHeader header{.rows = encoded.rows.size(), .deep = encoded.deep.size()};
            out.write(reinterpret_cast<const char *>(&header), sizeof header);
            out.write(reinterpret_cast<const char *>(encoded.rows.data()),
                      static_cast<std::streamsize>(encoded.rows.size() * sizeof(float)));
            out.write(reinterpret_cast<const char *>(encoded.deep.data()),
                      static_cast<std::streamsize>(encoded.deep.size() * sizeof(float)));
            if (!out) return;
        }
        std::error_code ec;
        std::filesystem::rename(partial, path, ec);
        if (ec) return;
        disk_bytes += sizeof(DiskHeader) + encoded.bytes();
        if (disk_bytes > options.image_cache_disk_bytes) prune();
    }

    /// Drops the least recently used files until the cache is under 90% of its size.
    void prune() {
        std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> files;
        std::error_code ec;
        disk_bytes = 0;
        for (const auto &entry : std::filesystem::directory_iterator(disk, ec)) {
            if (entry.path().extension() != ".bin") continue;
            files.emplace_back(entry.last_write_time(ec), entry.path());
            disk_bytes += entry.file_size(ec);
        }
        std::ranges::sort(files);
        for (const auto &[time, path] : files) {
            if (disk_bytes <= options.image_cache_disk_bytes / 10 * 9) break;
            const auto size = std::filesystem::file_size(path, ec);
            if (std::filesystem::remove(path, ec)) disk_bytes -= size;
        }
    }

    /// From memory, then disk.
    std::shared_ptr<const Encoded> find(const std::string &key) {
        if (key.empty()) return nullptr;
        if (const auto found = cache.find(key); found != cache.end()) {
            lru.splice(lru.begin(), lru, found->second);
            ++timings.images_cached;
            return found->second->second;
        }
        if (auto loaded = load(key)) {
            remember(key, loaded);
            ++timings.images_cached;
            ++timings.images_from_disk;
            return loaded;
        }
        return nullptr;
    }

    /// The encoder output of `image`, from the cache when its key is known.
    std::shared_ptr<const Encoded> encoded(const Image &image) {
        if (image.encoded) return std::static_pointer_cast<const Encoded>(image.encoded);
        if (auto found = find(image.key)) return found;
        if (!image.patches) throw std::runtime_error("an uncached image without patches");
        const vision::Grid seen{.h = image.patches->grid_h, .w = image.patches->grid_w};
        if (seen != grid) {
            throw std::runtime_error(std::format("an image of {}x{} patches; the encoder takes {}x{}", seen.h, seen.w, grid.h, grid.w));
        }
        const auto started = std::chrono::steady_clock::now();
        auto out = encode_image(*image.patches);
        timings.vision_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        ++timings.images_encoded;
        if (!image.key.empty()) {
            remember(image.key, out);
            store(image.key, *out);
        }
        return out;
    }

    /// A prefix already in the language model's KV state (stateful mode).
    struct Past {
        const Sequence *prefix = nullptr;
        /// Its length in the state, right-padded to a bucket.
        usize length = 0;
        /// [length]: 1 for its tokens, 0 for the padding after them.
        std::vector<i64> mask;
    };

    /// Runs `prefix` alone, right-padded, leaving its keys and values in the
    /// state for the continuations. Right padding keeps every attention row non-empty.
    Past run_prefix(const Sequence &prefix) {
        const Sequence *one = &prefix;
        run(std::span<const Sequence *const>(&one, 1), nullptr);
        Past out{.prefix = &prefix, .length = bucket_length(prefix.ids.size())};
        out.mask.assign(out.length, 0);
        std::fill_n(out.mask.begin(), prefix.ids.size(), 1);
        return out;
    }

    /// `batch` from a fresh state, or, with `past`, as continuations of the
    /// prefix in the state (every row reads it through beam_idx 0).
    std::vector<float> run(std::span<const Sequence> batch, const Past *past = nullptr) {
        std::vector<const Sequence *> pointers;
        for (const auto &sequence : batch) pointers.push_back(&sequence);
        return run(std::span<const Sequence *const>(pointers), past);
    }

    std::vector<float> run(std::span<const Sequence *const> batch, const Past *past) {
        using clock = std::chrono::steady_clock;
        const auto seconds = [](clock::time_point since) { return std::chrono::duration<double>(clock::now() - since).count(); };
        const auto started = clock::now();
        double device = 0; // vision, token embeddings and language model, excluded from host time
        ++timings.runs;
        usize longest = 0;
        for (const auto *sequence : batch) longest = std::max(longest, sequence->ids.size());
        const usize length = bucket_length(longest);
        const usize rows = bucket_batch(batch.size());
        if (length == 0) throw std::runtime_error(std::format("a sequence of {} tokens is longer than the largest bucket", longest));
        if (rows == 0 || rows > max_batch(static_cast<u32>(longest))) {
            throw std::runtime_error(std::format("a batch of {} at {} tokens is too large", batch.size(), length));
        }
        const usize before = past ? past->length : 0;
        const usize span = before + length; // attention mask width

        std::vector<i64> ids(rows * length, k_pad_token);
        std::vector<i64> mask(rows * span, 0);
        std::vector<i64> positions(3 * rows * length, 1);
        std::vector<u8> visual(rows * length, 0);
        std::vector<i64> last(rows, 0);
        std::vector<i32> beams(rows, 0);
        if (!past) std::iota(beams.begin(), beams.end(), 0);

        // Image rows in batch order: row-major over (sequence, position), the
        // order the graph's deepstack cumsum reads them in.
        std::vector<std::shared_ptr<const Encoded>> pictures;
        for (usize b = 0; b < rows; ++b) {
            if (past) std::copy(past->mask.begin(), past->mask.end(), mask.begin() + static_cast<std::ptrdiff_t>(b * span));
            if (b >= batch.size()) {
                // Padding rows hold one token, so their attention is never empty.
                mask[b * span + before] = 1;
                for (usize r = 0; r < 3; ++r) positions[(r * rows + b) * length] = 0;
                continue;
            }
            const auto &sequence = *batch[b];
            // Positions come from the whole sequence (prefix and continuation).
            std::vector<i32> whole;
            usize images = sequence.images.size();
            if (past) {
                whole = past->prefix->ids;
                images += past->prefix->images.size();
            }
            whole.insert(whole.end(), sequence.ids.begin(), sequence.ids.end());
            const std::vector<vision::Grid> grids(images, grid);
            auto placed = prompt::position_ids(whole, image_token, grids);
            if (!placed) throw std::runtime_error(placed.error());
            const usize n = sequence.ids.size();
            const usize offset = whole.size() - n;
            for (usize t = 0; t < n; ++t) {
                ids[b * length + t] = sequence.ids[t];
                mask[b * span + before + t] = 1;
                visual[b * length + t] = sequence.ids[t] == image_token;
                for (usize r = 0; r < 3; ++r) positions[(r * rows + b) * length + t] = (*placed)[r * whole.size() + offset + t];
            }
            last[b] = static_cast<i64>(n - 1);
            for (const auto &image : sequence.images) {
                const auto vision_started = clock::now();
                pictures.push_back(encoded(image));
                device += seconds(vision_started);
            }
        }

        const auto text_started = clock::now();
        text.set_input_tensor(view(ov::element::i64, {rows, length}, ids));
        text.infer();
        const auto token_embeds = text.get_output_tensor(0);
        const double text_seconds = seconds(text_started);
        timings.token_embedding_seconds += text_seconds;
        device += text_seconds;
        std::vector<float> embeds(token_embeds.data<float>(), token_embeds.data<float>() + token_embeds.get_size());

        // Image rows replace their placeholders' token embeddings.
        usize image_tokens = 0;
        for (const auto &picture : pictures) image_tokens += picture->rows.size() / hidden;
        std::vector<float> deepstack;
        if (image_tokens > 0) {
            usize image = 0;
            usize within = 0;
            for (usize i = 0; i < rows * length; ++i) {
                if (!visual[i]) continue;
                std::memcpy(&embeds[i * hidden], &pictures[image]->rows[within * hidden], hidden * sizeof(float));
                if (++within * hidden == pictures[image]->rows.size()) {
                    ++image;
                    within = 0;
                }
            }
            deepstack.resize(3 * image_tokens * hidden);
            for (usize layer = 0; layer < 3; ++layer) {
                usize at = layer * image_tokens * hidden;
                for (const auto &picture : pictures) {
                    const usize count = picture->deep.size() / 3;
                    std::memcpy(&deepstack[at], &picture->deep[layer * count], count * sizeof(float));
                    at += count;
                }
            }
        } else {
            // The graph indexes row 0 even without images; an empty tensor yields NaN.
            deepstack.assign(3 * hidden, 0.0f);
        }

        if (!past && stateful) language.reset_state();
        language.set_tensor("attention_mask", view(ov::element::i64, {rows, span}, mask));
        language.set_tensor("position_ids", view(ov::element::i64, {3, rows, length}, positions));
        language.set_tensor("inputs_embeds", view(ov::element::f32, {rows, length, hidden}, embeds));
        language.set_tensor("visual_pos_masks", view(ov::element::boolean, {rows, length}, visual));
        language.set_tensor("deepstack_visual_embeds", view(ov::element::f32, {3, std::max<usize>(image_tokens, 1), hidden}, deepstack));
        language.set_tensor("beam_idx", view(ov::element::i32, {rows}, beams));
        language.set_tensor("last_index", view(ov::element::i64, {rows, 1}, last));
        const auto language_started = clock::now();
        language.infer();
        const auto out = language.get_output_tensor(0);
        const double language_seconds = seconds(language_started);
        timings.language_seconds += language_seconds;
        device += language_seconds;
        const auto *data = out.data<float>();
        std::vector<float> result(data, data + batch.size() * hidden);
        timings.host_seconds += seconds(started) - device;
        return result;
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
        std::vector<Sequence> batch{
            Sequence{.ids = std::move(text), .images = {Image{.patches = std::make_shared<const image::Patches>(image::patchify(blank))}}}};
        run(batch);
    }
};

std::optional<Split> split_prefix(std::span<const Sequence> sequences, usize shared_images, i32 image_token, usize min_prefix) {
    if (sequences.size() < 2) return std::nullopt;
    const auto &first = sequences.front().ids;
    usize common = first.size() - 1;
    for (const auto &sequence : sequences) {
        if (sequence.ids.empty() || sequence.images.size() < shared_images) return std::nullopt;
        common = std::min(common, sequence.ids.size() - 1);
        const auto [a, b] = std::mismatch(first.begin(), first.begin() + static_cast<std::ptrdiff_t>(common), sequence.ids.begin());
        common = static_cast<usize>(a - first.begin());
    }
    // Stop at the first picture beyond the shared ones: identical placeholders
    // do not mean identical pictures.
    usize runs = 0;
    for (usize i = 0; i < common; ++i) {
        if (first[i] == image_token && (i == 0 || first[i - 1] != image_token) && ++runs > shared_images) {
            common = i;
            break;
        }
    }
    // The prefix holds exactly the shared pictures, none cut in two.
    if (common > 0 && first[common - 1] == image_token && common < first.size() && first[common] == image_token) return std::nullopt;
    usize in_prefix = 0;
    for (usize i = 0; i < common; ++i) in_prefix += first[i] == image_token && (i == 0 || first[i - 1] != image_token);
    if (in_prefix != shared_images || common < min_prefix) return std::nullopt;

    Split out;
    out.prefix.ids.assign(first.begin(), first.begin() + static_cast<std::ptrdiff_t>(common));
    out.prefix.images.assign(sequences.front().images.begin(),
                             sequences.front().images.begin() + static_cast<std::ptrdiff_t>(shared_images));
    for (const auto &sequence : sequences) {
        Sequence rest;
        rest.ids.assign(sequence.ids.begin() + static_cast<std::ptrdiff_t>(common), sequence.ids.end());
        rest.images.assign(sequence.images.begin() + static_cast<std::ptrdiff_t>(shared_images), sequence.images.end());
        out.rest.push_back(std::move(rest));
    }
    return out;
}

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
        self->stateful = options.prefix_cache;
        if (!self->stateful) make_stateless(language);
        gather_last(language);
        self->hidden = static_cast<usize>(language->output(0).get_partial_shape()[2].get_length());
        self->language = core.compile_model(language, options.language_device,
                                            compile_config(options.language_device, options.dynamic_quantization_group))
                             .create_infer_request();
        self->text = core.compile_model(path("text_embeddings"), options.language_device,
                                        compile_config(options.language_device, options.dynamic_quantization_group))
                         .create_infer_request();

        // The vision encoder sees one shape, compiled statically.
        self->grid = {.h = options.image_size / image::k_patch_size, .w = options.image_size / image::k_patch_size};
        const usize n = self->grid.patches();
        auto patch = core.read_model(path("vision_embeddings"));
        patch->reshape(ov::PartialShape{static_cast<i64>(n), image::k_patch_values});
        self->patch = core.compile_model(patch, options.vision_device, compile_config(options.vision_device, 0)).create_infer_request();
        auto merger = core.read_model(path("vision_embeddings_merger"));
        merger->reshape(std::map<std::string, ov::PartialShape>{
            {"hidden_states", ov::PartialShape{static_cast<i64>(n), static_cast<i64>(k_vision_hidden)}},
            {"attention_mask", ov::PartialShape{1, static_cast<i64>(n), static_cast<i64>(n)}},
            {"rotary_pos_emb", ov::PartialShape{static_cast<i64>(n), vision::k_rotary_dim}},
        });
        self->merger = core.compile_model(merger, options.vision_device, compile_config(options.vision_device, 0)).create_infer_request();

        // Position embeddings depend only on the grid: interpolate them once.
        auto table = vision::position_table(self->grid);
        auto position = core.compile_model(path("vision_embeddings_pos"), "CPU", compile_config("CPU", 0)).create_infer_request();
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

        // The disk cache is keyed by model, revision and image size, so a
        // re-export or another letterbox never reads stale encoder outputs.
        if (!options.image_cache_dir.empty()) {
            std::string revision = "unknown";
            if (std::ifstream manifest(options.dir / "manifest.json"); manifest) {
                std::stringstream text;
                text << manifest.rdbuf();
                glz::generic parsed;
                if (!glz::read<glz::opts{.null_terminated = false}>(parsed, text.str()) && parsed.contains("revision") &&
                    parsed["revision"].is_string()) {
                    revision = parsed["revision"].get_string().substr(0, 12);
                }
            }
            self->disk = options.image_cache_dir / std::format("{}-{}-{}", options.dir.filename().string(), revision, options.image_size);
            std::error_code ec;
            std::filesystem::create_directories(self->disk, ec);
            if (ec) return std::unexpected("cannot create " + self->disk.string() + ": " + ec.message());
            for (const auto &entry : std::filesystem::directory_iterator(self->disk, ec)) {
                if (entry.path().extension() == ".tmp")
                    std::filesystem::remove(entry.path(), ec);
                else if (entry.path().extension() == ".bin")
                    self->disk_bytes += entry.file_size(ec);
            }
        }

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
std::shared_ptr<const void> Engine::lookup(std::string_view key) { return self->find(std::string(key)); }
Timings Engine::timings() const noexcept { return self->timings; }

std::expected<std::vector<float>, std::string> Engine::last_hidden_after(const Sequence &prefix, std::span<const Sequence> continuations) {
    if (!self->stateful) return std::unexpected("prefix reuse needs Options::prefix_cache");
    if (continuations.empty()) return std::vector<float>();
    try {
        const auto past = self->run_prefix(prefix);
        ++self->timings.prefixes_run;
        return self->run(continuations, &past);
    } catch (const std::exception &error) {
        return std::unexpected(error.what());
    }
}

std::expected<std::vector<float>, std::string> Engine::last_hidden(std::span<const Sequence> batch) {
    if (batch.empty()) return std::vector<float>();
    try {
        return self->run(batch);
    } catch (const std::exception &error) {
        return std::unexpected(error.what());
    }
}

} // namespace qvl::engine
