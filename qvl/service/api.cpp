#include "api.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

#include <glaze/json.hpp>

namespace qvl::service {

namespace {

namespace http = lighter::http;

struct ImageField {
    std::optional<std::string> path;
    std::optional<std::string> data;
};

struct InputField {
    std::optional<std::string> text;
    std::optional<ImageField> image;
};

using InputValue = std::variant<std::string, InputField>;

struct EmbeddingsBody {
    std::string model;
    std::variant<std::string, std::vector<InputValue>> input;
    std::optional<std::string> instruction;
    std::optional<u32> dimensions;
    std::optional<std::string> priority;
    std::optional<std::string> encoding_format;
};

struct RerankBody {
    std::string model;
    InputValue query;
    std::vector<InputValue> documents;
    std::optional<std::string> instruction;
    std::optional<usize> top_n;
    std::optional<std::string> priority;
};

struct Usage {
    usize prompt_tokens = 0;
    usize total_tokens = 0;
};

struct EmbeddingDatum {
    std::string_view object = "embedding";
    usize index = 0;
    std::vector<float> embedding;
};

struct EmbeddingsReply {
    std::string_view object = "list";
    std::vector<EmbeddingDatum> data;
    std::string model;
    Usage usage;
};

struct RerankDatum {
    usize index = 0;
    float relevance_score = 0;
};

struct RerankReply {
    std::vector<RerankDatum> results;
    std::string model;
    Usage usage;
};

struct ErrorBody {
    std::string message;
    std::string_view type;
};

struct ErrorReply {
    ErrorBody error;
};

struct ModelEntry {
    std::string id;
    std::string_view object = "model";
    std::string_view owned_by = "qvl";
};

struct ModelList {
    std::string_view object = "list";
    std::vector<ModelEntry> data;
};

struct WorkerMetrics {
    std::string model;
    u64 jobs = 0;
    u64 sequences = 0;
    u64 batches = 0;
    u64 tokens = 0;
    double busy_seconds = 0;
    usize queued = 0;
};

struct Metrics {
    std::vector<WorkerMetrics> workers;
    u64 rss_bytes = 0;
};

constexpr glz::opts k_read{.null_terminated = false, .error_on_unknown_keys = false};

http::ServerResponse json(i32 status, const auto &value) {
    std::string body;
    std::ignore = glz::write_json(value, body);
    return http::ServerResponse::json(status, std::move(body));
}

http::ServerResponse error(i32 status, std::string message) {
    const std::string_view type = status >= 500 ? "server_error" : status == 404 ? "not_found_error" : "invalid_request_error";
    return json(status, ErrorReply{.error = {.message = std::move(message), .type = type}});
}

std::optional<std::string> base64_decode(std::string_view text) {
    constexpr auto table = [] {
        std::array<i8, 256> out{};
        out.fill(-1);
        for (i32 i = 0; i < 26; ++i) {
            out['A' + i] = static_cast<i8>(i);
            out['a' + i] = static_cast<i8>(26 + i);
        }
        for (i32 i = 0; i < 10; ++i) out['0' + i] = static_cast<i8>(52 + i);
        out['+'] = out['-'] = 62;
        out['/'] = out['_'] = 63;
        return out;
    }();
    std::string out;
    out.reserve(text.size() * 3 / 4);
    u32 buffer = 0;
    i32 bits = 0;
    for (unsigned char c : text) {
        if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
        const i8 value = table[c];
        if (value < 0) return std::nullopt;
        buffer = (buffer << 6) | static_cast<u32>(value);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out += static_cast<char>((buffer >> bits) & 0xFF);
        }
    }
    return out;
}

u64 resident_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.starts_with("VmRSS:")) {
            std::istringstream fields(line.substr(6));
            u64 kib = 0;
            fields >> kib;
            return kib * 1024;
        }
    }
    return 0;
}

bool serves(const std::optional<ModelConfig> &model, std::string_view name) {
    if (!model) return false;
    return name.empty() || name == model->name || std::ranges::find(model->aliases, name) != model->aliases.end();
}

} // namespace

Api::Api(const Config &config, Worker *embedder, Worker *reranker) : config(&config), embedder(embedder), reranker(reranker) {
    for (const auto &root : config.image_roots) {
        std::error_code ec;
        auto canonical = std::filesystem::weakly_canonical(expand_home(root), ec);
        if (!ec) image_roots.push_back(std::move(canonical));
    }
}

lighter::Task<http::ServerResponse> Api::handle(http::ServerRequest request) {
    const auto &path = request.path;
    const bool get = request.method == http::method::get;
    const bool post = request.method == http::method::post;

    // Reads an input's image: a file under an allowed root, or base64 data.
    auto image_of = [&](const InputField &field, std::string &out) -> std::optional<std::string> {
        if (!field.image) return std::nullopt;
        const auto &image = *field.image;
        if (image.data) {
            std::string_view data = *image.data;
            if (data.starts_with("data:")) {
                const auto comma = data.find(',');
                if (comma == std::string_view::npos) return "a data URL without a comma";
                data.remove_prefix(comma + 1);
            }
            auto bytes = base64_decode(data);
            if (!bytes) return "image data is not base64";
            out = std::move(*bytes);
        } else if (image.path) {
            std::error_code ec;
            const auto canonical = std::filesystem::weakly_canonical(expand_home(*image.path), ec);
            const bool allowed = !ec && std::ranges::any_of(image_roots, [&](const auto &root) {
                const auto relative = canonical.lexically_relative(root);
                return !relative.empty() && *relative.begin() != "..";
            });
            if (!allowed) return "image path is outside the allowed roots: " + *image.path;
            const auto size = std::filesystem::file_size(canonical, ec);
            if (ec) return "cannot read " + *image.path;
            if (size > config->max_image_bytes) return "image file is too large: " + *image.path;
            std::ifstream stream(canonical, std::ios::binary);
            std::stringstream buffer;
            buffer << stream.rdbuf();
            out = std::move(buffer).str();
        } else {
            return "an image needs \"path\" or \"data\"";
        }
        if (out.size() > config->max_image_bytes) return "image is too large";
        return std::nullopt;
    };

    auto input_of = [&](InputValue &value, Input &out) -> std::optional<std::string> {
        if (auto *text = std::get_if<std::string>(&value)) {
            out.text = std::move(*text);
            return std::nullopt;
        }
        auto &field = std::get<InputField>(value);
        out.text = std::move(field.text);
        if (field.image) {
            std::string bytes;
            if (auto problem = image_of(field, bytes)) return problem;
            out.image = std::move(bytes);
        }
        if (!out.text && !out.image) return "an input needs \"text\" or \"image\"";
        return std::nullopt;
    };

    if (get && path == "/health") {
        co_return http::ServerResponse::json(200, R"({"status":"ok"})");
    }
    if (get && (path == "/v1/models" || path == "/v3/models")) {
        ModelList list;
        for (const auto *model : {config->embedding ? &*config->embedding : nullptr, config->reranker ? &*config->reranker : nullptr}) {
            if (!model) continue;
            list.data.push_back({.id = model->name});
            for (const auto &alias : model->aliases) list.data.push_back({.id = alias});
        }
        co_return json(200, list);
    }
    if (get && path == "/metrics") {
        Metrics metrics{.rss_bytes = resident_bytes()};
        for (auto [worker, model] : {std::pair{embedder, &config->embedding}, std::pair{reranker, &config->reranker}}) {
            if (!worker) continue;
            const auto stats = worker->stats();
            metrics.workers.push_back({.model = (*model)->name,
                                       .jobs = stats.jobs,
                                       .sequences = stats.sequences,
                                       .batches = stats.batches,
                                       .tokens = stats.tokens,
                                       .busy_seconds = stats.busy_seconds,
                                       .queued = stats.queued});
        }
        co_return json(200, metrics);
    }

    if (post && (path == "/v3/embeddings" || path == "/v1/embeddings")) {
        EmbeddingsBody body;
        if (auto ec = glz::read<k_read>(body, request.body))
            co_return error(400, "invalid request: " + glz::format_error(ec, request.body));
        if (!embedder || !serves(config->embedding, body.model)) co_return error(404, "no such embedding model: " + body.model);
        if (body.encoding_format && *body.encoding_format != "float") co_return error(400, "only encoding_format \"float\" is supported");

        std::vector<InputValue> values;
        if (auto *one = std::get_if<std::string>(&body.input)) {
            values.emplace_back(std::move(*one));
        } else {
            values = std::move(std::get<std::vector<InputValue>>(body.input));
        }
        if (values.empty()) co_return error(400, "input is empty");
        if (values.size() > config->max_inputs) co_return error(400, "too many inputs");
        const auto dimensions = body.dimensions.value_or(0);
        if (dimensions > embedder->hidden()) co_return error(400, "dimensions exceeds the model's");

        EmbedJob job{.instruction = body.instruction.value_or(""), .dimensions = dimensions};
        job.bulk = body.priority ? *body.priority == "bulk" : values.size() > 4;
        for (usize i = 0; i < values.size(); ++i) {
            Input input;
            if (auto problem = input_of(values[i], input)) co_return error(400, "input " + std::to_string(i) + ": " + *problem);
            job.inputs.push_back(std::move(input));
        }

        auto result = co_await embedder->embed(std::move(job));
        if (!result) co_return error(result.error().status, result.error().message);
        EmbeddingsReply reply{.model = config->embedding->name, .usage = {.prompt_tokens = result->tokens, .total_tokens = result->tokens}};
        for (usize i = 0; i < result->vectors.size(); ++i) reply.data.push_back({.index = i, .embedding = std::move(result->vectors[i])});
        co_return json(200, reply);
    }

    if (post && (path == "/v3/rerank" || path == "/v1/rerank")) {
        RerankBody body;
        if (auto ec = glz::read<k_read>(body, request.body))
            co_return error(400, "invalid request: " + glz::format_error(ec, request.body));
        if (!reranker || !serves(config->reranker, body.model)) co_return error(404, "no such reranker: " + body.model);
        if (body.documents.empty()) co_return error(400, "documents is empty");
        if (body.documents.size() > config->max_inputs) co_return error(400, "too many documents");

        RerankJob job{.instruction = body.instruction.value_or("")};
        job.bulk = body.priority && *body.priority == "bulk";
        if (auto problem = input_of(body.query, job.query)) co_return error(400, "query: " + *problem);
        for (usize i = 0; i < body.documents.size(); ++i) {
            Input input;
            if (auto problem = input_of(body.documents[i], input)) co_return error(400, "document " + std::to_string(i) + ": " + *problem);
            job.documents.push_back(std::move(input));
        }

        auto result = co_await reranker->rerank(std::move(job));
        if (!result) co_return error(result.error().status, result.error().message);
        RerankReply reply{.model = config->reranker->name, .usage = {.prompt_tokens = result->tokens, .total_tokens = result->tokens}};
        for (usize i = 0; i < result->scores.size(); ++i) reply.results.push_back({.index = i, .relevance_score = result->scores[i]});
        std::ranges::stable_sort(reply.results, [](const auto &a, const auto &b) { return a.relevance_score > b.relevance_score; });
        if (body.top_n && *body.top_n < reply.results.size()) reply.results.resize(*body.top_n);
        co_return json(200, reply);
    }

    const bool known =
        path == "/health" || path == "/metrics" || path.ends_with("/models") || path.ends_with("/embeddings") || path.ends_with("/rerank");
    co_return known ? error(405, "method not allowed") : error(404, "no such endpoint: " + path);
}

} // namespace qvl::service
