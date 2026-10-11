#include "worker.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <format>
#include <functional>
#include <mutex>
#include <numeric>
#include <span>
#include <thread>
#include <variant>

#include <qvl/image/image.h>
#include <qvl/prompt/prompt.h>

namespace qvl::service {

namespace {

using EmbedSender = lighter::Oneshot<Result<Embeddings>>::Sender;
using RerankSender = lighter::Oneshot<Result<Scores>>::Sender;

/// A job on the worker: its request, where its answer goes and, once
/// prepared, its sequences and their hidden states.
struct Pending {
    std::variant<EmbedJob, RerankJob> job;
    std::variant<EmbedSender, RerankSender> sender;
    std::vector<engine::Sequence> sequences;
    /// Shared by every sequence, which then holds only its remainder (rerank).
    std::optional<engine::Sequence> prefix;
    std::vector<float> hidden;
    usize remaining = 0;
    usize tokens = 0;
    bool failed = false;

    bool bulk() const {
        return std::visit([](const auto &j) { return j.bulk; }, job);
    }

    void fail(Failure failure) {
        if (failed) return;
        failed = true;
        std::visit(
            [&](auto &s) {
                using Sender = std::remove_cvref_t<decltype(s)>;
                if constexpr (std::is_same_v<Sender, EmbedSender>) {
                    std::move(s).send(Result<Embeddings>(std::unexpected(std::move(failure))));
                } else {
                    std::move(s).send(Result<Scores>(std::unexpected(std::move(failure))));
                }
            },
            sender);
    }
};

struct Batch {
    std::shared_ptr<Pending> job;
    std::vector<usize> indices;
};

/// A picture's cache key: two independent 64-bit hashes of its bytes and its size.
std::string image_key(std::string_view bytes) {
    u64 fnv = 14695981039346656037ull;
    for (unsigned char c : bytes) fnv = (fnv ^ c) * 1099511628211ull;
    return std::format("{:016x}{:016x}-{}", fnv, std::hash<std::string_view>{}(bytes), bytes.size());
}

/// Cuts `text` back until render(text) tokenizes within `limit`.
std::expected<std::vector<i32>, std::string>
fit(const tokenizer::Tokenizer &tokenizer, const std::function<std::string(std::string_view)> &render, std::string_view text, u32 limit) {
    auto ids = tokenizer.encode(render(text));
    for (int round = 0; ids.size() > limit && round < 12 && !text.empty(); ++round) {
        const double keep = std::max(0.0, double(limit) / double(ids.size()) * 0.97);
        usize cut = static_cast<usize>(double(text.size()) * keep);
        while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut; // whole UTF-8 sequences
        text = text.substr(0, cut);
        ids = tokenizer.encode(render(text));
    }
    if (ids.size() > limit) return std::unexpected(std::format("the input needs {} tokens; the limit is {}", ids.size(), limit));
    return ids;
}

} // namespace

struct Worker::Self {
    engine::Engine engine;
    mutable std::mutex mutex;
    std::condition_variable wake;
    std::deque<std::shared_ptr<Pending>> incoming;
    bool stopping = false;
    WorkerStats stats;
    std::thread thread;

    explicit Self(engine::Engine engine) : engine(std::move(engine)) {}

    void submit(std::shared_ptr<Pending> pending) {
        {
            std::lock_guard lock(mutex);
            incoming.push_back(std::move(pending));
            ++stats.queued;
        }
        wake.notify_one();
    }

    /// A picture for the engine: its cached encoder output when the same bytes
    /// were seen before, otherwise its decoded patches.
    Result<engine::Image> picture(const std::string &bytes) {
        engine::Image out{.key = image_key(bytes)};
        out.encoded = engine.lookup(out.key);
        if (out.encoded) return out;
        auto decoded = image::decode(std::as_bytes(std::span(bytes)));
        if (!decoded) return std::unexpected(Failure{400, decoded.error()});
        out.patches = std::make_shared<const image::Patches>(image::patchify(image::letterbox(*decoded, engine.options().image_size)));
        return out;
    }

    u32 image_tokens() const {
        const u32 side = engine.options().image_size / (image::k_patch_size * image::k_merge_size);
        return side * side;
    }

    /// Prompts, tokens and images for every input.
    std::optional<Failure> prepare(Pending &pending) {
        const auto &tokenizer = engine.tokenizer();
        const auto limit = engine.max_length();
        if (auto *job = std::get_if<EmbedJob>(&pending.job)) {
            for (usize i = 0; i < job->inputs.size(); ++i) {
                const auto &input = job->inputs[i];
                engine::Sequence sequence;
                if (input.image) {
                    auto image = picture(*input.image);
                    if (!image) return Failure{400, std::format("input {}: {}", i, image.error().message)};
                    sequence.images.push_back(std::move(*image));
                }
                const u32 tokens = input.image ? image_tokens() : 0;
                auto render = [&](std::string_view text) {
                    return prompt::embedding(
                        job->instruction, {.text = input.text ? std::optional(std::string(text)) : std::nullopt, .image_tokens = tokens});
                };
                auto ids = fit(tokenizer, render, input.text.value_or(""), limit);
                if (!ids) return Failure{400, std::format("input {}: {}", i, ids.error())};
                sequence.ids = std::move(*ids);
                pending.sequences.push_back(std::move(sequence));
            }
            return std::nullopt;
        }

        auto &job = std::get<RerankJob>(pending.job);
        std::optional<engine::Image> query_image;
        if (job.query.image) {
            auto image = picture(*job.query.image);
            if (!image) return Failure{400, "query: " + image.error().message};
            query_image = std::move(*image);
        }
        const prompt::Content query{.text = job.query.text, .image_tokens = query_image ? image_tokens() : 0};
        for (usize i = 0; i < job.documents.size(); ++i) {
            const auto &document = job.documents[i];
            engine::Sequence sequence;
            if (query_image) sequence.images.push_back(*query_image);
            if (document.image) {
                auto image = picture(*document.image);
                if (!image) return Failure{400, std::format("document {}: {}", i, image.error().message)};
                sequence.images.push_back(std::move(*image));
            }
            const u32 tokens = document.image ? image_tokens() : 0;
            auto render = [&](std::string_view text) {
                return prompt::rerank(job.instruction, query,
                                      {.text = document.text ? std::optional(std::string(text)) : std::nullopt, .image_tokens = tokens});
            };
            auto ids = fit(tokenizer, render, document.text.value_or(""), limit);
            if (!ids) return Failure{400, std::format("document {}: {}", i, ids.error())};
            sequence.ids = std::move(*ids);
            pending.sequences.push_back(std::move(sequence));
        }
        // The system turn, instruction and query are the same in every pair:
        // run them once and every document from their KV cache.
        if (engine.prefix_cache()) {
            if (auto split = engine::split_prefix(pending.sequences, query_image ? 1 : 0, engine.image_token())) {
                pending.tokens += split->prefix.ids.size() * pending.sequences.size();
                pending.prefix = std::move(split->prefix);
                pending.sequences = std::move(split->rest);
            }
        }
        return std::nullopt;
    }

    /// Longest first; sequences share a batch up to the bucket's limit.
    std::vector<Batch> batches(const std::shared_ptr<Pending> &pending) const {
        std::vector<usize> order(pending->sequences.size());
        std::iota(order.begin(), order.end(), 0);
        std::ranges::stable_sort(order,
                                 [&](usize a, usize b) { return pending->sequences[a].ids.size() > pending->sequences[b].ids.size(); });
        std::vector<Batch> out;
        for (usize at = 0; at < order.size();) {
            const auto &first = pending->sequences[order[at]];
            Batch batch{.job = pending, .indices = {order[at++]}};
            const u32 capacity = engine.max_batch(static_cast<u32>(first.ids.size()));
            while (at < order.size() && batch.indices.size() < capacity) batch.indices.push_back(order[at++]);
            out.push_back(std::move(batch));
        }
        return out;
    }

    void finish(Pending &pending) {
        const auto hidden = engine.hidden();
        const usize n = pending.sequences.size();
        if (auto *job = std::get_if<EmbedJob>(&pending.job)) {
            Embeddings out{.tokens = pending.tokens};
            const usize dims = job->dimensions == 0 ? hidden : std::min<usize>(job->dimensions, hidden);
            for (usize i = 0; i < n; ++i) {
                std::vector<float> vector(pending.hidden.begin() + static_cast<std::ptrdiff_t>(i * hidden),
                                          pending.hidden.begin() + static_cast<std::ptrdiff_t>(i * hidden + dims));
                double norm = 0;
                for (float v : vector) norm += double(v) * v;
                const double scale = norm > 0 ? 1 / std::sqrt(norm) : 0;
                for (auto &v : vector) v = static_cast<float>(v * scale);
                out.vectors.push_back(std::move(vector));
            }
            std::move(std::get<EmbedSender>(pending.sender)).send(std::move(out));
            return;
        }
        const auto head = engine.rerank_head();
        Scores out{.tokens = pending.tokens};
        for (usize i = 0; i < n; ++i) {
            double logit = 0;
            for (usize d = 0; d < hidden; ++d) logit += double(pending.hidden[i * hidden + d]) * head[d];
            out.scores.push_back(static_cast<float>(1 / (1 + std::exp(-logit))));
        }
        std::move(std::get<RerankSender>(pending.sender)).send(std::move(out));
    }

    void run() {
        std::deque<Batch> interactive;
        std::deque<Batch> bulk;
        for (;;) {
            std::deque<std::shared_ptr<Pending>> arrived;
            {
                std::unique_lock lock(mutex);
                wake.wait(lock, [&] { return stopping || !incoming.empty() || !interactive.empty() || !bulk.empty(); });
                if (stopping) break;
                arrived.swap(incoming);
            }

            for (auto &pending : arrived) {
                if (auto failure = prepare(*pending)) {
                    pending->fail(std::move(*failure));
                    std::lock_guard lock(mutex);
                    --stats.queued;
                    continue;
                }
                pending->hidden.resize(pending->sequences.size() * engine.hidden());
                pending->remaining = pending->sequences.size();
                for (const auto &sequence : pending->sequences) pending->tokens += sequence.ids.size();
                if (pending->remaining == 0) {
                    finish(*pending);
                    std::lock_guard lock(mutex);
                    --stats.queued;
                    continue;
                }
                for (auto &batch : batches(pending)) (pending->bulk() ? bulk : interactive).push_back(std::move(batch));
            }
            if (interactive.empty() && bulk.empty()) continue;

            auto &queue = interactive.empty() ? bulk : interactive;
            auto batch = std::move(queue.front());
            queue.pop_front();
            auto &pending = *batch.job;
            if (pending.failed) continue;

            std::vector<engine::Sequence> sequences;
            for (auto index : batch.indices) sequences.push_back(std::move(pending.sequences[index]));
            const auto started = std::chrono::steady_clock::now();
            auto hidden = pending.prefix ? engine.last_hidden_after(*pending.prefix, sequences) : engine.last_hidden(sequences);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            usize tokens = 0;
            for (usize k = 0; k < sequences.size(); ++k) {
                tokens += sequences[k].ids.size();
                pending.sequences[batch.indices[k]].ids = std::move(sequences[k].ids); // kept for the count in finish()
            }
            {
                std::lock_guard lock(mutex);
                ++stats.batches;
                stats.sequences += batch.indices.size();
                stats.tokens += tokens;
                stats.busy_seconds += seconds;
                stats.engine = engine.timings();
            }
            if (!hidden) {
                pending.fail({500, hidden.error()});
                std::lock_guard lock(mutex);
                --stats.queued;
                continue;
            }
            const auto width = engine.hidden();
            for (usize k = 0; k < batch.indices.size(); ++k) {
                std::copy_n(hidden->begin() + static_cast<std::ptrdiff_t>(k * width), width,
                            pending.hidden.begin() + static_cast<std::ptrdiff_t>(batch.indices[k] * width));
            }
            pending.remaining -= batch.indices.size();
            if (pending.remaining == 0) {
                finish(pending);
                std::lock_guard lock(mutex);
                ++stats.jobs;
                --stats.queued;
            }
        }

        // Stopping: whatever is left gets an answer rather than silence.
        for (auto *queue : {&interactive, &bulk}) {
            for (auto &batch : *queue) batch.job->fail({503, "the service is stopping"});
        }
        std::lock_guard lock(mutex);
        for (auto &pending : incoming) pending->fail({503, "the service is stopping"});
        incoming.clear();
    }
};

Worker::Worker(engine::Engine engine) : self(std::make_unique<Self>(std::move(engine))) {
    self->thread = std::thread([self = self.get()] { self->run(); });
}

Worker::~Worker() {
    {
        std::lock_guard lock(self->mutex);
        self->stopping = true;
    }
    self->wake.notify_one();
    self->thread.join();
}

lighter::Task<Result<Embeddings>> Worker::embed(EmbedJob job) {
    lighter::Oneshot<Result<Embeddings>> done;
    auto pending = std::make_shared<Pending>(Pending{.job = std::move(job), .sender = done.sender()});
    self->submit(std::move(pending));
    auto answer = co_await done.wait();
    if (!answer) co_return std::unexpected(Failure{503, "the worker dropped the job"});
    co_return std::move(*answer);
}

lighter::Task<Result<Scores>> Worker::rerank(RerankJob job) {
    lighter::Oneshot<Result<Scores>> done;
    auto pending = std::make_shared<Pending>(Pending{.job = std::move(job), .sender = done.sender()});
    self->submit(std::move(pending));
    auto answer = co_await done.wait();
    if (!answer) co_return std::unexpected(Failure{503, "the worker dropped the job"});
    co_return std::move(*answer);
}

engine::Kind Worker::engine_kind() const noexcept { return self->engine.kind(); }
usize Worker::hidden() const noexcept { return self->engine.hidden(); }

WorkerStats Worker::stats() const {
    std::lock_guard lock(self->mutex);
    return self->stats;
}

} // namespace qvl::service
