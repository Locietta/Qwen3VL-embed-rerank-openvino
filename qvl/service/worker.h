#pragma once

#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <lighter/async/runtime/oneshot.h>
#include <lighter/async/runtime/task.h>
#include <lighter/types.hpp>
#include <qvl/engine/engine.h>

namespace qvl::service {

using namespace lighter::types;

/// Text and/or an image (encoded file bytes).
struct Input {
    std::optional<std::string> text;
    std::optional<std::string> image;
};

/// Why a job failed: `status` is the HTTP status to answer with.
struct Failure {
    i32 status = 500;
    std::string message;
};

template <typename T>
using Result = std::expected<T, Failure>;

struct Embeddings {
    /// Normalized, one per input.
    std::vector<std::vector<float>> vectors;
    usize tokens = 0;
};

struct Scores {
    /// sigmoid(h . (w_yes - w_no)), one per document.
    std::vector<float> scores;
    usize tokens = 0;
};

struct EmbedJob {
    std::vector<Input> inputs;
    std::string instruction;
    /// Matryoshka truncation before normalizing; 0 keeps every dimension.
    u32 dimensions = 0;
    /// Bulk jobs (backfills) yield to interactive ones between batches.
    bool bulk = false;
};

struct RerankJob {
    Input query;
    std::vector<Input> documents;
    std::string instruction;
    bool bulk = false;
};

struct WorkerStats {
    u64 jobs = 0;
    u64 sequences = 0;
    u64 batches = 0;
    u64 tokens = 0;
    double busy_seconds = 0;
    usize queued = 0;
};

/// Owns an Engine and runs it on its own thread. Jobs are submitted from the
/// event loop and answered through a lighter::Oneshot, so awaiting one never
/// blocks the loop. Each job is preprocessed (prompt, tokens, images) on the
/// worker, cut into bucket-sized batches, and its batches queue behind those of
/// interactive jobs if it is bulk.
struct Worker {
    explicit Worker(engine::Engine engine);
    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;
    /// Finishes the batch in progress and fails the rest.
    ~Worker();

    lighter::Task<Result<Embeddings>> embed(EmbedJob job) pre(engine_kind() == engine::Kind::EMBEDDING);
    lighter::Task<Result<Scores>> rerank(RerankJob job) pre(engine_kind() == engine::Kind::RERANKER);

    engine::Kind engine_kind() const noexcept;
    usize hidden() const noexcept;
    WorkerStats stats() const;

    struct Self;

private:
    std::unique_ptr<Self> self;
};

} // namespace qvl::service
