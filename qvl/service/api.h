#pragma once

#include <filesystem>
#include <vector>

#include <lighter/async/runtime/task.h>
#include <lighter/http/server.h>
#include <qvl/service/config.h>
#include <qvl/service/worker.h>

namespace qvl::service {

/// The HTTP API over the workers:
///
/// - POST /v3/embeddings (also /v1/embeddings): OpenAI-style, plus "instruction"
///   (the query task, sent as the system turn), image inputs
///   ({"text": ..., "image": {"path" | "data": ...}}), "dimensions" (Matryoshka)
///   and "priority" ("interactive" | "bulk").
/// - POST /v3/rerank (also /v1/rerank): Cohere-style {query, documents, top_n},
///   where the query and each document may be text, an image or both.
/// - GET /health, GET /v1/models, GET /metrics.
struct Api {
    Api(const Config &config, Worker *embedder, Worker *reranker);

    lighter::Task<lighter::http::ServerResponse> handle(lighter::http::ServerRequest request);

private:
    const Config *config;
    Worker *embedder;
    Worker *reranker;
    std::vector<std::filesystem::path> image_roots;
};

} // namespace qvl::service
