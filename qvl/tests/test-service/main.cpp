#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <format>
#include <memory>
#include <string>

#include <glaze/json.hpp>

#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/task.h>
#include <lighter/async/runtime/when.h>
#include <lighter/http/http.h>
#include <lighter/http/server.h>
#include <qvl/service/api.h>
#include <qvl/tests/support/fixtures.h>

namespace {

using namespace qvl::test;
namespace service = qvl::service;
namespace http = lighter::http;

std::string base64(std::string_view bytes) {
    static constexpr char table[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    u32 buffer = 0;
    int bits = 0;
    for (unsigned char c : bytes) {
        buffer = (buffer << 8) | c;
        bits += 8;
        while (bits >= 6) {
            bits -= 6;
            out += table[(buffer >> bits) & 63];
        }
    }
    if (bits > 0) out += table[(buffer << (6 - bits)) & 63];
    while (out.size() % 4) out += '=';
    return out;
}

struct Reply {
    i32 status = 0;
    glz::generic body;
};

lighter::Task<Reply> post(http::Client &client, i32 port, std::string_view path, std::string body) {
    auto request = client.on().post(std::format("http://127.0.0.1:{}{}", port, path));
    request.json_text(std::move(body));
    request.timeout(std::chrono::seconds(60));
    auto response = co_await std::move(request).send();
    require(response.has_value(), response ? "" : response.error().message());
    Reply out{.status = response->status};
    const auto text = response->text();
    require(!glz::read<glz::opts{.null_terminated = false}>(out.body, text), "unparseable reply: " + std::string(text));
    co_return out;
}

double cosine(const glz::generic &a, const glz::generic &b) {
    double out = 0;
    const auto &x = a.get_array();
    const auto &y = b.get_array();
    for (usize i = 0; i < x.size(); ++i) out += x[i].get_number() * y[i].get_number();
    return out;
}

lighter::Task<> checks(http::Server &server) {
    http::Client client;
    const auto port = server.port();
    const auto card = base64(read_file(fixture_dir() / "card-256.png"));

    // Texts with and without an instruction, and an image as data.
    auto embedded = co_await post(client, port, "/v3/embeddings",
                                  std::format(R"({{"model":"Qwen3-VL-Embedding-2B","input":["群友E: 我对芒果过敏，吃一口就起疹子",
                                      {{"text":"群友C: 这期卡池又歪了，两百抽一个金都没有"}},{{"image":{{"data":"{}"}}}}]}})",
                                              card));
    require(embedded.status == 200, "embeddings status " + std::to_string(embedded.status));
    const auto &data = embedded.body["data"].get_array();
    require(data.size() == 3 && data[0]["embedding"].get_array().size() == 2048, "three 2048-d embeddings");
    auto query = co_await post(
        client, port, "/v3/embeddings",
        R"({"input":"谁对什么东西过敏来着","instruction":"Given a question about a group chat, retrieve the chat messages that answer it","dimensions":1024})");
    require(query.status == 200, "query status");
    const auto &q = query.body["data"][0]["embedding"];
    require(q.get_array().size() == 1024, "dimensions truncates");
    double norm = 0;
    for (const auto &v : q.get_array()) norm += v.get_number() * v.get_number();
    require(std::abs(norm - 1) < 1e-4, "truncated vectors are renormalized");
    std::printf("embeddings ok; usage %g tokens\n", embedded.body["usage"]["prompt_tokens"].get_number());

    // The same image by path matches the data upload.
    auto by_path = co_await post(client, port, "/v1/embeddings",
                                 std::format(R"({{"input":[{{"image":{{"path":"{}"}}}}]}})", (fixture_dir() / "card-256.png").string()));
    require(by_path.status == 200, "image by path");
    const double same = cosine(by_path.body["data"][0]["embedding"], data[2]["embedding"]);
    require(same > 0.9999, std::format("path and data images differ: {}", same));

    // Rerank: the matching document first, top_n respected.
    auto ranked = co_await post(
        client, port, "/v3/rerank",
        R"({"query":"芒果过敏","documents":["这期卡池又歪了","群友E: 我对芒果过敏，吃一口就起疹子","周末去看电影"],"top_n":2})");
    require(ranked.status == 200, "rerank status");
    const auto &results = ranked.body["results"].get_array();
    require(results.size() == 2 && results[0]["index"].get_number() == 1, "rerank puts the allergy line first");
    std::printf("rerank ok: top score %.3f\n", results[0]["relevance_score"].get_number());

    // Errors.
    require((co_await post(client, port, "/v3/embeddings", R"({"model":"nope","input":"x"})")).status == 404, "unknown model");
    require((co_await post(client, port, "/v3/embeddings", R"({"input":)")).status == 400, "bad JSON");
    require((co_await post(client, port, "/v3/embeddings", R"({"input":[{"image":{"path":"/etc/passwd"}}]})")).status == 400,
            "paths outside the roots are refused");
    require((co_await post(client, port, "/v3/embeddings", R"({"input":[{"image":{"data":"aGVsbG8="}}]})")).status == 400,
            "undecodable images are refused");
    std::printf("errors ok\n");
}

/// Runs the checks, stopping the server however they end, so serve() returns.
lighter::Task<> exercise(http::Server &server) {
    try {
        co_await checks(server);
    } catch (...) {
        server.stop();
        throw;
    }
    server.stop();
}

} // namespace

int main() {
    try {
        const auto embedding_dir = model_dir(k_embedding_model);
        if (!embedding_dir) return 0;
        service::Config config;
        config.image_roots = {fixture_dir().string()};
        auto load = [&](qvl::engine::Kind kind, std::string_view name) {
            auto engine = qvl::engine::Engine::load(
                kind, {.dir = *model_dir(name), .warm_up = false, .prefix_cache = kind == qvl::engine::Kind::RERANKER});
            require(engine.has_value(), engine ? "" : engine.error());
            return std::make_unique<service::Worker>(std::move(*engine));
        };
        auto embedder = load(qvl::engine::Kind::EMBEDDING, k_embedding_model);
        auto reranker = load(qvl::engine::Kind::RERANKER, k_reranker_model);
        service::Api api(config, embedder.get(), reranker.get());

        lighter::EventLoop loop;
        auto server = http::Server::listen("127.0.0.1", 0, {}, loop);
        require(server.has_value(), "listen");
        auto handler = [&](http::ServerRequest request) { return api.handle(std::move(request)); };
        // A named closure: the coroutine refers to it until it finishes.
        auto body = [&]() -> lighter::Task<> {
            auto served = co_await lighter::when_all(server->serve(handler), exercise(*server));
            require(served.has_value(), "serve failed");
        };
        auto task = body();
        loop.schedule(task);
        loop.run();
        std::ignore = task.value();
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
