#include <chrono>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <glaze/json.hpp>
#include <openvino/openvino.hpp>

#include <lighter/async/io/loop.h>
#include <lighter/async/runtime/interrupt.h>
#include <lighter/async/runtime/sync.h>
#include <lighter/async/runtime/task.h>
#include <lighter/async/runtime/when.h>
#include <lighter/http/server.h>
#include <qvl/service/api.h>
#include <qvl/service/config.h>
#include <qvl/service/worker.h>
#include <qvl/tokenizer/tokenizer.h>

#include <xmake/version/qvl.hpp>

namespace {

using namespace qvl;

void print_version() {
    std::printf("qvl-server %.*s (%.*s%s)\n", static_cast<int>(xmake::version::version.size()), xmake::version::version.data(),
                static_cast<int>(xmake::version::commit.size()), xmake::version::commit.data(), xmake::version::dirty ? "-dirty" : "");
}

int list_devices() {
    try {
        ov::Core core;
        std::printf("OpenVINO %s\n", ov::get_openvino_version().buildNumber);
        for (const auto &device : core.get_available_devices()) {
            std::printf("%s: %s\n", device.c_str(), core.get_property(device, ov::device::full_name).c_str());
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "qvl-server: %s\n", error.what());
        return 1;
    }
    return 0;
}

/// `--tokenize <tokenizer.json>`: one JSON string per stdin line in, its token
/// ids out (space-separated), for comparing with Hugging Face tokenizers.
int tokenize(const char *path) {
    auto tokenizer = tokenizer::Tokenizer::load(path);
    if (!tokenizer) {
        std::fprintf(stderr, "qvl-server: %s\n", tokenizer.error().c_str());
        return 1;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        std::string text;
        if (glz::read_json(text, line)) {
            std::fprintf(stderr, "qvl-server: not a JSON string: %s\n", line.c_str());
            return 1;
        }
        std::string out;
        for (const auto id : tokenizer->encode(text)) out += std::to_string(id) + ' ';
        std::puts(out.c_str());
    }
    return 0;
}

std::unique_ptr<service::Worker> start(engine::Kind kind, const service::Config &config, const std::optional<service::ModelConfig> &model) {
    if (!model) return nullptr;
    const auto started = std::chrono::steady_clock::now();
    auto loaded = engine::Engine::load(
        kind,
        {
            .dir = service::expand_home(config.models_dir) / model->name,
            .language_device = model->device,
            .vision_device = model->vision_device,
            .cache_dir = config.cache_dir.empty() ? std::filesystem::path() : service::expand_home(config.cache_dir),
            .length_buckets = config.length_buckets,
            .batch_buckets = config.batch_buckets,
            .max_batch_tokens = config.max_batch_tokens,
            .image_size = config.image_size,
            .dynamic_quantization_group = config.dynamic_quantization_group,
            .prefix_cache = kind == engine::Kind::RERANKER && config.rerank_prefix_cache,
            .image_cache_bytes = config.image_cache_bytes,
            .image_cache_dir = config.image_cache_dir.empty() ? std::filesystem::path() : service::expand_home(config.image_cache_dir),
            .image_cache_disk_bytes = config.image_cache_disk_bytes,
        });
    if (!loaded) {
        std::fprintf(stderr, "qvl-server: cannot load %s: %s\n", model->name.c_str(), loaded.error().c_str());
        std::exit(1);
    }
    std::printf("loaded %s on %s/%s in %.1fs\n", model->name.c_str(), model->device.c_str(), model->vision_device.c_str(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    std::fflush(stdout);
    return std::make_unique<service::Worker>(std::move(*loaded));
}

/// After the first SIGINT/SIGTERM, stops the server gracefully, then idles
/// until serve() returns and cancels it.
lighter::Task<> stop_on_signal(lighter::http::Server &server, lighter::InterruptSource &interrupts) {
    // The token's wait() cancels when the signal arrives; catch that and carry on.
    std::ignore = co_await interrupts.token().wait().catch_cancel();
    std::printf("stopping\n");
    std::fflush(stdout);
    server.stop();
    lighter::Event never;
    co_await never.wait();
}

int serve(const service::Config &config) {
    auto embedder = start(engine::Kind::EMBEDDING, config, config.embedding);
    auto reranker = start(engine::Kind::RERANKER, config, config.reranker);
    service::Api api(config, embedder.get(), reranker.get());

    lighter::EventLoop loop;
    auto interrupts = lighter::InterruptSource::create(loop);
    if (!interrupts) {
        std::fprintf(stderr, "qvl-server: cannot watch signals: %s\n", std::string(interrupts.error().message()).c_str());
        return 1;
    }
    auto server = lighter::http::Server::listen(config.host, config.port, {.max_body_bytes = config.max_body_bytes}, loop);
    if (!server) {
        std::fprintf(stderr, "qvl-server: cannot listen on %s:%d: %s\n", config.host.c_str(), config.port,
                     std::string(server.error().message()).c_str());
        return 1;
    }
    std::printf("listening on %s:%d\n", config.host.c_str(), server->port());
    std::fflush(stdout);

    auto run = [&]() -> lighter::Task<lighter::Error> {
        auto handler = [&](lighter::http::ServerRequest request) { return api.handle(std::move(request)); };
        auto served = co_await lighter::when_any(server->serve(handler), stop_on_signal(*server, *interrupts));
        if (served.has_error()) co_return served.error();
        co_return lighter::Error();
    };
    auto task = run();
    loop.schedule(task);
    loop.run();
    const auto failure = task.value();
    if (failure && *failure) {
        std::fprintf(stderr, "qvl-server: %s\n", std::string(failure->message()).c_str());
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char **argv) {
    std::optional<std::string> config_path;
    bool check = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument = argv[i];
        if (argument == "--version") {
            print_version();
            return 0;
        }
        if (argument == "--devices") return list_devices();
        if (argument == "--tokenize" && i + 1 < argc) return tokenize(argv[i + 1]);
        if (argument == "--check-config") {
            check = true;
        } else if (argument == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        } else {
            std::fprintf(
                stderr,
                "usage: qvl-server --config <config.json> [--check-config] | --version | --devices | --tokenize <tokenizer.json>\n");
            return 2;
        }
    }

    service::Config config;
    if (config_path) {
        auto loaded = service::load_config(*config_path);
        if (!loaded) {
            std::fprintf(stderr, "qvl-server: %s\n", loaded.error().c_str());
            return 1;
        }
        config = std::move(*loaded);
    }
    if (check) {
        std::printf("config ok\n");
        return 0;
    }
    print_version();
    return serve(config);
}
