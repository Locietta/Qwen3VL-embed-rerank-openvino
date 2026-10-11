This is "qvl" (repository `Qwen3VL-embed-rerank-openvino`), a local inference service for Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B. It runs the OpenVINO C++ runtime directly on an Intel iGPU (later also the NPU) and serves `/v3/embeddings` and `/v3/rerank` over HTTP on the loopback. Its main client is lilia (sibling repository). Notes about a particular deployment stay out of the tree: if present, read the untracked `AGENTS.local.md`.

This project is built with C++26 and xmake, on the same toolchain and conventions as lilia and liminal. Check the coding style guide at `docs/dev/code-style-guide.md`; the architecture and the model details are in `docs/dev/design.md`.

It builds and runs on Linux only. Edit anywhere; build and test on the Linux host through the remote scripts. Compiler toolchain, OpenVINO, C dependencies and python packages are managed by pixi:

- `pixi run format <paths...>`: format given files
- `pixi run test-remote-linux --host <ssh-host>` (and `-sanitize`): build and run all tests on a Linux host
- `pixi run deploy-cachy [--host <ssh-host>] [--restart]`: build and install a release on the inference host
- On Linux: `pixi run configure`, `pixi run build`, `pixi run test-all`
- `pixi run -e tools python tools/<script>.py`: model export, reference implementations and fixtures (Linux)

C++ dependencies should be managed by xmake; OpenVINO and the Intel GPU runtime come from conda-forge through pixi. Headers must remain platform-neutral.

`lighter` (coroutine runtime, HTTP client and server) is an xmake package from `Locietta/lighter`, pinned through the recipe in the `xmake` submodule (`Locietta/xmake-repo`). Change lighter in its own repository. Code ported from transformers, optimum-intel and Qwen's reference implementations is listed in `docs/dev/provenance.md`.

Test fixtures must be synthetic: never commit real chat messages or stickers from lilia's database.

Commit headlines follow `<type>(<scope>): <subject>`.
