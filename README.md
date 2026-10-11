# Qwen3VL-embed-rerank-openvino

A small HTTP service that runs [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) and [Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) on an Intel iGPU with the OpenVINO C++ runtime. It owns the whole input pipeline (tokenizer, image preprocessing, Qwen3-VL position ids) so that it can keep the GPU to a few fixed input shapes.

- `POST /v3/embeddings`: text and image embeddings (OpenAI-style request, with an `instruction` field).
- `POST /v3/rerank`: Cohere-style reranking of text or image documents.

Linux only. See `AGENTS.md` for the build and `docs/dev/design.md` for how it works.

## License

Apache-2.0.
