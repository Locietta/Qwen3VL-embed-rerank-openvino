# qvl design

qvl serves two Qwen3-VL models with the OpenVINO C++ runtime: Qwen3-VL-Embedding-2B
(`/v3/embeddings`) and Qwen3-VL-Reranker-2B (`/v3/rerank`). It replaced an OVMS
container serving Qwen3-Embedding-0.6B for lilia on an Arrow Lake host (Arc iGPU
without XMX, 30 GB shared memory). Off-the-shelf serving did not work for these
models (2026-10): OpenVINO GenAI 2026.4.1's `EmbeddingPipeline` fails on text-only
input (it feeds the language model an empty deepstack tensor), OVMS does not
serve them, and the GPU plugin's memory grows with every new input shape. Under
memory pressure it once returned wrong vectors without an error.

## Shape

```text
client ──HTTP──► lighter::http::Server (event loop thread)
                   │ Api: JSON in/out, image files and base64
                   ▼
                 Worker (one thread per model) ── Engine ── OpenVINO (GPU)
                   ▲ answers through lighter::Oneshot
```

- **`tokenizer/`**: Qwen2 byte-level BPE configured from the model's
  `tokenizer.json` as the `tokenizers` library applies it: added tokens split
  out first, NFC (utf8proc), the Split pre-tokenizer regex (PCRE2, UTF + UCP),
  byte-level mapping, merges by rank (heap, leftmost on ties), then the
  post-processor template. The embedder appends `<|endoftext|>`; the reranker
  appends nothing. Checked token for token against `tokenizers` on the fixture
  corpus.
- **`image/`**: stb_image decode (first GIF frame; alpha over white, as Qwen's
  reference pastes RGBA), Catmull-Rom resize, a letterbox to a fixed square
  (default 256: 16x16 patches, 64 language-model tokens), `smart_resize`, and
  Qwen3-VL's patch layout (`[patches, 1536]`, frame duplicated, rows in 2x2
  merge-block order, `(p - 127.5) / 127.5`).
  - The letterbox scales small images up to fill the square; Qwen's reference
    resizes each image to its own size. One shape keeps the vision encoder
    static.
  - GIF/PNG transparency goes over white even for palette images; Pillow's
    `convert("RGB")` would show their palette colour.
- **`vision/`**: `fast_pos_embed_interpolate`'s four corner indices and weights
  (torch's float32 `linspace`, which counts the second half down from the end)
  and `rot_pos_emb`'s angles, both in merge-block order.
- **`prompt/`**: the embedder's chat prompt (instruction as the system turn,
  with a full stop added unless it ends in punctuation; image before text;
  `NULL` when empty), the reranker's (`<Instruct>: …<Query>:…\n<Document>:…`
  under a fixed judging system turn), and M-RoPE position ids
  (`get_rope_index`: image k at `(s, s + row, s + column)`, text after it from
  `s + max(rows, columns)`, padding 1).
- **`engine/`**: the five IR files of a model, exported by `tools/export.py`
  (optimum-intel `feature-extraction`, int8 weights).
  - The language model is stateful as exported (56 KV-cache variables). qvl
    replaces every ReadValue with its initializer and drops the Assigns: an
    embedding pass needs no cache, and the states would hold
    `[batch, 8, length, 128]` floats each.
  - A Gather on a new `last_index [batch, 1]` input makes the output
    `[batch, 1, hidden]`, so only each sequence's last token leaves the device.
  - The vision encoder (patch embed, then 24 blocks and the merger) is reshaped
    to the one letterbox shape and compiled statically. Static shapes also
    sidestep the GPU plugin's miscompile of the merger with a dynamic token
    dimension (openvino#37573). The position-table lookup depends only on the
    grid, so it runs once at load, on CPU.
  - `deepstack_visual_embeds` is `[3, image tokens, hidden]`, rows in the
    order the graph's cumsum reads `visual_pos_masks`. Text-only batches pass
    one zero row: the graph indexes row 0 even without images, and an empty
    tensor gives NaN.
  - Batches are right-padded to length buckets (64…2048) and batch buckets
    (1, 4, 16) under a token budget (8192). Padding rows hold one token so
    their attention is never empty. The model only ever sees these shapes, and
    `warm_up` compiles them all at load, which keeps GPU memory bounded.
  - Dynamic quantization is off (`DYNAMIC_QUANTIZATION_GROUP_SIZE=0`). With it
    on, the int8 vision merger changes its output by 2–13% for a 1e-6 change
    in its input.
- **`service/`**:
  - **Worker**: owns an Engine on a thread. A job is preprocessed there
    (decode, letterbox, prompt, tokens, truncation of the text to fit
    `max_length`), cut into batches (longest first; images alone), and queued.
    Interactive batches run before bulk ones (backfills), so a query waits at
    most one batch.
  - **Pooling**: embeddings are L2-normalized after optional Matryoshka
    truncation (`dimensions`); rerank scores are
    `sigmoid(h_last · (w_yes − w_no))` with `rerank_head.f32` from the export.
  - **Api**: request and response JSON (glaze). Image paths must resolve under
    `image_roots`.

## Accuracy (2026-10-11, fixtures from `tools/make_fixtures.py`)

- CPU against the OpenVINO reference (the same IR through optimum, dynamic
  quantization off): cosine 1.00000 for text and images; rerank scores agree to
  1e-4. This pins every host-side step.
- GPU against the same reference: text ≥ 0.9999, images 0.997–0.999, rerank
  within 0.01.
- Against the original PyTorch weights (float32), int8 costs text 0.997–0.999
  and images 0.97–0.98.

## API

See `service/api.h`. Embeddings follow OpenAI's shape with extras:
`instruction` (the query's task, as the system turn; documents go without),
`{"text", "image": {"path" | "data"}}` inputs, `dimensions`, `priority`
(`interactive` | `bulk`; by default more than 4 inputs is bulk). Rerank follows
Cohere's: `query`, `documents` (text, image or both), `top_n`, results sorted by
`relevance_score`.

## Deployment

A systemd user unit (`deploy/qvl.service`) runs the release binary from
`~/services/qvl/bin/current`, built on the host by `pixi run deploy-cachy`. The
binary links OpenVINO 2026.4, its GPU plugin and Intel's compute runtime
(OpenCL) from the deploy tree's Pixi environment. The host needs only the kernel
driver: no container, no sudo. Models are exported on the host with
`pixi run -e tools python tools/export.py embedding|reranker --out ~/services/qvl/models`.

## Next

- NPU experiments: the static vision encoder and the reranker are candidates;
  a symmetric int8 export (`--sym`) and Intel's NPU user-space driver would be
  needed.
