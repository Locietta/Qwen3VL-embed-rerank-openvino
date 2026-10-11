# Code taken from elsewhere

## Ported logic (Apache-2.0)

Re-implemented in C++ from these sources, checked against them through the
fixtures (`tools/make_fixtures.py` calls the originals):

- transformers 5.0.0, `models/qwen3_vl/modeling_qwen3_vl.py`:
  `Qwen3VLModel.get_rope_index` → `qvl/prompt/prompt.cpp` (`position_ids`);
  `Qwen3VLVisionModel.fast_pos_embed_interpolate` and `rot_pos_emb` →
  `qvl/vision/vision.cpp`. `models/qwen2_vl/image_processing_qwen2_vl*`
  (patch layout, normalization) → `qvl/image/image.cpp` (`patchify`).
- qwen-vl-utils 0.0.14, `vision_process.smart_resize` → `qvl/image/image.cpp`.
- optimum-intel 2.2.0, `openvino/modeling_visual_language.py`
  (`_OVQwen3VLForCausalLM`): how the exported IR is driven (vision embeddings,
  merger inputs, deepstack placement) → `qvl/engine/engine.cpp`.
- QwenLM/Qwen3-VL-Embedding, `src/models/qwen3_vl_embedding.py` and
  `qwen3_vl_reranker.py` (`main`, 2026-10): conversation formats, instruction
  punctuation, last-token pooling, the yes/no scoring head →
  `qvl/prompt/prompt.cpp`, `qvl/service/worker.cpp`; mirrored in Python by
  `tools/reference.py`.
- Hugging Face `tokenizers` (BPE model, Split and ByteLevel pre-tokenizers,
  TemplateProcessing) → `qvl/tokenizer/tokenizer.cpp`; GPT-2's
  `bytes_to_unicode` table.

## From lilia

The build scaffolding (root `xmake.lua`, `pixi.toml`, formatting, hooks,
remote test and deploy scripts, `qvl/xmake/rules/version.lua`) and
`docs/dev/code-style-guide.md` were adapted from lilia, which took them from
liminal. `tools/reference.py` grew out of lilia's untracked benchmark script
`vlembed.py`.

## Third-party libraries (built by xmake or taken from Pixi)

lighter (MIT), PCRE2 (BSD), utf8proc (MIT), stb_image and stb_image_resize2
(public domain / MIT), glaze (MIT, through lighter), OpenVINO (Apache-2.0),
Intel compute runtime (MIT).

The models: Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B, Apache-2.0.
