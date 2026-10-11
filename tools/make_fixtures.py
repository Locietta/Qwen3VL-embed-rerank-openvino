"""Write qvl's golden test data from the reference implementations.

    pixi run -e tools python tools/make_fixtures.py --models ~/services/qvl/models --out qvl/tests/fixtures

Everything here is synthetic: made-up sentences and procedurally drawn images,
never lilia's chats or stickers. Writes index.json (cases, token ids, prompts,
scores) plus raw little-endian arrays (*.bin) it describes, and the images.
"""

import argparse
import json
import pathlib
import types

import numpy as np
import torch
from PIL import Image, ImageDraw
from tokenizers import Tokenizer
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLVisionModel,
    Qwen3VLVisionRotaryEmbedding,
)

import reference

EMBEDDING = "Qwen3-VL-Embedding-2B-int8"
RERANKER = "Qwen3-VL-Reranker-2B-int8"
EMBEDDING_REPO = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_REPO = "Qwen/Qwen3-VL-Reranker-2B"
CHAT_TASK = (
    "Given a question about a group chat, retrieve the chat messages that answer it"
)

CORPUS = [
    "",
    "hello world",
    "Hello, World! How's it going? I'm fine; we'll see.",
    "我对芒果过敏，吃一口就起疹子",
    "群友A: 这期卡池又歪了，两百抽一个金都没有",
    "笑死 2333 hhh 绷不住了",
    "混合 English 和中文 mixed text，with punctuation！",
    "emoji 😂🤣👍🏽 and flags 🇨🇳 and zwj 👨‍👩‍👧",
    "  leading and trailing spaces   ",
    "tabs\tand\nnewlines\r\nand\n\n\nblank lines\n",
    "numbers 1234567890 12.5% -3e10 ０１２３",
    "full-width ＡＢＣ ｱｲｳ and half-width ｶﾀｶﾅ",
    "日本語のテキストとカタカナ、ひらがな。",
    "한국어 텍스트도 있습니다.",
    "Ünïcödé café vs café, ﬁ ligature, Å Å",
    "code: def f(x): return x**2  # comment\n    if x > 0: pass",
    "url https://example.com/path?q=1&r=2#frag and mail a.b@example.org",
    "special <|im_start|>user\nhi<|im_end|>\n and <|image_pad|><|image_pad|>",
    "not special <|im_start without close and <| image_pad |>",
    "'s 't 're 've 'm 'll 'd 'S 'LL don't I've",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "哈" * 40,
    "———…… 【括号】《书名》「引号」 ～",
    "​ zero width  nbsp 　ideographic space",
]


class Arrays:
    def __init__(self, out):
        self.out = out
        self.index = {}

    def add(self, name, array):
        array = np.ascontiguousarray(array)
        dtype = {np.dtype("float32"): "f32", np.dtype("int64"): "i64"}[array.dtype]
        file = f"{name}.bin"
        array.astype(array.dtype.newbyteorder("<")).tofile(self.out / file)
        self.index[name] = {"file": file, "dtype": dtype, "shape": list(array.shape)}


def draw_card(width, height, seed):
    """A deterministic picture: gradient, shapes and a few glyphs."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            (x * 255 // max(width - 1, 1)),
            (y * 255 // max(height - 1, 1)),
            ((x + y) * 127 // max(width + height - 2, 1)),
        ],
        -1,
    )
    image = Image.fromarray(base.astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(6):
        x0, y0 = int(rng.integers(0, width - 8)), int(rng.integers(0, height - 8))
        x1, y1 = (
            x0 + int(rng.integers(4, max(5, width // 2))),
            y0 + int(rng.integers(4, max(5, height // 2))),
        )
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        (draw.ellipse if rng.integers(0, 2) else draw.rectangle)(
            [x0, y0, x1, y1], fill=color
        )
    draw.text((4, 4), "qvl", fill=(255, 255, 255))
    return image


def images(out):
    made = {
        "card-256": draw_card(256, 256, 1),
        "wide-96x64": draw_card(96, 64, 2),
        "small-64": draw_card(64, 64, 3),
        "cat-256": draw_card(256, 256, 4),
    }
    for name, image in made.items():
        image.save(out / f"{name}.png")

    # Transparency and animation, for the decoder.
    rgba = Image.new("RGBA", (200, 120), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).ellipse([20, 10, 180, 110], fill=(200, 30, 60, 255))
    ImageDraw.Draw(rgba).rectangle([0, 0, 60, 40], fill=(10, 120, 250, 128))
    rgba.save(out / "rgba-200x120.png")
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, mask=rgba.getchannel("A"))
    first = Image.new("RGBA", (48, 32), (255, 0, 0, 255))
    second = Image.new("RGBA", (48, 32), (0, 255, 0, 255))
    first.save(
        out / "anim.gif", save_all=True, append_images=[second], duration=100, loop=0
    )
    return made, {"rgba-200x120": np.asarray(white, dtype=np.uint8)}


def vision_tables(arrays, grids):
    """fast_pos_embed_interpolate's indices and weights and rot_pos_emb, by
    calling transformers' own methods on stand-ins."""
    for h, w in grids:
        grid = torch.tensor([[1, h, w]])
        captured = {}

        class Corners:
            # One-hot per corner, so the weighted sum leaves the four weights.
            weight = torch.zeros(1)

            def __call__(self, idx):
                captured["idx"] = idx.clone()
                out = torch.zeros(4, idx.shape[1], 4)
                for corner in range(4):
                    out[corner, :, corner] = 1
                return out

        stand_in = types.SimpleNamespace(
            pos_embed=Corners(),
            num_grid_per_side=48,
            config=types.SimpleNamespace(spatial_merge_size=2),
            spatial_merge_size=2,
            rotary_pos_emb=Qwen3VLVisionRotaryEmbedding(32),
        )
        weights = Qwen3VLVisionModel.fast_pos_embed_interpolate(
            stand_in, grid
        )  # [N, 4], merge order
        idx = (
            captured["idx"]
            .view(4, 1, h // 2, 2, w // 2, 2)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(4, -1)
        )
        rotary = Qwen3VLVisionModel.rot_pos_emb(stand_in, grid)
        arrays.add(f"pos-idx-{h}x{w}", idx.numpy().astype(np.int64))
        arrays.add(
            f"pos-weight-{h}x{w}", weights.T.contiguous().numpy().astype(np.float32)
        )
        arrays.add(f"rotary-{h}x{w}", rotary.numpy().astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--skip-torch", action="store_true", help="only the OpenVINO reference"
    )
    args = parser.parse_args()
    models = args.models.expanduser()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    arrays = Arrays(out)
    index = {}

    # Tokenizers, exactly as the `tokenizers` library encodes.
    index["tokenizer"] = {}
    for name in (EMBEDDING, RERANKER):
        tokenizer = Tokenizer.from_file(str(models / name / "tokenizer.json"))
        index["tokenizer"][name] = [
            {"text": text, "ids": tokenizer.encode(text).ids} for text in CORPUS
        ]

    pictures, decoded = images(out)
    arrays.add("decoded-rgba-200x120", decoded["rgba-200x120"].astype(np.int64))

    embedder = reference.Embedder(models / EMBEDDING, "ov")
    processor = embedder.processor

    # Prompts as rendered by the chat template, image placeholders expanded.
    prompt_cases = [
        {"instruction": None, "text": "我对芒果过敏", "image": None},
        {"instruction": CHAT_TASK, "text": "谁对什么东西过敏来着", "image": None},
        {"instruction": "Retrieve stickers。", "text": None, "image": "wide-96x64"},
        {"instruction": None, "text": "配字", "image": "small-64"},
        {"instruction": None, "text": None, "image": None},
    ]
    index["embedding_prompts"] = []
    for case in prompt_cases:
        item = {
            "text": case["text"],
            "image": pictures[case["image"]] if case["image"] else None,
        }
        conversation = reference.embedding_conversation(item, case["instruction"])
        inputs = embedder.inputs([conversation])
        tokens = 0
        if case["image"]:
            thw = inputs["image_grid_thw"][0].tolist()
            tokens = thw[1] * thw[2] // 4
        text = processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False
        )[0]
        text = (
            text.replace("<|image_pad|>", "<|image_pad|>" * tokens, 1)
            if tokens
            else text
        )
        ids = inputs["input_ids"][0].tolist()
        index["embedding_prompts"].append(
            {**case, "image_tokens": tokens, "prompt": text, "ids": ids}
        )
        if case["image"]:
            config = embedder.model.config
            stand_in = types.SimpleNamespace(config=config)
            position, _ = Qwen3VLModel.get_rope_index(
                stand_in,
                inputs["input_ids"],
                inputs["image_grid_thw"],
                None,
                inputs["attention_mask"],
            )
            arrays.add(
                f"positions-{case['image']}", position[:, 0, :].numpy().astype(np.int64)
            )

    # pixel_values straight from the processor (no resize).
    for name in ("wide-96x64", "small-64"):
        inputs = embedder.inputs(
            [reference.embedding_conversation({"image": pictures[name]})]
        )
        arrays.add(f"pixels-{name}", inputs["pixel_values"].numpy().astype(np.float32))
        index.setdefault("grids", {})[name] = inputs["image_grid_thw"][0].tolist()

    vision_tables(arrays, [(16, 16), (4, 6), (4, 4), (18, 32)])

    # Embeddings: documents bare, queries with an instruction, images.
    embedding_cases = [
        {"text": "群友A: 周末去看了新上映的动画电影，作画是真的顶"},
        {"text": "群友E: 我对芒果过敏，吃一口就起疹子"},
        {"text": "群友C: 这期卡池又歪了，两百抽一个金都没有"},
        {"text": "谁对什么东西过敏来着", "instruction": CHAT_TASK},
        {"text": "抽卡运气很差的那个", "instruction": CHAT_TASK},
        {"image": "card-256"},
        {"image": "cat-256", "text": "一只猫"},
        {"image": "wide-96x64"},
    ]
    index["embeddings"] = embedding_cases
    rerank_cases = {
        "query": {"text": "抽卡运气很差的那个"},
        "documents": [
            {"text": "群友C: 这期卡池又歪了，两百抽一个金都没有"},
            {"text": "群友E: 我对芒果过敏，吃一口就起疹子"},
            {"text": "群友F: 这期卡池运气不错，十连就出了两个金"},
            {"image": "card-256"},
            {"image": "cat-256", "text": "一只猫"},
        ],
    }
    index["rerank"] = rerank_cases

    def items(cases):
        return [
            {
                "text": case.get("text"),
                "image": pictures[case["image"]] if case.get("image") else None,
            }
            for case in cases
        ]

    def embed_all(model):
        return np.concatenate(
            [
                model.embed(items([case]), case.get("instruction"))
                for case in embedding_cases
            ]
        )

    arrays.add("embeddings-ov", embed_all(embedder).astype(np.float32))
    del embedder
    reranker = reference.Reranker(models / RERANKER, "ov")
    query = items([rerank_cases["query"]])[0]
    index["rerank_scores_ov"] = reranker.scores(query, items(rerank_cases["documents"]))
    del reranker

    if not args.skip_torch:
        torch_embedder = reference.Embedder(models / EMBEDDING, "torch", EMBEDDING_REPO)
        arrays.add("embeddings-torch", embed_all(torch_embedder).astype(np.float32))
        del torch_embedder
        torch_reranker = reference.Reranker(models / RERANKER, "torch", RERANKER_REPO)
        index["rerank_scores_torch"] = torch_reranker.scores(
            query, items(rerank_cases["documents"])
        )

    index["arrays"] = arrays.index
    (out / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"make_fixtures: wrote {len(arrays.index)} arrays to {out}")


if __name__ == "__main__":
    main()
