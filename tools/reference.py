"""Reference Qwen3-VL embedder and reranker for qvl's fixtures and parity checks.

They follow Qwen's reference code (QwenLM/Qwen3-VL-Embedding, src/models/
qwen3_vl_embedding.py and qwen3_vl_reranker.py): the same conversations, chat
template and last-token pooling. Two backends:

- "ov": the exported IR through optimum-intel on CPU, with dynamic quantization
  off (the int8 vision merger is numerically unstable with it on). This runs
  the same IR as qvl and should match it to cosine ~1.
- "torch": the original weights in float32, the ground truth.

An item is {"text": str | None, "image": PIL.Image | None}. Images must
already have sides that are multiples of 32 (qvl letterboxes before this
point), so the processor's resize is the identity on both sides.
"""

import pathlib
import unicodedata

import numpy as np
import torch
from qwen_vl_utils.vision_process import process_vision_info
from transformers import AutoProcessor

EMBEDDING_INSTRUCTION = "Represent the user's input."
RERANK_INSTRUCTION = (
    "Given a search query, retrieve relevant candidates that answer the query."
)
RERANK_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)
MIN_PIXELS = 4 * 32 * 32
MAX_PIXELS = 1800 * 32 * 32
OV_CONFIG = {"DYNAMIC_QUANTIZATION_GROUP_SIZE": "0", "INFERENCE_PRECISION_HINT": "f32"}


def punctuated(instruction):
    instruction = instruction.strip()
    if instruction and not unicodedata.category(instruction[-1]).startswith("P"):
        instruction += "."
    return instruction


def image_part(image):
    return {
        "type": "image",
        "image": image,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }


def embedding_conversation(item, instruction=None):
    content = []
    if item.get("image") is not None:
        content.append(image_part(item["image"]))
    if item.get("text") is not None:
        content.append({"type": "text", "text": item["text"]})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    system = punctuated(instruction) if instruction else EMBEDDING_INSTRUCTION
    return [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": content},
    ]


def rerank_side(item, prefix):
    content = [{"type": "text", "text": prefix}]
    if item.get("image") is None and item.get("text") is None:
        content.append({"type": "text", "text": "NULL"})
        return content
    if item.get("image") is not None:
        content.append(image_part(item["image"]))
    if item.get("text") is not None:
        content.append({"type": "text", "text": item["text"]})
    return content


def rerank_conversation(query, document, instruction=None):
    content = [
        {"type": "text", "text": "<Instruct>: " + (instruction or RERANK_INSTRUCTION)}
    ]
    content += rerank_side(query, "<Query>:")
    content += rerank_side(document, "\n<Document>:")
    return [
        {"role": "system", "content": [{"type": "text", "text": RERANK_SYSTEM}]},
        {"role": "user", "content": content},
    ]


class Backbone:
    """Qwen3-VL up to the last hidden state, on either backend."""

    def __init__(self, model_dir, backend="ov", repo=None):
        self.model_dir = pathlib.Path(model_dir)
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, padding_side="right"
        )
        self.backend = backend
        if backend == "ov":
            from optimum.intel import OVModelForVisualCausalLM

            self.model = OVModelForVisualCausalLM.from_pretrained(
                self.model_dir, device="CPU", ov_config=OV_CONFIG
            )
        else:
            from transformers import Qwen3VLModel

            self.model = Qwen3VLModel.from_pretrained(repo, dtype=torch.float32).eval()

    def inputs(self, conversations):
        text = self.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        images, _, _ = process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        return self.processor(
            text=text, images=images, padding=True, do_resize=False, return_tensors="pt"
        )

    @torch.no_grad()
    def last_hidden(self, conversations):
        """[batch, hidden] at each sequence's last real token."""
        inputs = self.inputs(conversations)
        if self.backend == "ov":
            self.model.rope_deltas = None  # otherwise computed once per sequence
            hidden = torch.as_tensor(
                np.asarray(self.model(**inputs).logits), dtype=torch.float32
            )
        else:
            hidden = self.model(**inputs).last_hidden_state.to(torch.float32)
        mask = inputs["attention_mask"]
        last = mask.shape[1] - 1 - mask.flip(dims=[1]).argmax(dim=1)
        return hidden[torch.arange(hidden.shape[0]), last].numpy()


class Embedder(Backbone):
    def embed(self, items, instruction=None):
        hidden = self.last_hidden(
            [embedding_conversation(item, instruction) for item in items]
        )
        return hidden / np.linalg.norm(hidden, axis=1, keepdims=True)


class Reranker(Backbone):
    def __init__(self, model_dir, backend="ov", repo=None):
        super().__init__(model_dir, backend, repo)
        self.head = np.fromfile(self.model_dir / "rerank_head.f32", dtype="<f4")

    def scores(self, query, documents, instruction=None):
        # One pair at a time, as the reference reranker does.
        out = []
        for document in documents:
            hidden = self.last_hidden(
                [rerank_conversation(query, document, instruction)]
            )[0]
            out.append(float(1 / (1 + np.exp(-hidden @ self.head))))
        return out
