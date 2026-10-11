"""Export a Qwen3-VL embedding or reranker model to OpenVINO IR for qvl.

    pixi run -e tools python tools/export.py embedding --out ~/services/qvl/models
    pixi run -e tools python tools/export.py reranker --out ~/services/qvl/models

Writes <out>/<name>/ with the IR files optimum produces for `feature-extraction`
(the language model outputs the last hidden state; there is no lm_head), the
tokenizer and config files, and manifest.json (source revision, tool versions,
sha256 of every file). For the reranker it also writes rerank_head.f32: the
2048 floats of lm_head["yes"] - lm_head["no"], so qvl scores a pair as
sigmoid(h_last . head) like Qwen's reference reranker.
"""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import pathlib
import shutil
import subprocess
import sys

MODELS = {
    "embedding": "Qwen/Qwen3-VL-Embedding-2B",
    "reranker": "Qwen/Qwen3-VL-Reranker-2B",
}
TOOLS = ["optimum-intel", "optimum", "transformers", "torch", "openvino", "nncf"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rerank_head(snapshot, out):
    """lm_head["yes"] - lm_head["no"]; the reranker ties lm_head to embed_tokens."""
    import torch
    from safetensors import safe_open
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    vocab = tokenizer.get_vocab()
    yes, no = vocab["yes"], vocab["no"]
    weight = None
    for file in sorted(pathlib.Path(snapshot).glob("*.safetensors")):
        with safe_open(file, framework="pt") as tensors:
            for name in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
                if name in tensors.keys():
                    weight = tensors.get_tensor(name)
                    break
        if weight is not None:
            break
    if weight is None:
        sys.exit("export: no lm_head or embed_tokens weight in the snapshot")
    head = (weight[yes].to(torch.float32) - weight[no].to(torch.float32)).numpy()
    head.astype("<f4").tofile(out / "rerank_head.f32")
    return {"yes": yes, "no": no, "dim": int(head.shape[0])}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="models root")
    parser.add_argument("--revision", default="main", help="Hugging Face revision")
    parser.add_argument("--weight-format", default="int8", choices=["int8", "fp16"])
    parser.add_argument(
        "--sym", action="store_true", help="symmetric int8 (for the NPU)"
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    repo = MODELS[args.model]
    snapshot = snapshot_download(repo, revision=args.revision)
    revision = pathlib.Path(snapshot).name
    suffix = args.weight_format + ("-sym" if args.sym else "")
    name = f"{repo.split('/')[1]}-{suffix}"
    out = args.out.expanduser() / name
    staging = out.with_name(name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)

    command = [
        sys.executable,
        "-m",
        "optimum.commands.optimum_cli",
        "export",
        "openvino",
        "--model",
        snapshot,
        "--task",
        "feature-extraction",
        "--weight-format",
        args.weight_format,
        str(staging),
    ]
    if args.sym:
        command.insert(-1, "--sym")
    print("export:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    for extra in (
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
    ):
        if (pathlib.Path(snapshot) / extra).exists() and not (staging / extra).exists():
            shutil.copy2(pathlib.Path(snapshot) / extra, staging / extra)
    manifest = {
        "model": args.model,
        "repo": repo,
        "revision": revision,
        "weight_format": suffix,
        "created": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "tools": {tool: importlib.metadata.version(tool) for tool in TOOLS},
    }
    if args.model == "reranker":
        manifest["rerank_head"] = rerank_head(snapshot, staging)
    manifest["files"] = {
        file.name: sha256(file) for file in sorted(staging.iterdir()) if file.is_file()
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    print(f"export: wrote {out}")


if __name__ == "__main__":
    main()
