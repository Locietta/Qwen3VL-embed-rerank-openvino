"""Soak a running qvl-server with random shapes and watch its memory.

    python tools/soak.py --url http://127.0.0.1:8099 --unit qvl-soak --requests 2000 --images qvl/tests/fixtures

Sends embeddings (1-32 inputs of random length, sometimes images) and rerank
requests (1-20 documents). Every 100 requests it re-embeds a fixed probe and
records the systemd unit's cgroup memory: the probe must not drift (silent
corruption) and memory must level off after the first shapes. Standard library
only; synthetic text.
"""

import argparse
import base64
import json
import math
import pathlib
import random
import subprocess
import time
import urllib.request

WORDS = "群友 今天 卡池 又歪了 芒果 过敏 周末 电影 服务器 显卡 哈哈 笑死 绷不住 日语 考试 火锅 杭州 搬家 the quick brown fox jumps over lazy dog 2333".split()


def post(url, path, body):
    request = urllib.request.Request(
        url + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def text(rng):
    return "".join(rng.choice(WORDS) for _ in range(rng.randint(1, 120)))


def memory(unit):
    value = subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "MemoryCurrent", "--value"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(value) if value.isdigit() else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://127.0.0.1:8099")
    parser.add_argument(
        "--unit", required=True, help="systemd user unit running qvl-server"
    )
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument(
        "--images", type=pathlib.Path, help="directory of PNG files to send"
    )
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    images = (
        [
            base64.b64encode(p.read_bytes()).decode()
            for p in sorted(args.images.glob("*.png"))
        ]
        if args.images
        else []
    )

    probe_body = {"input": "群友E: 我对芒果过敏，吃一口就起疹子"}
    probe = post(args.url, "/v3/embeddings", probe_body)["data"][0]["embedding"]
    started = time.time()
    inputs = documents = 0
    for i in range(1, args.requests + 1):
        if rng.random() < 0.2:
            docs = [text(rng) for _ in range(rng.randint(1, 20))]
            post(args.url, "/v3/rerank", {"query": text(rng)[:60], "documents": docs})
            documents += len(docs)
        else:
            batch = []
            for _ in range(rng.choice([1, 1, 1, 2, 4, 8, 16, 32])):
                if images and rng.random() < 0.1:
                    batch.append(
                        {
                            "image": {"data": rng.choice(images)},
                            "text": text(rng)[:30] if rng.random() < 0.5 else None,
                        }
                    )
                else:
                    batch.append(text(rng))
            batch = [
                b
                if isinstance(b, str)
                else {k: v for k, v in b.items() if v is not None}
                for b in batch
            ]
            post(args.url, "/v3/embeddings", {"input": batch})
            inputs += len(batch)
        if i % 100 == 0:
            again = post(args.url, "/v3/embeddings", probe_body)["data"][0]["embedding"]
            cosine = sum(a * b for a, b in zip(probe, again))
            print(
                f"{i:5d} requests, {inputs} inputs, {documents} documents, {time.time() - started:6.0f}s: "
                f"memory {memory(args.unit) / 2**20:7.0f} MiB, probe cosine {cosine:.6f}",
                flush=True,
            )
            if not math.isclose(cosine, 1.0, abs_tol=1e-4):
                raise SystemExit("soak: the probe drifted")


if __name__ == "__main__":
    main()
