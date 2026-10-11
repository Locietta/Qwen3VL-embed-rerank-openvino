#!/usr/bin/env python3
"""Synchronize this worktree to a remote Linux host and test it there."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


def git_output(repo: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", repo, *arguments])


def worktree_files(repo: Path) -> list[bytes]:
    root = git_output(
        repo,
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        ".",
        ":(exclude)xmake",
    ).split(b"\0")
    xmake = git_output(
        repo / "xmake",
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        ".",
    ).split(b"\0")
    files = [path for path in root if path]
    files.extend(b"xmake/" + path for path in xmake if path)
    return sorted(files)


def msys_path(path: Path) -> str:
    path = path.resolve()
    drive, tail = os.path.splitdrive(path)
    if not drive:
        raise RuntimeError(f"cannot convert path to MSYS2 form: {path}")
    return f"/{drive[0].lower()}{tail.replace('\\', '/')}"


def run(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def ssh_command(msys2: str, host: str, arguments: list[str]) -> list[str]:
    remote_command = " ".join(shlex.quote(argument) for argument in arguments)
    return [msys2, "ssh", "-o", "BatchMode=yes", host, remote_command]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally build and test this worktree on a remote Linux host."
    )
    parser.add_argument(
        "--mode",
        choices=("normal", "release", "sanitize"),
        default="normal",
        help="configuration to validate",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("QVL_REMOTE_LINUX_HOST"),
        help="MSYS2 SSH host alias (or set QVL_REMOTE_LINUX_HOST)",
    )
    parser.add_argument(
        "--remote-root",
        default=".cache/qvl-ci",
        help="path below the remote user's home directory",
    )
    parser.add_argument("--jobs", type=int, help="parallel build and test jobs")
    args = parser.parse_args()

    if not args.host:
        parser.error("--host or QVL_REMOTE_LINUX_HOST is required")
    if args.jobs is not None and args.jobs < 1:
        parser.error("--jobs must be positive")
    jobs = args.jobs or int(
        os.environ.get("QVL_REMOTE_LINUX_JOBS", "6" if args.mode == "sanitize" else "8")
    )
    if jobs < 1:
        parser.error("QVL_REMOTE_LINUX_JOBS must be positive")

    msys2 = shutil.which("msys2")
    if msys2 is None:
        print("test-remote-linux: the msys2 wrapper was not found", file=sys.stderr)
        return 1

    repo = Path(git_output(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    files = worktree_files(repo)
    commit = git_output(repo, "rev-parse", "HEAD").decode().strip()
    dirty = bool(
        git_output(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    )
    run_id = uuid.uuid4().hex
    incoming = f"{args.remote_root}/incoming/{run_id}"

    print(f"Synchronizing {len(files)} files to {args.host}:{incoming}", flush=True)
    manifest_path: Path | None = None
    incoming_created = False
    try:
        run(ssh_command(msys2, args.host, ["mkdir", "-p", "--", incoming]))
        incoming_created = True
        with tempfile.NamedTemporaryFile(
            prefix="qvl-remote-linux-", suffix=".files", delete=False
        ) as manifest:
            manifest.write(b"\0".join(files))
            manifest.write(b"\0")
            manifest_path = Path(manifest.name)

        run(
            [
                msys2,
                "rsync",
                "--recursive",
                "--links",
                "--checksum",
                "--compress",
                "--relative",
                "--from0",
                f"--files-from={msys_path(manifest_path)}",
                f"{msys_path(repo)}/",
                f"{args.host}:{incoming}/",
            ]
        )

        remote_arguments = [
            "bash",
            f"{incoming}/scripts/test-remote-linux.sh",
            incoming,
            args.remote_root,
            args.mode,
            str(jobs),
            commit,
            "1" if dirty else "0",
        ]
        return subprocess.run(
            ssh_command(msys2, args.host, remote_arguments), check=False
        ).returncode
    finally:
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)
        if incoming_created:
            subprocess.run(
                ssh_command(msys2, args.host, ["rm", "-rf", "--", incoming]),
                check=False,
            )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"test-remote-linux: {error}", file=sys.stderr)
        sys.exit(1)
