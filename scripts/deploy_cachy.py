#!/usr/bin/env python3
"""Upload this worktree to the inference host, build a release there and install it.

The build happens on the host itself under ~/services/qvl/src so the binary
links against that tree's Pixi environment. Installing never restarts the
running service unless --restart is given.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from test_remote_linux import git_output, msys_path, run, ssh_command, worktree_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--host",
        default=os.environ.get("QVL_DEPLOY_HOST", "cachy"),
        help="MSYS2 SSH host alias (default: cachy, or QVL_DEPLOY_HOST)",
    )
    parser.add_argument(
        "--restart", action="store_true", help="restart qvl.service afterwards"
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="deploy uncommitted changes (the version is marked -dirty)",
    )
    args = parser.parse_args()

    msys2 = shutil.which("msys2")
    if msys2 is None:
        print("deploy: the msys2 wrapper was not found", file=sys.stderr)
        return 1

    repo = Path(git_output(Path.cwd(), "rev-parse", "--show-toplevel").decode().strip())
    dirty = bool(
        git_output(repo, "status", "--porcelain=v1", "--untracked-files=normal")
    )
    if dirty and not args.allow_dirty:
        print(
            "deploy: the worktree has uncommitted changes (use --allow-dirty)",
            file=sys.stderr,
        )
        return 1
    commit = git_output(repo, "rev-parse", "HEAD").decode().strip()
    files = worktree_files(repo)
    incoming = f"services/qvl/incoming/{uuid.uuid4().hex}"

    print(f"Uploading {len(files)} files to {args.host}:{incoming}", flush=True)
    manifest_path: Path | None = None
    try:
        run(ssh_command(msys2, args.host, ["mkdir", "-p", "--", incoming]))
        with tempfile.NamedTemporaryFile(
            prefix="qvl-deploy-", suffix=".files", delete=False
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
        remote = [
            "bash",
            f"{incoming}/scripts/deploy-remote.sh",
            incoming,
            commit,
            "1" if dirty else "0",
            "1" if args.restart else "0",
        ]
        return subprocess.run(
            ssh_command(msys2, args.host, remote), check=False
        ).returncode
    finally:
        if manifest_path is not None:
            manifest_path.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        print(f"deploy: {error}", file=sys.stderr)
        sys.exit(1)
