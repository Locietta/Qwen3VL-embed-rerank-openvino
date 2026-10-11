#!/usr/bin/env python3
"""Install project git hooks by symlinking hooks/ into .git/hooks/.

Usage:
    uv run scripts/setup_hooks.py
"""

import os
import subprocess
import sys


def main():
    repo_root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()

    src_dir = os.path.join(repo_root, "hooks")
    dst_dir = os.path.join(repo_root, ".git", "hooks")

    if not os.path.isdir(src_dir):
        print(f"error: {src_dir} not found", file=sys.stderr)
        return 1

    hooks = [f for f in os.listdir(src_dir) if not f.startswith(".")]
    if not hooks:
        print("No hooks to install.")
        return 0

    for name in hooks:
        src = os.path.join(src_dir, name)
        dst = os.path.join(dst_dir, name)

        # Remove existing hook (file or symlink) so we can replace it.
        if os.path.lexists(dst):
            os.remove(dst)

        # Use a relative symlink so the repo stays relocatable.
        rel = os.path.relpath(src, dst_dir)
        os.symlink(rel, dst)
        print(f"  {name} -> {rel}")

    print(f"\nInstalled {len(hooks)} hook(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
