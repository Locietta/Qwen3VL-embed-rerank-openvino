#!/usr/bin/env python3
"""Claude Code PostToolUse hook: format a file in place after Edit/Write.

Reuses the same clang-format invocation as hooks/pre-commit, but since
this fires right after an edit rather than at commit time, it just
reformats the file silently instead of blocking and printing errors.
"""

import json
import os
import subprocess
import sys

EXTS_CLANG = {".cpp", ".h", ".hpp"}


def main():
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    tool_input = payload.get("tool_input") or {}
    tool_response = payload.get("tool_response") or {}
    filepath = tool_response.get("filePath") or tool_input.get("file_path")
    if not filepath or not os.path.isfile(filepath):
        return 0

    ext = os.path.splitext(filepath)[1].lower()
    if ext not in EXTS_CLANG:
        return 0

    subprocess.run(["clang-format", "-i", filepath], capture_output=True, timeout=30)
    return 0


if __name__ == "__main__":
    sys.exit(main())
