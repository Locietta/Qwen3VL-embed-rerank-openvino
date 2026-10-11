#!/usr/bin/env bash
set -euo pipefail

incoming_arg=${1:?usage: test-remote-linux.sh <incoming-root> <ci-root> <mode> <jobs> <commit> <dirty>}
ci_root_arg=${2:?usage: test-remote-linux.sh <incoming-root> <ci-root> <mode> <jobs> <commit> <dirty>}
mode=${3:?usage: test-remote-linux.sh <incoming-root> <ci-root> <mode> <jobs> <commit> <dirty>}
jobs=${4:?usage: test-remote-linux.sh <incoming-root> <ci-root> <mode> <jobs> <commit> <dirty>}
commit=${5:?usage: test-remote-linux.sh <incoming-root> <ci-root> <mode> <jobs> <commit> <dirty>}
dirty=${6:?usage: test-remote-linux.sh <incoming-root> <ci-root> <mode> <jobs> <commit> <dirty>}

absolute_home_path() {
    case "$1" in
        /*) realpath -m -- "$1" ;;
        *) realpath -m -- "$HOME/$1" ;;
    esac
}

ci_root=$(absolute_home_path "$ci_root_arg")
incoming_root=$(absolute_home_path "$incoming_arg")
case "$incoming_root" in
    "$ci_root"/incoming/*) ;;
    *)
        echo "test-remote-linux: refusing unexpected incoming path: $incoming_root" >&2
        exit 2
        ;;
esac
case "$mode" in
    normal | release | sanitize) ;;
    *)
        echo "test-remote-linux: unknown mode: $mode" >&2
        exit 2
        ;;
esac
case "$jobs" in
    '' | *[!0-9]* | 0)
        echo "test-remote-linux: jobs must be positive" >&2
        exit 2
        ;;
esac
case "$commit" in
    *[!0-9a-f]* | '')
        echo "test-remote-linux: invalid source commit" >&2
        exit 2
        ;;
esac
case "$dirty" in
    0 | 1) ;;
    *)
        echo "test-remote-linux: dirty state must be 0 or 1" >&2
        exit 2
        ;;
esac

cleanup() {
    case "$incoming_root" in
        "$ci_root"/incoming/*) rm -rf -- "$incoming_root" ;;
    esac
}
trap cleanup EXIT

mkdir -p "$ci_root/logs" "$ci_root/pixi" "$ci_root/worktrees"
exec 9>"$ci_root/test.lock"
if ! flock -n 9; then
    echo "Another remote Linux test is active; waiting for its build lock."
    flock 9
fi

source_root="$ci_root/worktrees/$mode"
mkdir -p "$source_root"
rsync \
    --recursive \
    --links \
    --checksum \
    --delete \
    --delete-delay \
    --exclude=/.pixi/ \
    --exclude=/.xmake/ \
    --exclude=/build\*/ \
    --exclude=/dist/ \
    --exclude=/tmp/ \
    "$incoming_root/" \
    "$source_root/"

if [[ ! -e "$source_root/.pixi" ]]; then
    ln -s "$ci_root/pixi" "$source_root/.pixi"
fi

if ! command -v pixi >/dev/null; then
    echo "test-remote-linux: pixi is not available on the remote PATH" >&2
    exit 1
fi
if ! command -v xmake >/dev/null; then
    echo "test-remote-linux: xmake is not available on the remote PATH" >&2
    exit 1
fi

export QVL_SOURCE_COMMIT=$commit
export QVL_SOURCE_DIRTY=$dirty
log_file="$ci_root/logs/$(date -u +%Y%m%dT%H%M%SZ)-$mode-${commit:0:12}.log"

cd "$source_root"
echo "Testing $commit$([[ $dirty == 1 ]] && printf '%s' -dirty) in $source_root"
echo "Mode: $mode; jobs: $jobs; log: $log_file"

build_and_test() {
    pixi run xmake build -y -j "$jobs"
    pixi run xmake test -j "$jobs"
}

{
    case "$mode" in
        normal)
            pixi run xmake f -m releasedbg --sanitizers=n -y
            build_and_test
            ;;
        release)
            pixi run xmake f -m release --sanitizers=n -y
            build_and_test
            ;;
        sanitize)
            pixi run xmake f -m releasedbg --sanitizers=y -y
            pixi run xmake build -y -j "$jobs"
            pixi run xmake test -j "$jobs" -vvvD
            ;;
    esac
} 2>&1 | tee "$log_file"
