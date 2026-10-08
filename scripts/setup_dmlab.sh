#!/usr/bin/env bash
# Build DeepMind Lab from the lab/ submodule and install it as the deepmind_lab Python module.
#
# Usage: bash scripts/setup_dmlab.sh [--python PATH] [--lab-dir DIR] [--skip-deps] [--skip-build]
#
#   --python PATH   interpreter to install into (default: env1's python via conda)
#   --lab-dir DIR   DeepMind Lab checkout to patch and build (default: lab/ submodule)
#   --skip-deps     do not check apt packages / bazelisk
#   --skip-build    do not run bazel or pip; only patch and smoke-test the installed module
#
# lab/ is pinned upstream DeepMind Lab; scripts/dmlab_patches/modern-toolchain.patch makes it
# build on GCC 14+/Bazel 6/Python 3.12+ (dependency pins, system glib, a libxml fix, a
# re-runnable vm.pk3 genrule, and an importlib-based loader). Applying it is idempotent.
# Run the long build inside tmux; it takes tens of minutes.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$REPO/scripts/dmlab_patches/modern-toolchain.patch"
LAB="$REPO/lab"
PY=""
SKIP_DEPS=0
SKIP_BUILD=0
APT_PACKAGES=(libffi-dev gettext freeglut3-dev libsdl2-dev zip libosmesa6-dev python3-dev
              python3-numpy python3-pil build-essential libglib2.0-dev)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --lab-dir) LAB="$(cd "$2" && pwd)"; shift 2 ;;
    --skip-deps) SKIP_DEPS=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PY" ]]; then
  PY="$(conda run -n env1 python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
  [[ -n "$PY" ]] || { echo "could not find env1; pass --python PATH" >&2; exit 1; }
fi
echo "target interpreter: $PY"

echo "=== [1/7] lab checkout ==="
if [[ "$LAB" == "$REPO/lab" ]]; then
  git -C "$REPO" submodule update --init lab
fi
git -C "$LAB" rev-parse HEAD

echo "=== [2/7] toolchain patch ==="
if git -C "$LAB" apply --reverse --check "$PATCH" 2>/dev/null; then
  echo "already applied"
elif git -C "$LAB" apply --check "$PATCH" 2>/dev/null; then
  git -C "$LAB" apply "$PATCH"
  echo "applied"
else
  echo "patch neither applies nor is already applied; lab/ has other local changes" >&2
  git -C "$LAB" status --short >&2
  exit 1
fi

echo "=== [3/7] system packages ==="
if [[ $SKIP_DEPS -eq 0 ]]; then
  missing=()
  for pkg in "${APT_PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing+=("$pkg")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "missing packages; install them yourself, then rerun:" >&2
    echo "  sudo apt-get install -y ${missing[*]}" >&2
    exit 1
  fi
  echo "all present"
else
  echo "skipped"
fi

echo "=== [4/7] bazelisk ==="
if [[ $SKIP_DEPS -eq 0 ]]; then
  if ! command -v bazel >/dev/null 2>&1; then
    mkdir -p "$HOME/.local/bin"
    curl -fsSL -o "$HOME/.local/bin/bazel" \
      https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64
    chmod +x "$HOME/.local/bin/bazel"
  fi
  echo "bazel: $(command -v bazel || echo "$HOME/.local/bin/bazel")"
else
  echo "skipped"
fi

if [[ $SKIP_BUILD -eq 0 ]]; then
  # Bazel's python_system repo rule probes whichever python3 is first on PATH, once per server.
  # Conda's base python3 has no numpy, so put the system interpreter (which apt's python3-numpy
  # installed into) first and restart the server.
  export PATH="$HOME/.local/bin:/usr/bin:/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/sbin"

  echo "=== [5/7] bazel build (long) ==="
  (cd "$LAB" && bazel shutdown || true)
  (cd "$LAB" && bazel build -c opt --python_version=PY3 //python/pip_package:build_pip_package)

  echo "=== [6/7] wheel + pip install ==="
  OUT="$(mktemp -d)"
  # Use bazel info rather than the ./bazel-bin convenience symlink, which an interrupted run can
  # leave as a stray zero-byte file.
  BAZEL_BIN="$(cd "$LAB" && bazel info bazel-bin -c opt --python_version=PY3)"
  (cd "$LAB" && "$BAZEL_BIN/python/pip_package/build_pip_package" "$OUT")
  "$PY" -m pip install "$(ls "$OUT"/*.whl | head -1)" --force-reinstall --no-deps
  "$PY" -m pip install dm_env
else
  echo "=== [5/7] bazel build: skipped ==="
  echo "=== [6/7] wheel + pip install: skipped ==="
fi

echo "=== [7/7] smoke test ==="
cd "$REPO"
"$PY" - <<'EOF'
import os
import numpy as np
import torch
import deepmind_lab

env = deepmind_lab.Lab("tests/empty_room_test", ["RGB_INTERLEAVED"],
                       config={"width": "84", "height": "84"})
env.reset()
for _ in range(20):
    env.step(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc), num_steps=4)
obs = env.observations()["RGB_INTERLEAVED"]
env.close()
print("built-in level OK:", obs.shape, obs.dtype)

env = deepmind_lab.Lab("square_arena", ["RGB_INTERLEAVED", "DEBUG.POS.TRANS"],
                       config={"width": "64", "height": "64",
                               "levelDirectory": os.path.abspath("levels")})
env.reset(seed=0)
env.step(np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc), num_steps=4)
env.close()
print("levels/square_arena.lua OK")
if torch.cuda.is_available():
    print("CUDA tensor OK:", torch.from_numpy(obs).float().cuda().mean().item())
print("SMOKE TEST PASSED")
EOF
