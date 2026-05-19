#!/usr/bin/env bash
# Bootstrap a local build environment on Linux when you do NOT have sudo.
#
# Background: `manimpango` (a hard dep of `manim`) publishes no Linux wheels
# on PyPI — it must be compiled from source, which requires Pango, Cairo,
# pkg-config and the CPython development headers. On a shared cluster you
# usually cannot apt-install those system-wide. The workaround here is to
# materialise them inside a conda environment (which is fully user-local),
# and then drive Python dependency management with uv on top of that env.
#
# Usage:
#   ./scripts/bootstrap-linux.sh           # creates env $KL3D_ENV (default: kl3d)
#   KL3D_ENV=foo ./scripts/bootstrap-linux.sh
#
# After it finishes, activate the env and `uv sync`:
#   micromamba activate kl3d                 # or: conda/mamba activate kl3d
#   uv sync --extra gui
#   uv run manim-slides render -ql kl_3d.py KLDivergence3D

set -euo pipefail

ENV_NAME="${KL3D_ENV:-kl3d}"

if command -v micromamba >/dev/null 2>&1; then
    CONDA=micromamba
elif command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
else
    echo "error: need micromamba, mamba or conda on PATH" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH" >&2
    echo "       install it from https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

echo "==> using $CONDA to create env '$ENV_NAME' from conda-forge"
"$CONDA" create -n "$ENV_NAME" -c conda-forge -y \
    "python=3.11" \
    cairo \
    pango \
    pkg-config \
    ffmpeg

cat <<EOF

==> done.
   Next steps:
     $CONDA activate $ENV_NAME
     uv sync --extra gui
     uv run manim-slides render -ql kl_3d.py KLDivergence3D
EOF
