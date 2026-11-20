#!/usr/bin/env bash
# Runs the MuJoCo tests from an unpacked bundle, inside an environment created next to this
# script so nothing outside this directory is touched: first every vehicle in a perfect world,
# then every vehicle under the imperfections it should expect. Extra arguments go to both.
#
# --video needs an OpenGL context without a display. Pick one the machine actually has:
#   MUJOCO_GL=egl ./run.sh --video      (GPU, needs libEGL)
#   MUJOCO_GL=osmesa ./run.sh --video   (CPU, needs libOSMesa)
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
"${PYTHON:-python3}" -m venv "$here/.venv"
"$here/.venv/bin/pip" install --quiet -r "$here/requirements.txt"
# The wheel keeps its version number between builds, so pip would otherwise keep the old one.
"$here/.venv/bin/pip" install --quiet --force-reinstall --no-deps "$here"/april_engine-*.whl
# A Jupyter kernel exports MPLBACKEND for its own inline plots; that backend is not installed
# here and the tests only write files, so the variable must not reach them.
python=(env -u MPLBACKEND "$here/.venv/bin/python")
"${python[@]}" "$here/mujoco_tests.py" --out "$here/results" "$@"
"${python[@]}" "$here/robustness.py" --out "$here/results/robustness" "$@"
