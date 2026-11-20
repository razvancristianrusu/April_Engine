#!/usr/bin/env bash
# Packs the MuJoCo tests, the vehicle descriptions and a portable April Engine wheel into one
# archive for a machine with no Rust toolchain:
#   tar xzf april_engine_sim.tar.gz && sim/run.sh
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(dirname "$here")
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir "$stage/sim"
# zig links against an old glibc, so the wheel also loads on distributions older than this one.
(cd "$root/python" && uvx --from 'maturin[zig]' maturin build --release --zig \
    --compatibility manylinux_2_28 --out "$stage/sim")
cp "$here"/*.py "$here/requirements.txt" "$here/run.sh" "$stage/sim"
cp -r "$root/vehicles" "$stage/vehicles"
tar -czf "$root/target/april_engine_sim.tar.gz" -C "$stage" sim vehicles
echo "$root/target/april_engine_sim.tar.gz"
