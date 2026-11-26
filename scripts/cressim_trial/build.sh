#!/usr/bin/env bash
# Compile the upstream CUDA engine and the small cucumber adapter.
set -euo pipefail
cd "$(dirname "$0")/../.."
rev=09aa5009b8580351f516b6df7660e87821fc5eb6
src=.cache/cressim-src
build=.cache/cressim-build
mkdir -p .cache/cressim-run
if [[ ! -d "$src/.git" ]]; then
    git clone https://github.com/yafei-ou/CRESSim-MPM.git "$src"
    git -C "$src" checkout --detach "$rev"
fi
[[ "$(git -C "$src" rev-parse HEAD)" == "$rev" ]] || { echo "Expected upstream $rev in $src" >&2; exit 1; }
cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CRESSIM_CUDA_ARCH:-89}" -DENGINE_STATIC=OFF -DENABLE_TESTS=OFF -DENABLE_EXAMPLES=OFF
cmake --build "$build" -j"${CRESSIM_JOBS:-6}"
includes=()
for d in "$src"/include "$src"/include/*/; do includes+=("-I$d"); done
g++ -O3 -std=c++17 -fPIC -shared "${includes[@]}" scripts/cressim_trial/bridge.cpp \
    -L"$build/bin" -Wl,-rpath,"$PWD/$build/bin" -lcrmpm_engine -o .cache/cressim-run/bridge.so
