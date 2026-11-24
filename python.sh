#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$root/.cache/home" "$root/.cache/kit/"{cache,data,logs}
exec docker run --rm --ulimit core=0 --network none --runtime=nvidia --gpus all --env NVIDIA_DRIVER_CAPABILITIES=all --env XDG_CACHE_HOME=/workspace/.cache --user "$(id -u):$(id -g)" --group-add 1234 --env HOME=/workspace/.cache/home \
  --mount "type=bind,src=$root,dst=/workspace" --workdir /workspace \
  --mount "type=bind,src=$root/.cache/kit/cache,dst=/isaac-sim/kit/cache" \
  --mount "type=bind,src=$root/.cache/kit/data,dst=/isaac-sim/kit/data" \
  --mount "type=bind,src=$root/.cache/kit/logs,dst=/isaac-sim/kit/logs" \
  --entrypoint /bin/bash "${MPM_IMAGE:-fruit-cutting-research:local}" -c '
  libs=(/isaac-sim/extscache/omni.usd.libs-*)
  export PYTHONPATH="/workspace/src:/workspace/third_party/EEF-Cutting-Simulation:${libs[0]}${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="${libs[0]}/bin:${LD_LIBRARY_PATH:-}"
  export OPENBLAS_NUM_THREADS=1
  export TI_OFFLINE_CACHE_FILE_PATH=$PWD/.cache/taichi
  exec /isaac-sim/python.sh "$@"' bash "$@"
