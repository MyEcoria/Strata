#!/bin/sh
# Builds and runs tools/flint/kv_test.cu (the KV codecs on their own).  AMD (HIP) as in the build-mmq tree of docs/FLINT.md.
#   sh tools/flint/kv_test.sh [keys]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${KV_TEST_OUT:-/tmp/strata_kv_test}
CXX=${CXX:-/usr/lib/llvm-22/bin/clang++}
$CXX -O3 -std=c++20 --offload-arch=${GPU_ARCH:-gfx1201} -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 -DSTRATA_USE_HIP=1 \
  -I"$ROOT/include" -I"$ROOT/include/strata/hip_compat" -include "$ROOT/include/strata/hip_compat/cuda_runtime.h" \
  -x hip "$ROOT/tools/flint/kv_test.cu" -o "$OUT" -L/usr/lib/x86_64-linux-gnu -l:libamdhip64.so
"$OUT" "$@"
